"""MoVieS backbone wrapper for gs_bridge (GPU B side).

Encapsulates the MoVieS `SplatRecon` model behind a minimal interface:
    encode_window(frames, c2w_frames, fxfycxcy, timestamps) -> WindowGaussians

Handles the three known MoVieS loading pitfalls (see CLAUDE.md "已知坑"):
    1. `src/options.py` asserts `./resources` exists at import time (module-level opt_dict).
    2. `VGGSplaT.__init__` unconditionally downloads VGGT-1B from HF (vggt_init=True),
       even for pure inference where the ckpt fully overwrites it afterwards.
    3. The import chain (`src.utils` -> wandb/accelerate, `src.models` -> diffusers).

This module must run inside the MoVieS conda environment with the MoVieS repo root
as a sibling of gs_bridge (config.movies_repo). NOT importable in the base env.

Usage:
    enc = MoVieSEncoder(movies_repo="MoVieS", ckpt="MoVieS/resources/movies_ckpt.safetensors")
    out = enc.encode_window(frames, c2w, fxfy, timestamps)
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import MoVieS with the pitfalls patched (must happen before `src.options` import)
# ---------------------------------------------------------------------------

_MOVIES_IMPORTED = False


def _ensure_movies_importable(movies_repo: str) -> None:
    """Patch the MoVieS import chain so `src.options` / `src.models` load cleanly.

    Patches:
        - creates <repo>/resources if missing (options.py __post_init__ asserts it)
        - appends <repo> and <repo>/extensions/vggt to sys.path
    """
    global _MOVIES_IMPORTED
    if _MOVIES_IMPORTED:
        return
    repo = os.path.abspath(movies_repo)
    assert os.path.isdir(repo), f"MoVieS repo not found: {repo}"

    # Pitfall 1: resources dir must exist for Options.__post_init__
    resources = os.path.join(repo, "resources")
    os.makedirs(resources, exist_ok=True)

    # sys.path: repo root (for `src.`), extensions/vggt (for vggt package)
    for p in (repo, os.path.join(repo, "extensions", "vggt")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    _MOVIES_IMPORTED = True


@dataclass
class WindowGaussians:
    """One encoded MoVieS window (chunk), in the window's canonical frame.

    Fields mirror MoVieS backbone outputs (see src/models/networks/vggsplat.py:261-333):
        model_outputs: dict of (B=1, F_in, C, H, W): depth/depth_conf/color/scale/
            rotation/opacity/conf (+ offset, motion_* merged at query time)
        pred_motions: (1, F_out, F_in, 3+1, H, W) per-output-timestep motion offsets
        pred_motion_gs: list of F_out dict of (1, F_in, C, H, W) time-conditioned attrs
        input_c2w: (1, F_in, 4, 4) canonical c2w actually fed to the backbone
        input_fxfycxcy: (1, F_in, 4) normalized intrinsics
        input_timesteps: (1, F_in) window-local [0, 1]
        output_timesteps: (1, F_out) window-local [0, 1]
    Plus the HY-side bookkeeping:
        T_canon2hy: (4, 4) window canonical -> HY world
        scale_canon2hy: float, canonical unit -> HY world unit
        chunk_idx: int
    """
    model_outputs: Dict[str, torch.Tensor]
    pred_motions: Optional[torch.Tensor]
    pred_motion_gs: Optional[List[Dict[str, torch.Tensor]]]
    input_c2w: torch.Tensor
    input_fxfycxcy: torch.Tensor
    input_timesteps: torch.Tensor
    output_timesteps: torch.Tensor
    T_canon2hy: np.ndarray
    scale_canon2hy: float
    chunk_idx: int


class MoVieSEncoder:
    """Loads MoVieS once and encodes chunk windows feed-forward.

    The encoder applies MoVieS' canonical normalization itself (first window pose ->
    identity) instead of relying on the dataset pipeline, and records the analytic
    inverse (T_canon2hy, scale) in each WindowGaussians -- window -> HY world is then
    a known transform, not an estimation problem (DESIGN.md section 0).
    """

    def __init__(
        self,
        movies_repo: str = "MoVieS",
        ckpt: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        frames_chunk_size: int = 16,
        vggt_local: Optional[str] = None,
    ):
        _ensure_movies_importable(movies_repo)
        from src.options import opt_dict
        from src.models import SplatRecon
        from safetensors.torch import load_file

        self.device = device
        self.dtype = dtype
        self.frames_chunk_size = frames_chunk_size

        opt = opt_dict["movies"]
        # Pitfall 2 (VGGT-1B download): vggt_init downloads facebook/VGGT-1B at build.
        # If a local VGGT-1B snapshot exists, point HF_HOME at it; otherwise let it
        # download once (cached). `vggt_local` may be a HF cache dir containing
        # hub/models--facebook--VGGT-1B/.
        if vggt_local is not None:
            os.environ.setdefault("HF_HOME", vggt_local)

        model = SplatRecon(opt, load_lpips=False)
        if ckpt is not None and os.path.exists(ckpt):
            state = load_file(ckpt, device="cpu")
            model.load_state_dict(state, strict=True)
        elif ckpt is not None:
            raise FileNotFoundError(f"MoVieS ckpt not found: {ckpt}")
        # else: no ckpt -> random init (for shape/dry-run tests only)
        self.model = model.to(device).eval()
        self._opt = opt

        # size constraints from options (input_res / size_divisor)
        self.encode_res = opt.input_res  # (224, 224)
        self.size_divisor = opt.size_divisor  # 14

    # ------------------------------------------------------------------
    # preprocessing helpers
    # ------------------------------------------------------------------

    def _resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(F, 3, H, W) in [0,1] -> (F, 3, h, w) at MoVieS encode resolution.

        Crops-to-then-resizes (like the paper's evaluation protocol) is NOT applied;
        we do a plain bilinear resize of the full frame -- HY frames are the world
        the asset must represent, cropping would cut recalled FOV content.
        Aspect ratio deviates from training distribution (832:480 vs 1:1) -- an
        accepted risk for v1, revisit at M6 if encode quality suffers.
        """
        F, C, H, W = frames.shape
        h, w = self.encode_res
        if (H, W) == (h, w):
            return frames
        return torch.nn.functional.interpolate(
            frames, size=(h, w), mode="bilinear", align_corners=False
        )

    def _normalize_window(
        self, c2w_frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """HY-world window poses -> canonical (first pose identity).

        Returns (c2w_canonical (F,4,4), T_canon2hy (4,4)).
        NOTE: MoVieS' dataset pipeline also applies a SCALE normalization
        (`norm_xyz=True`: mean |point| -> camera_norm_unit). We intentionally do NOT
        rescale the scene: HY world units are already sane (w=0.08/frame, chunk
        spans ~2.5 units, znear/zfar 0.001/100 covers it). If encode quality
        degrades, revisit by recording scale_canon2hy != 1 here.
        """
        first_inv = np.linalg.inv(c2w_frames[0])
        c2w_can = np.einsum("ij,fjk->fik", first_inv, c2w_frames)
        return c2w_can, c2w_frames[0]

    # ------------------------------------------------------------------
    # main encode API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_window(
        self,
        frames: torch.Tensor,
        c2w_frames: np.ndarray,
        fxfycxcy: np.ndarray,
        global_timestamps: np.ndarray,
        chunk_idx: int,
    ) -> WindowGaussians:
        """Encode one HY chunk (16 pixel frames) into a MoVieS window.

        Args:
            frames: (F, 3, H, W) float in [0, 1], HY decode output (any resolution).
            c2w_frames: (F, 4, 4) HY-world c2w of each frame (SE3-interpolated).
            fxfycxcy: (4,) or (F, 4) normalized intrinsics (HY convention -> already
                converted by camera_conv.hy_ks_to_movies_fxfycxcy).
            global_timestamps: (F,) global frame timestamps; normalized to [0, 1]
                window-locally here (MoVieS convention).
            chunk_idx: int, for bookkeeping.

        Returns:
            WindowGaussians on self.device (bf16 outputs stay bf16; bookkeeping numpy).
        """
        assert frames.ndim == 4 and frames.shape[1] == 3
        F = frames.shape[0]
        assert c2w_frames.shape == (F, 4, 4)
        if fxfycxcy.ndim == 1:
            fxfycxcy = np.repeat(fxfycxcy[None], F, axis=0)
        assert fxfycxcy.shape == (F, 4)

        # 1. resize to encode resolution
        frames_enc = self._resize_frames(frames)
        h, w = frames_enc.shape[-2:]

        # 2. canonical normalization (record inverse)
        c2w_can, T_canon2hy = self._normalize_window(c2w_frames)

        # 3. window-local timesteps in [0, 1]
        ts = np.asarray(global_timestamps, dtype=np.float64)
        ts_local = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-8)

        # output timesteps: same grid as input (query any time later by re-running
        # the motion heads is NOT possible without the aggregator tokens; v1 renders
        # only at the input grid -- chunk boundary frames are exactly on-grid)
        t_in = torch.tensor(ts_local, dtype=torch.float32).unsqueeze(0)
        t_out = t_in.clone()

        device, dtype = self.device, self.dtype
        images = frames_enc.unsqueeze(0).to(device=device, dtype=dtype)
        C2W = torch.from_numpy(c2w_can).float().unsqueeze(0).to(device=device, dtype=dtype)
        FXFY = torch.from_numpy(fxfycxcy).float().unsqueeze(0).to(device=device, dtype=dtype)
        T_IN = t_in.to(device=device, dtype=dtype)
        T_OUT = t_out.to(device=device, dtype=dtype)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)):
            model_outputs, pred_motions, pred_motion_gs = self.model.backbone(
                images, C2W, FXFY, T_IN, T_OUT,
                frames_chunk_size=self.frames_chunk_size,
            )

        return WindowGaussians(
            model_outputs=model_outputs,
            pred_motions=pred_motions,
            pred_motion_gs=pred_motion_gs,
            input_c2w=C2W,
            input_fxfycxcy=FXFY,
            input_timesteps=t_in,
            output_timesteps=t_out,
            T_canon2hy=T_canon2hy,
            scale_canon2hy=1.0,
            chunk_idx=chunk_idx,
        )

    def gaussian_count(self, window: WindowGaussians) -> int:
        """N = F_in * H * W (before opacity pruning at render time)."""
        depth = window.model_outputs["depth"]
        return depth.shape[1] * depth.shape[-2] * depth.shape[-1]
