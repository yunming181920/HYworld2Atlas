"""Query the global asset: render memory views at (latent pose, version cap).

The render pipeline per query (mirrors src/infer_davis_nvs.py:84-96):
    1. resolve query latent indices -> windows (version-capped)
    2. window-local: bake the motion offsets + time-conditioned attributes for the
       query time into model_outputs (time is NOT a renderer input in MoVieS)
    3. transform the query pose (HY world) into the window's canonical frame
    4. gs_renderer.render(... height, width) at the query resolution (decoupled
       from encode resolution)

Returns RGB (+ optional depth). Depth comes back in the window's canonical scale;
query.py maps it back to HY world scale via T_canon2hy (rigid part; scale=1 in v1).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gs_bridge.asset_server.asset import GlobalAsset, WindowSlot
from gs_bridge.asset_server.encoder import WindowGaussians
from gs_bridge.common.camera_conv import canonical_normalize


class AssetQueryResult:
    __slots__ = ("images", "depths", "version", "latent_indices", "chunk_ids")

    def __init__(self, images, depths, version, latent_indices, chunk_ids):
        self.images = images            # (V, 3, H, W) float32 numpy in [0,1]
        self.depths = depths            # (V, 1, H, W) float32 numpy or None
        self.version = version          # asset version actually used
        self.latent_indices = latent_indices  # (V,) echo
        self.chunk_ids = chunk_ids      # (V,) which window served each view


class AssetQuerier:
    """Renders memory views from the GlobalAsset on the MoVieS device."""

    def __init__(
        self,
        asset: GlobalAsset,
        renderer,                       # MoVieS GaussianRenderer (model.gs_renderer)
        trajectory,                     # TrajectorySync (latent pose table, MoVieS conv)
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.asset = asset
        self.renderer = renderer
        self.traj = trajectory
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # per-view render
    # ------------------------------------------------------------------

    def _bake_time(self, window: WindowGaussians, time_idx: int) -> Dict[str, torch.Tensor]:
        """model_outputs with motion baked for output timestep `time_idx`.

        Follows infer_davis_nvs.py:86-88: offset <- pred_motions[:, i, :, :3],
        update(pred_motion_gs[i]). Returns a SHALLOW-COPIED dict so the window's
        stored model_outputs is never mutated (windows are immutable).
        """
        mo = dict(window.model_outputs)  # shallow copy
        if window.pred_motions is not None:
            mo["offset"] = window.pred_motions[:, time_idx, :, :3, ...]
        if window.pred_motion_gs is not None:
            mo.update(window.pred_motion_gs[time_idx])
        return mo

    def _query_pose_to_canonical(
        self, c2w_query_hy: np.ndarray, window: WindowGaussians
    ) -> np.ndarray:
        """HY-world query pose -> window canonical frame.

        canonical = inv(c2w_first_hy) @ c2w  (camera_norm_type="canonical").
        We recompute from T_canon2hy (= c2w_first_hy) instead of storing the forward
        transform to keep WindowGaussians minimal: c2w_can = inv(T_canon2hy) @ c2w.
        """
        T_hy2canon = np.linalg.inv(window.T_canon2hy)
        return T_hy2canon @ c2w_query_hy

    def _render_one(
        self,
        window: WindowSlot,
        c2w_query_hy: np.ndarray,
        fxfycxcy_query: np.ndarray,
        time_idx: int,
        height: int,
        width: int,
        with_depth: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        w = window.window
        mo = self._bake_time(w, time_idx)

        c2w_can = self._query_pose_to_canonical(c2w_query_hy, w)
        device, dtype = self.device, self.dtype

        C2W_q = (
            torch.from_numpy(c2w_can).float().unsqueeze(0).unsqueeze(0)
            .to(device=device, dtype=dtype)
        )  # (1, 1, 4, 4)
        FXFY_q = (
            torch.from_numpy(np.asarray(fxfycxcy_query)).float().unsqueeze(0).unsqueeze(0)
            .to(device=device, dtype=dtype)
        )  # (1, 1, 4)

        with torch.autocast(
            self.device if self.device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=(dtype == torch.bfloat16 and self.device.startswith("cuda")),
        ):
            out = self.renderer.render(
                mo,                                   # baked model_outputs
                w.input_c2w,                          # input (canonical) C2W
                w.input_fxfycxcy,                     # input intrinsics
                C2W_q,                                # target canonical C2W
                FXFY_q,                               # target intrinsics
                height=height,
                width=width,
            )

        img = out["image"][0, 0].float().clamp(0, 1).cpu().numpy()  # (3, H, W)
        dep = None
        if with_depth and "depth" in out:
            dep = out["depth"][0, 0, 0].float().cpu().numpy()[None]  # (1, H, W)
        return img, dep

    # ------------------------------------------------------------------
    # public query API
    # ------------------------------------------------------------------

    def query(
        self,
        latent_indices: np.ndarray,
        version_cap: Optional[int] = None,
        height: int = 480,
        width: int = 832,
        with_depth: bool = True,
    ) -> Optional[AssetQueryResult]:
        """Render memory views for query latents, honoring the version cap.

        For each latent index: pick its covering window (version-capped), bake the
        window-local time nearest the latent's global time, render at the latent's
        trajectory pose. Latents not covered by any visible window (rollout start)
        are rendered by the temporally-nearest window (find_window_for_latent
        fallback) -- callers should treat early views as low-confidence.
        """
        indices = np.asarray(latent_indices, dtype=int)
        assert indices.ndim == 1
        if self.asset.visible_slots(version_cap) is None and len(self.asset) == 0:
            return None
        if len(self.asset) == 0:
            return None

        # resolve all latents -> (window, time_idx) first (consistency snapshot)
        plan: List[Tuple[WindowSlot, int, int]] = []  # (slot, time_idx, latent)
        for li in indices:
            slot = self.asset.find_window_for_latent(int(li), version_cap)
            if slot is None:
                return None  # asset empty under this cap
            # window-local time index nearest the latent's frame time
            time_idx = self._nearest_time_index(slot.window, int(li))
            plan.append((slot, time_idx, int(li)))

        version_used = max(s.version for s, _, _ in plan)
        images, depths, chunk_ids = [], [], []
        for slot, time_idx, li in plan:
            # query pose from the trajectory table (already MoVieS convention, HY world)
            c2w_q = self.traj.c2w_latents[li]
            fxfy_q = self.traj.fxfycxcy[min(li, self.traj.num_latents - 1)]
            img, dep = self._render_one(
                slot, c2w_q, fxfy_q, time_idx, height, width, with_depth
            )
            images.append(img)
            if dep is not None:
                depths.append(dep)
            chunk_ids.append(slot.chunk_idx)

        images = np.stack(images, axis=0)  # (V, 3, H, W)
        depths = np.stack(depths, axis=0) if (with_depth and depths) else None
        self.asset.stats["queries"] += 1
        return AssetQueryResult(
            images=images,
            depths=depths,
            version=version_used,
            latent_indices=indices,
            chunk_ids=np.array(chunk_ids, dtype=int),
        )

    def _nearest_time_index(self, window: WindowGaussians, latent_idx: int) -> int:
        """Map a global latent to the window's output-timestep grid index.

        Window covers latents [latent_start, latent_end) -> local fraction
        u = (latent_idx - latent_start) / (latent_end - latent_start) in [0, 1],
        snapped to the nearest output timestep index.
        """
        n_out = window.output_timesteps.shape[1]
        chunk_l = window.chunk_idx * (n_out)  # window spans n_out == F_in latents? No:
        # window covers chunk_latent_frames (4) latents; n_out == F_in == frames.
        # Map by latent fraction within the chunk:
        start = window.chunk_idx * 4  # chunk_latent_frames = 4 (v1)
        end = start + 4
        u = (latent_idx - start) / (end - start)
        u = min(max(u, 0.0), 1.0)
        t_grid = np.linspace(0.0, 1.0, n_out)
        return int(np.argmin(np.abs(t_grid - u)))
