"""Rendered RGB -> HY VAE context latents (GPU A side, HY environment).

The DiT consumes history as *latents* (pipeline: `context_latents = latents[:, :,
selected_frame_indices]`). The original memory frames were latent-encoded by the
VAE during the AR rollout's first-chunk pass. Our replacement path re-encodes
rendered RGB views through the SAME causal 3D VAE so the DiT sees a numerically
compatible context.

Key subtlety: HunyuanVideo-1.5 VAE is causal-3D -- a latent depends on the frames
BEFORE it (first-frame + causal temporal convs). A standalone encode of a rendered
view sequence only matches the rollout's internal latents if the causal prefix
matches. v1 strategy: encode each queried window of views together with its
natural prefix (the chunk boundary frames), per the time_map latent structure --
in practice we encode the query's rendered sequence as one [1+4k] frame block,
mirroring how HY encodes chunk 0 (frame 0 + 16-frame groups).
"""

import numpy as np
import torch

from gs_bridge.common.time_map import frame_to_latent


class LatentPrep:
    """Wraps the HY pipeline's VAE encoder for rendered memory views."""

    def __init__(self, pipe):
        """pipe: HunyuanVideo_1_5_Pipeline (must carry .vae)."""
        self.vae = pipe.vae
        self.vae_scale = getattr(pipe, "vae_scale_factor_spatial", 8) or 8
        # HY VAE temporal compression: 4 frames -> 1 latent, first frame separate
        self.temporal_compression = 4

    @torch.no_grad()
    def encode_views(self, images: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
        """(V, 3, H, W) float32 [0,1] numpy -> latent tensor (1, C, L, h, w).

        Encodes the sequence as one causal block. V should follow the [1 + 4k]
        frame structure of HY video (frame 0 extra); a general V works but the
        last partial group is zero-padded to a multiple of 4 (the pad latents are
        returned too -- caller drops what it does not need).
        """
        assert images.ndim == 4 and images.shape[1] == 3
        V = images.shape[0]
        # pad to 1 + 4k
        if V == 1:
            padded = 1
        else:
            groups = (V - 1 + 3) // 4
            padded = 1 + groups * 4
        if padded != V:
            pad = np.repeat(images[-1:], padded - V, axis=0)
            images = np.concatenate([images, pad], axis=0)

        x = torch.from_numpy(images).float().unsqueeze(0).to(device)  # (1,V,3,H,W)
        x = (x - 0.5) / 0.5  # [0,1] -> [-1,1], HY VAE input convention

        # HY VAE encode expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            latents = self.vae.encode(x)
        if hasattr(latents, "latent_dist"):
            latents = latents.latent_dist.sample()
        latents = latents.to(dtype)
        # HY convention: shift factor + scale (see pipeline _encode_video or vae config)
        shift = getattr(self.vae.config, "shift_factor", None)
        scale = getattr(self.vae.config, "scaling_factor", None)
        if shift is not None and scale is not None:
            latents = (latents - shift) / scale
        return latents  # (1, C, L, h, w) with L = (padded-1)//4 + 1

    @staticmethod
    def latent_count(frame_count: int) -> int:
        """Frames -> latents: 1 + ceil((V-1)/4)."""
        if frame_count <= 1:
            return frame_count
        return 1 + (frame_count - 1 + 3) // 4

    @staticmethod
    def drop_pad_latents(latents: torch.Tensor, wanted_latents: int) -> torch.Tensor:
        """Trim encoded latents down to the wanted count (pad-group removal)."""
        L = latents.shape[2]
        if L <= wanted_latents:
            return latents
        return latents[:, :, :wanted_latents]
