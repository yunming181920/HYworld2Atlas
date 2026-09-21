"""gs_bridge shared data types (both sides, pure torch/numpy, no heavy deps)."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CameraPose:
    """A single camera pose in HY world coordinates (w2c convention source side).

    Attributes:
        c2w: (4, 4) camera-to-world matrix, OpenCV convention (x-right, y-down, z-forward).
        fxfycxcy: (4,) normalized intrinsics (fx, fy, cx, cy), all in [0, 1] units of image width.
        latent_idx: index of the latent this pose belongs to (int).
        chunk_idx: index of the chunk this pose belongs to (int).
        frame_idx: pixel-frame index if this pose was interpolated (Optional[int]).
    """
    c2w: np.ndarray
    fxfycxcy: np.ndarray
    latent_idx: int
    chunk_idx: int
    frame_idx: Optional[int] = None


@dataclass
class TrajectorySync:
    """One-time startup message: full precomputed trajectory in MoVieS convention.

    Built from HY's `pose_to_input` output (w2c latents + normalized Ks), converted to
    c2w + fxfycxcy per latent, with per-frame interpolation done lazily on the server side
    via `camera_conv.interpolate_pose` (the latent table here is the interpolation anchor).

    Attributes:
        c2w_latents: (L, 4, 4) c2w per latent, HY world frame.
        fxfycxcy: (L, 4) normalized intrinsics per latent.
        num_latents: L.
    """
    c2w_latents: np.ndarray
    fxfycxcy: np.ndarray
    num_latents: int

    def __post_init__(self):
        assert self.c2w_latents.shape[0] == self.num_latents
        assert self.fxfycxcy.shape[0] == self.num_latents
        assert self.c2w_latents.shape[1:] == (4, 4)
        assert self.fxfycxcy.shape[1] == 4


@dataclass
class ChunkPacket:
    """GPU A -> GPU B update packet: decoded pixel frames of one finished chunk.

    Poses are NOT included: the server already holds the trajectory table (TrajectorySync)
    and derives this chunk's per-frame poses via SE3 interpolation on `chunk_idx`.

    Attributes:
        chunk_idx: int, the finished chunk index (version becomes chunk_idx + 1 in latents).
        frames: (F, 3, H, W) float32 RGB in [0, 1], F = 16 for HY AR chunks.
        timestamps: (F,) float32 global pixel-frame timestamps (seconds or frame counts,
            consistent across all packets; server normalizes per-window).
    """
    chunk_idx: int
    frames: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self):
        assert self.frames.ndim == 4 and self.frames.shape[1] == 3
        assert self.frames.shape[0] == self.timestamps.shape[0]


@dataclass
class AssetQuery:
    """GPU A -> GPU B query: render memory views from the asset.

    Attributes:
        version_cap: only windows with version <= version_cap are visible.
            The duplex pipeline passes (current_chunk - 2): while chunk t is being denoised,
            chunk t-1's asset update may still be running, so only t-2 is guaranteed complete.
        latent_indices: (V,) latent indices to render (memory slot poses, from the
            precomputed trajectory; rope/attention alignment is per-latent).
        render_height: output render resolution (480p direct-out by default).
        render_width: output render resolution.
        with_depth: whether to also return rendered depths.
    """
    version_cap: int
    latent_indices: np.ndarray
    render_height: int = 480
    render_width: int = 832
    with_depth: bool = True


@dataclass
class AssetRender:
    """GPU B -> GPU A query response.

    Attributes:
        images: (V, 3, H, W) float32 RGB in [0, 1].
        depths: (V, 1, H, W) float32 metric depth in HY world scale, or None.
        version: asset version actually used (<= version_cap of the query).
        latent_indices: echo of the query indices, aligned with images.
    """
    images: np.ndarray
    depths: Optional[np.ndarray]
    version: int
    latent_indices: np.ndarray

    def __post_init__(self):
        assert self.images.ndim == 4 and self.images.shape[1] == 3
        assert self.latent_indices.shape[0] == self.images.shape[0]
        if self.depths is not None:
            assert self.depths.shape[0] == self.images.shape[0]
