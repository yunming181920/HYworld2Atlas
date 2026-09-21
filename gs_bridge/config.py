"""gs_bridge configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BridgeConfig:
    # --- trajectory / chunk ---
    num_latents: int = 32                 # 'w-31' -> 32 latents
    chunk_latent_frames: int = 4          # HY AR chunk = 4 latents
    hy_resolution: tuple = (480, 832)     # (H, W) HY output

    # --- asset ---
    asset_capacity_windows: int = 20      # ring buffer size, aligned with memory_frames=20
    query_pose_mode: str = "future"       # "future" (v1) | "fov_kept" (ablation)

    # --- MoVieS encoding ---
    encode_resolution: int = 224          # square encode res (divisor 14)
    encode_timestep_stride: int = 1       # feed every pixel frame

    # --- query rendering ---
    render_height: int = 480
    render_width: int = 832
    render_with_depth: bool = True

    # --- duplex timing ---
    version_cap_offset: int = 1           # while denoising chunk t, cap = (t-1) - offset + 1
                                           # idea.txt: t uses t-2 asset -> see time_map.window_version

    # --- IPC (M4) ---
    server_host: str = "127.0.0.1"
    server_port: int = 9820
    ipc_timeout_s: float = 120.0

    # --- paths ---
    movies_repo: str = "MoVieS"
    movies_ckpt: str = "MoVieS/resources/movies_ckpt.safetensors"
    hy_repo: str = "HY-WorldPlay"
