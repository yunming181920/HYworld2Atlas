"""GPU A client: HY process talks to the asset server over IPC.

Runs INSIDE the HY conda environment (no MoVieS imports here). Responsibilities:
    - trajectory sync at startup (from pose_to_input's viewmats/Ks)
    - push each finished chunk's decoded frames (CHUNK_UPDATE)
    - pull memory views before each chunk's denoising (QUERY_REQ)
    - optional Atlas observation (ATLAS_REQ)

The client deliberately knows nothing about gaussians -- it trades pixels for
rendered views. Memory-latent integration lives in memory_replace.py / latent_prep.py.
"""

import os
import socket
import time
from typing import List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from gs_bridge.common import ipc_protocol as ipc
from gs_bridge.common.camera_conv import hy_ks_to_movies_fxfycxcy, hy_w2c_to_c2w
from gs_bridge.common.types import AssetRender


class AssetClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9820,
                 timeout: float = 120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.synced = False

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------

    def sync_trajectory(self, viewmats: np.ndarray, Ks: np.ndarray) -> int:
        """Send the full latent pose table once. viewmats: (L,4,4) w2c; Ks: (L,3,3) HY-normalized."""
        c2w = hy_w2c_to_c2w(viewmats)
        aspect = None  # centered principal point: conversion needs no aspect
        fxfy = hy_ks_to_movies_fxfycxcy(Ks, aspect)
        hdr, body = ipc.trajectory_sync_msg(c2w, fxfy)
        ipc.send_msg(self.sock, hdr, body)
        rh, _ = ipc.recv_msg(self.sock)
        assert rh["type"] == "ACK", f"trajectory sync failed: {rh}"
        self.synced = True
        return len(c2w)

    # ------------------------------------------------------------------
    # update path (per finished chunk)
    # ------------------------------------------------------------------

    def push_chunk(self, chunk_idx: int, frames: np.ndarray) -> int:
        """frames: (F, 3, H, W) float32 [0,1] decoded pixel frames of the chunk.

        Timestamps are implicit (server derives them); we ship frame indices as ts.
        Returns the server-reported asset version (chunk_idx + 1).
        """
        assert self.synced, "sync_trajectory first"
        assert frames.dtype == np.float32 and frames.ndim == 4
        ts = np.arange(frames.shape[0], dtype=np.float64)
        hdr, body = ipc.chunk_update_msg(chunk_idx, frames, ts)
        ipc.send_msg(self.sock, hdr, body)
        rh, _ = ipc.recv_msg(self.sock)
        if rh.get("error"):
            raise RuntimeError(f"chunk update failed: {rh}")
        assert rh["type"] == "ACK", f"unexpected response: {rh}"
        return rh["version"]

    # ------------------------------------------------------------------
    # query path (before denoising each chunk)
    # ------------------------------------------------------------------

    def query_views(self, version_cap: int, latent_indices: List[int],
                    height: int = 480, width: int = 832,
                    with_depth: bool = True) -> Optional[AssetRender]:
        """Render memory views for the given latents under the version cap."""
        assert self.synced, "sync_trajectory first"
        hdr, body = ipc.query_req_msg(
            version_cap, np.asarray(latent_indices, dtype=int), height, width, with_depth
        )
        ipc.send_msg(self.sock, hdr, body)
        rh, rb = ipc.recv_msg(self.sock)
        if rh.get("error") == "empty_asset":
            return None
        if rh.get("error"):
            raise RuntimeError(f"query failed: {rh}")
        images, depths, version = ipc.decode_query_resp(rh, rb)
        return AssetRender(
            images=images,
            depths=depths,
            version=version,
            latent_indices=np.array(rh["latent_indices"], dtype=int),
        )

    # ------------------------------------------------------------------
    # atlas path (frozen-time observation)
    # ------------------------------------------------------------------

    def atlas_view(self, c2w: np.ndarray, fxfycxcy: np.ndarray,
                   height: int = 480, width: int = 832,
                   time_fraction: float = 1.0,
                   version_cap: Optional[int] = None) -> Optional[np.ndarray]:
        """Free-camera observation at frozen time. Returns (3,H,W) or None if asset empty."""
        hdr, body = ipc.atlas_req_msg(version_cap, c2w, fxfycxcy, height, width, time_fraction)
        ipc.send_msg(self.sock, hdr, body)
        rh, rb = ipc.recv_msg(self.sock)
        if rh.get("error") == "empty_asset":
            return None
        images, _ = ipc.decode_atlas_resp(rh, rb)
        return images[0]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        ipc.send_msg(self.sock, *ipc.control_msg("PING"))
        rh, _ = ipc.recv_msg(self.sock)
        return rh["type"] == "PING_RESP"

    # ------------------------------------------------------------------
    # viewpoint reset (idea.txt point 4 -- closed loop)
    # ------------------------------------------------------------------

    def viewpoint_reset(self, rollback_version: int, c2w: np.ndarray,
                        fxfycxcy: np.ndarray, height: int = 480, width: int = 832
                        ) -> Tuple[Optional[np.ndarray], int]:
        """Abandon the current rollout; roll the asset back and render the new
        reference frame.

        Returns (reference (3,H,W) float32 [0,1] | None, asset version post-rollback).
        None means the rolled-back asset is empty (nothing observed yet).
        """
        hdr, body = ipc.viewpoint_reset_msg(rollback_version, c2w, fxfycxcy, height, width)
        ipc.send_msg(self.sock, hdr, body)
        rh, rb = ipc.recv_msg(self.sock)
        if rh.get("error") == "empty_asset":
            return None, rh.get("version", 0)
        if rh.get("error"):
            raise RuntimeError(f"viewpoint reset failed: {rh}")
        return ipc.decode_viewpoint_reset_resp(rh, rb)

    def resync_trajectory(self, viewmats: np.ndarray, Ks: np.ndarray) -> int:
        """Replace the trajectory table (new-viewpoint rollout, chunks numbered from 0)."""
        c2w = hy_w2c_to_c2w(viewmats)
        fxfy = hy_ks_to_movies_fxfycxcy(Ks, None)
        hdr, body = ipc.trajectory_resync_msg(c2w, fxfy)
        ipc.send_msg(self.sock, hdr, body)
        rh, _ = ipc.recv_msg(self.sock)
        assert rh["type"] == "ACK", f"trajectory resync failed: {rh}"
        return len(c2w)

    def close(self) -> None:
        try:
            ipc.send_msg(self.sock, *ipc.control_msg("SHUTDOWN"))
            ipc.recv_msg(self.sock)
        except (ConnectionError, socket.timeout, OSError):
            pass
        finally:
            self.sock.close()


# ---------------------------------------------------------------------------
# Query pose selection: "future" mode (v1 default) and "fov_kept" mode
# ---------------------------------------------------------------------------

def select_memory_latents_future(
    current_chunk: int,
    chunk_latent_frames: int = 4,
    memory_slots: int = 20,
) -> List[int]:
    """future mode: use the current chunk's own latent poses + recent history.

    The DiT's context length/rope layout must stay as-is, so we fill `memory_slots`
    latents: the current chunk's 4 (the upcoming poses) plus the most recent
    preceding latents. Ordering: sorted ascending (rope offset ordering).
    """
    cur_start = current_chunk * chunk_latent_frames
    future = list(range(cur_start, cur_start + chunk_latent_frames))
    fill = memory_slots - len(future)
    history = []
    if fill > 0:
        h_start = max(0, cur_start - fill)
        history = list(range(h_start, cur_start))
    return sorted(set(future + history))


def version_cap_for_chunk(current_chunk: int) -> int:
    """While denoising chunk t, only chunks <= t-2 are guaranteed encoded.

    The (t-1)-th update may be in flight on the server, so the cap must exclude it:
    visible versions <= (t-2) + 1 = t - 1.
    """
    if current_chunk < 2:
        return 0  # nothing guaranteed yet -- queries return None (empty under cap)
    return current_chunk - 1
