"""IPC wire protocol between hy_client (GPU A) and asset_server (GPU B).

v1 (DESIGN.md): raw TCP socket, JSON header + binary body. Big tensors travel as
raw bytes; the header declares dtype/shape so both sides decode without pickle
(no arbitrary-code-execution surface, and no torch dependency in common/).

Message flow:
    startup:   TRAJECTORY_SYNC   (client -> server, once)
               TRAJECTORY_RESYNC (client -> server, on viewpoint reset: replaces the
                                  pose table and invalidates server pose caches)
    update:    CHUNK_UPDATE      (client -> server, per finished chunk, async)
    query:     QUERY_REQ / QUERY_RESP (request-response)
    reset:     VIEWPOINT_RESET   (client -> server, atomic:
                                  rollback asset to the trusted boundary -> render the
                                  new-viewpoint reference frame -> expect TRAJECTORY_RESYNC)
    atlas:     ATLAS_REQ / ATLAS_RESP (optional observation channel)
    control:   PING / PING_RESP, SHUTDOWN

Header (one JSON line, then body bytes):
    {"type": ..., ...fields, "body_len": N}
"""

import json
import socket
import struct
from typing import Dict, Optional, Tuple

import numpy as np

HEADER_NEWLINE = b"\n"
MAX_BODY = 512 * 1024 * 1024  # 512 MB safety cap


# ---------------------------------------------------------------------------
# message builders (plain dicts; serialization happens in send/recv)
# ---------------------------------------------------------------------------

def trajectory_sync_msg(c2w_latents: np.ndarray, fxfycxcy: np.ndarray) -> Tuple[Dict, bytes]:
    """Startup: full latent pose table (MoVieS convention, HY world frame)."""
    body = c2w_latents.astype(np.float64).tobytes() + fxfycxcy.astype(np.float64).tobytes()
    header = {
        "type": "TRAJECTORY_SYNC",
        "num_latents": int(c2w_latents.shape[0]),
        "body_len": len(body),
    }
    return header, body


def chunk_update_msg(chunk_idx: int, frames: np.ndarray, timestamps: np.ndarray) -> Tuple[Dict, bytes]:
    """Per-chunk update: decoded pixel frames (F, 3, H, W) float32 [0,1] + global timestamps.

    Poses are omitted -- the server derives them from the trajectory table by chunk_idx
    (chunk_frame_range -> interpolate_frame_poses).
    """
    assert frames.dtype == np.float32
    body = frames.tobytes() + timestamps.astype(np.float64).tobytes()
    header = {
        "type": "CHUNK_UPDATE",
        "chunk_idx": int(chunk_idx),
        "frame_count": int(frames.shape[0]),
        "frame_shape": list(frames.shape),  # [F, 3, H, W]
        "body_len": len(body),
    }
    return header, body


def query_req_msg(version_cap: int, latent_indices: np.ndarray,
                  height: int, width: int, with_depth: bool) -> Tuple[Dict, bytes]:
    header = {
        "type": "QUERY_REQ",
        "version_cap": int(version_cap),
        "latent_indices": latent_indices.astype(int).tolist(),
        "height": int(height),
        "width": int(width),
        "with_depth": bool(with_depth),
        "body_len": 0,
    }
    return header, b""


def atlas_req_msg(version_cap: Optional[int], c2w: np.ndarray, fxfycxcy: np.ndarray,
                  height: int, width: int, time_fraction: float) -> Tuple[Dict, bytes]:
    """Free camera observation (Atlas): arbitrary pose, latest complete version."""
    body = c2w.astype(np.float64).tobytes() + fxfycxcy.astype(np.float64).tobytes()
    header = {
        "type": "ATLAS_REQ",
        "version_cap": None if version_cap is None else int(version_cap),
        "height": int(height),
        "width": int(width),
        "time_fraction": float(time_fraction),
        "body_len": len(body),
    }
    return header, body


def query_resp_msg(images: np.ndarray, depths: Optional[np.ndarray],
                   version: int, latent_indices: np.ndarray, chunk_ids: np.ndarray) -> Tuple[Dict, bytes]:
    images = images.astype(np.float32)
    body = images.tobytes()
    if depths is not None:
        body += depths.astype(np.float32).tobytes()
    header = {
        "type": "QUERY_RESP",
        "version": int(version),
        "latent_indices": latent_indices.astype(int).tolist(),
        "chunk_ids": chunk_ids.astype(int).tolist(),
        "image_shape": list(images.shape),
        "depth_shape": None if depths is None else list(depths.shape),
        "body_len": len(body),
    }
    return header, body


def atlas_resp_msg(images: np.ndarray, version: int) -> Tuple[Dict, bytes]:
    images = images.astype(np.float32)
    header = {
        "type": "ATLAS_RESP",
        "version": int(version),
        "image_shape": list(images.shape),
        "body_len": images.nbytes,
    }
    return header, images.tobytes()


def control_msg(kind: str) -> Tuple[Dict, bytes]:
    assert kind in ("PING", "PING_RESP", "SHUTDOWN", "ACK")
    return {"type": kind, "body_len": 0}, b""


# ---------------------------------------------------------------------------
# viewpoint reset (idea.txt point 4, closed loop -- see DESIGN.md section 4)
# ---------------------------------------------------------------------------

def viewpoint_reset_msg(rollback_version: int, c2w: np.ndarray,
                        fxfycxcy: np.ndarray, height: int, width: int) -> Tuple[Dict, bytes]:
    """Atomically: asset rollback + reference-frame render request at the new pose.

    rollback_version: the trusted boundary to roll back to (the t-2 version:
    the last version fully generated along the OLD trajectory). The server
    discards windows newer than this (in-flight t-1/t chunks are invalid under
    the new viewpoint: their content was conditioned on old-trajectory context).
    """
    body = c2w.astype(np.float64).tobytes() + fxfycxcy.astype(np.float64).tobytes()
    header = {
        "type": "VIEWPOINT_RESET",
        "rollback_version": int(rollback_version),
        "height": int(height),
        "width": int(width),
        "body_len": len(body),
    }
    return header, body


def viewpoint_reset_resp_msg(reference: np.ndarray, version: int) -> Tuple[Dict, bytes]:
    """reference: (3, H, W) float32 [0,1] rendered at the new viewpoint post-rollback."""
    reference = reference.astype(np.float32)
    header = {
        "type": "VIEWPOINT_RESET_RESP",
        "version": int(version),  # asset version after rollback
        "image_shape": list(reference.shape),
        "body_len": reference.nbytes,
    }
    return header, reference.tobytes()


def trajectory_resync_msg(c2w_latents: np.ndarray, fxfycxcy: np.ndarray,
                          first_chunk: int = 0) -> Tuple[Dict, bytes]:
    """Replace the trajectory table after a viewpoint reset.

    first_chunk: the chunk index the NEW rollout starts from, in NEW trajectory
    coordinates (the server's stale-insert guard uses chunk_idx, so the client
    must number the new rollout's chunks from 0 -- see rollout_fork block B).
    """
    body = c2w_latents.astype(np.float64).tobytes() + fxfycxcy.astype(np.float64).tobytes()
    header = {
        "type": "TRAJECTORY_RESYNC",
        "num_latents": int(c2w_latents.shape[0]),
        "body_len": len(body),
    }
    return header, body


def decode_viewpoint_reset(header: Dict, body: bytes) -> Tuple[int, np.ndarray, np.ndarray]:
    c2w = np.frombuffer(body[: 16 * 8], dtype=np.float64).reshape(4, 4).copy()
    fxfy = np.frombuffer(body[16 * 8: 16 * 8 + 32], dtype=np.float64).reshape(4).copy()
    return header["rollback_version"], c2w, fxfy


def decode_viewpoint_reset_resp(header: Dict, body: bytes) -> Tuple[np.ndarray, int]:
    shape = tuple(header["image_shape"])
    n = int(np.prod(shape))
    ref = np.frombuffer(body[: n * 4], dtype=np.float32).reshape(shape).copy()
    return ref, header["version"]


# ---------------------------------------------------------------------------
# socket transport
# ---------------------------------------------------------------------------

def send_msg(sock: socket.socket, header: Dict, body: bytes) -> None:
    payload = json.dumps(header).encode("utf-8") + HEADER_NEWLINE + body
    sock.sendall(payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Tuple[Dict, bytes]:
    """Read one (header, body) message. Header is a JSON line; body follows."""
    # read until newline for the header
    hdr = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("socket closed before header")
        if b == HEADER_NEWLINE:
            break
        hdr.extend(b)
        if len(hdr) > 1024 * 1024:
            raise ValueError("header too large")
    header = json.loads(hdr.decode("utf-8"))
    body_len = int(header.get("body_len", 0))
    if body_len > MAX_BODY:
        raise ValueError(f"body too large: {body_len}")
    body = recv_exact(sock, body_len) if body_len else b""
    return header, body


# ---------------------------------------------------------------------------
# body decoders (server side)
# ---------------------------------------------------------------------------

def decode_trajectory_sync(header: Dict, body: bytes) -> Tuple[np.ndarray, np.ndarray]:
    L = header["num_latents"]
    c2w = np.frombuffer(body[: L * 16 * 8], dtype=np.float64).reshape(L, 4, 4).copy()
    fxfy = np.frombuffer(body[L * 16 * 8:], dtype=np.float64).reshape(L, 4).copy()
    return c2w, fxfy


def decode_chunk_update(header: Dict, body: bytes) -> Tuple[int, np.ndarray, np.ndarray]:
    shape = tuple(header["frame_shape"])
    F = shape[0]
    n_px = int(np.prod(shape))
    frames = np.frombuffer(body[: n_px * 4], dtype=np.float32).reshape(shape).copy()
    ts = np.frombuffer(body[n_px * 4:], dtype=np.float64).copy()
    assert ts.shape[0] == F
    return header["chunk_idx"], frames, ts


def decode_atlas_req(header: Dict, body: bytes) -> Tuple[np.ndarray, np.ndarray]:
    c2w = np.frombuffer(body[: 16 * 8], dtype=np.float64).reshape(4, 4).copy()
    fxfy = np.frombuffer(body[16 * 8: 16 * 8 + 32], dtype=np.float64).reshape(4).copy()
    return c2w, fxfy


def decode_query_resp(header: Dict, body: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    img_shape = tuple(header["image_shape"])
    n_img = int(np.prod(img_shape))
    images = np.frombuffer(body[: n_img * 4], dtype=np.float32).reshape(img_shape).copy()
    depths = None
    if header["depth_shape"] is not None:
        d_shape = tuple(header["depth_shape"])
        n_d = int(np.prod(d_shape))
        depths = np.frombuffer(body[n_img * 4: n_img * 4 + n_d * 4], dtype=np.float32).reshape(d_shape).copy()
    return images, depths, header["version"]


def decode_atlas_resp(header: Dict, body: bytes) -> Tuple[np.ndarray, int]:
    shape = tuple(header["image_shape"])
    n = int(np.prod(shape))
    images = np.frombuffer(body[: n * 4], dtype=np.float32).reshape(shape).copy()
    return images, header["version"]
