"""GPU B asset server process: owns MoVieS + GlobalAsset, serves IPC.

Lifecycle (DESIGN.md section 3):
    1. boot: load MoVieS encoder (slow, one-time)
    2. TRAJECTORY_SYNC: receive the latent pose table once; build per-chunk pose
       caches lazily (chunk_frame_range -> interpolate_frame_poses)
    3. loop:
         CHUNK_UPDATE  -> encode window (canonicalize inside encoder) -> insert;
                          version = chunk_idx + 1
         QUERY_REQ     -> AssetQuerier.query(version_cap, latent_indices) -> respond
         ATLAS_REQ     -> frozen-time free-camera render from latest complete window
         PING/SHUTDOWN -> control

Duplex rule (idea.txt): while the client denoises chunk t it queries with
version_cap = t - 1 (chunk t-1 may still be encoding here -- the cap excludes it,
matching the "t-2 asset" semantics; see time_map.window_version).

Concurrency: v1 serves messages sequentially on one connection (the HY client is a
single consumer). Updates and queries interleave by arrival; no locking beyond
GlobalAsset's internal lock. Lookahead pre-rendering (M4 optimization) hooks in
after each successful QUERY_RESP.
"""

import argparse
import os
import socket
import sys
import threading
import time
import traceback
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from gs_bridge.common import ipc_protocol as ipc
from gs_bridge.common.camera_conv import interpolate_frame_poses
from gs_bridge.common.time_map import chunk_frame_range
from gs_bridge.common.types import TrajectorySync


class AssetServer:
    def __init__(self, movies_repo: str, ckpt: Optional[str], host: str, port: int,
                 capacity_windows: int = 20, device: str = "cuda"):
        from gs_bridge.asset_server.asset import GlobalAsset
        from gs_bridge.asset_server.encoder import MoVieSEncoder

        print(f"[server] booting MoVieS encoder (repo={movies_repo}, ckpt={ckpt})...", flush=True)
        t0 = time.perf_counter()
        self.encoder = MoVieSEncoder(movies_repo=movies_repo, ckpt=ckpt, device=device)
        print(f"[server] encoder ready in {time.perf_counter()-t0:.1f}s", flush=True)

        self.asset = GlobalAsset(capacity_windows=capacity_windows)
        self.trajectory: Optional[TrajectorySync] = None
        self.querier = None  # built after trajectory sync
        self.host, self.port = host, port
        self._chunk_pose_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.stats = {"updates": 0, "queries": 0, "atlas": 0, "encode_s": 0.0, "query_s": 0.0}

    # ------------------------------------------------------------------
    # trajectory / pose derivation
    # ------------------------------------------------------------------

    def require_trajectory(self) -> TrajectorySync:
        assert self.trajectory is not None, "TRAJECTORY_SYNC must arrive before data messages"
        return self.trajectory

    def chunk_poses(self, chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """(c2w_frames (F,4,4) HY world, timestamps (F,)) for a chunk, cached.

        Poses derive purely from the trajectory table: chunk_frame_range ->
        interpolate_frame_poses. The client never ships poses.
        """
        if chunk_idx not in self._chunk_pose_cache:
            traj = self.require_trajectory()
            fs, fe = chunk_frame_range(chunk_idx)
            # clamp to available latents (last chunk may request beyond-table frames)
            n_avail = self._n_available_frames(chunk_idx, fs, fe)
            frames = list(range(fs, min(fe, n_avail)))
            poses = interpolate_frame_poses(traj.c2w_latents, frames)
            self._chunk_pose_cache[chunk_idx] = (poses, np.array(frames, dtype=np.float64))
        return self._chunk_pose_cache[chunk_idx]

    def _n_available_frames(self, chunk_idx: int, fs: int, fe: int) -> int:
        traj = self.require_trajectory()
        # frames the trajectory can produce: 4L - 3 total
        return num_frames_from_latents(traj.num_latents)

    # ------------------------------------------------------------------
    # message handlers
    # ------------------------------------------------------------------

    def handle_trajectory_sync(self, header: Dict, body: bytes) -> None:
        from gs_bridge.asset_server.query import AssetQuerier

        c2w, fxfy = ipc.decode_trajectory_sync(header, body)
        self.trajectory = TrajectorySync(c2w_latents=c2w, fxfycxcy=fxfy, num_latents=header["num_latents"])
        self._chunk_pose_cache.clear()
        # querier needs the trajectory pose table (HY world, MoVieS intrinsics)
        self.querier = AssetQuerier(
            self.asset, self.encoder.model.gs_renderer, self.trajectory,
            device=self.encoder.device, dtype=self.encoder.dtype,
        )
        print(f"[server] trajectory synced: {self.trajectory.num_latents} latents", flush=True)

    def handle_chunk_update(self, header: Dict, body: bytes) -> Dict:
        import torch

        chunk_idx, frames_np, _ts = ipc.decode_chunk_update(header, body)
        poses, _ = self.chunk_poses(chunk_idx)
        assert poses.shape[0] == frames_np.shape[0], (
            f"chunk {chunk_idx}: {frames_np.shape[0]} frames vs {poses.shape[0]} poses -- "
            "client/server disagree on chunk frame count (trajectory sync mismatch?)"
        )
        traj = self.require_trajectory()
        # intrinsics: latent table -> per-frame (constant intrinsics in v1)
        fxfy = traj.fxfycxcy[0]

        t0 = time.perf_counter()
        frames_t = torch.from_numpy(frames_np).to(self.encoder.device)
        window = self.encoder.encode_window(
            frames=frames_t,
            c2w_frames=poses,
            fxfycxcy=fxfy,
            global_timestamps=np.arange(frames_np.shape[0], dtype=np.float64),
            chunk_idx=chunk_idx,
        )
        version = self.asset.insert_window(window)
        dt = time.perf_counter() - t0
        self.stats["updates"] += 1
        self.stats["encode_s"] += dt
        print(f"[server] chunk {chunk_idx} encoded in {dt:.2f}s -> version {version} "
              f"({len(self.asset)}/{self.asset.capacity} windows)", flush=True)
        return {"type": "ACK", "version": version}

    def handle_query(self, header: Dict) -> Dict:
        t0 = time.perf_counter()
        assert self.querier is not None, "query before trajectory sync"
        res = self.querier.query(
            latent_indices=np.array(header["latent_indices"], dtype=int),
            version_cap=header["version_cap"],
            height=header["height"],
            width=header["width"],
            with_depth=header["with_depth"],
        )
        dt = time.perf_counter() - t0
        self.stats["queries"] += 1
        self.stats["query_s"] += dt
        if res is None:
            return {"type": "QUERY_RESP", "error": "empty_asset", "body_len": 0}
        hdr, body = ipc.query_resp_msg(
            res.images, res.depths, res.version, res.latent_indices, res.chunk_ids
        )
        print(f"[server] query {len(header['latent_indices'])} views in {dt:.2f}s "
              f"(version {res.version})", flush=True)
        return (hdr, body)

    def handle_atlas(self, header: Dict, body: bytes) -> Tuple[Dict, bytes]:
        assert self.querier is not None
        c2w, fxfy = ipc.decode_atlas_req(header, body)
        slot = self.asset.snapshot_latest()
        if slot is None:
            return {"type": "ATLAS_RESP", "error": "empty_asset", "body_len": 0}, b""
        # frozen time at the requested fraction of the latest window
        w = slot.window
        n_out = w.output_timesteps.shape[1]
        time_idx = int(round(header["time_fraction"] * (n_out - 1)))
        time_idx = min(max(time_idx, 0), n_out - 1)
        mo = self.querier._bake_time(w, time_idx)
        import torch
        T_hy2canon = np.linalg.inv(w.T_canon2hy)
        c2w_can = T_hy2canon @ c2w
        C2W_q = torch.from_numpy(c2w_can).float()[None, None].to(
            device=self.encoder.device, dtype=self.encoder.dtype)
        FXFY_q = torch.from_numpy(fxfy).float()[None, None].to(
            device=self.encoder.device, dtype=self.encoder.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.encoder.dtype == torch.bfloat16):
            out = self.encoder.model.gs_renderer.render(
                mo, w.input_c2w, w.input_fxfycxcy, C2W_q, FXFY_q,
                height=header["height"], width=header["width"],
            )
        img = out["image"][0, 0].float().clamp(0, 1).cpu().numpy()[None]  # (1,3,H,W)
        self.stats["atlas"] += 1
        hdr, body = ipc.atlas_resp_msg(img, self.asset.version)
        return hdr, body

    # ------------------------------------------------------------------
    # viewpoint reset (DESIGN.md section 4)
    # ------------------------------------------------------------------

    def handle_viewpoint_reset(self, header: Dict, body: bytes) -> Tuple[Dict, bytes]:
        """Atomic viewpoint reset: rollback -> reference render.

        Semantics: the client abandons the current rollout at chunk t (its t-1/t
        content is conditioned on the old trajectory's context and must not enter
        the asset). The asset rolls back to `rollback_version` (= the t-2 trusted
        boundary), then renders the NEW viewpoint's reference frame from the
        rolled-back asset. The client follows with TRAJECTORY_RESYNC and starts a
        fresh rollout numbered from chunk 0.

        Stale in-flight CHUNK_UPDATEs are rejected by the asset's version guard
        (insert_window discards version <= current), so arrival order after the
        reset does not matter.
        """
        assert self.querier is not None, "viewpoint reset before trajectory sync"
        rollback_version, c2w, fxfy = ipc.decode_viewpoint_reset(header, body)
        version = self.asset.rollback_to(rollback_version)
        print(f"[server] viewpoint reset: rolled back to version {version} "
              f"({len(self.asset)} windows)", flush=True)

        # reference frame at the new viewpoint from the rolled-back asset
        slot = self.asset.snapshot_latest()
        if slot is None:
            return {"type": "VIEWPOINT_RESET_RESP", "error": "empty_asset",
                    "version": version, "body_len": 0}, b""
        import torch

        w = slot.window
        # frozen at the window's latest state (the boundary content itself)
        time_idx = w.output_timesteps.shape[1] - 1
        mo = self.querier._bake_time(w, time_idx)
        T_hy2canon = np.linalg.inv(w.T_canon2hy)
        c2w_can = T_hy2canon @ c2w
        C2W_q = torch.from_numpy(c2w_can).float()[None, None].to(
            device=self.encoder.device, dtype=self.encoder.dtype)
        FXFY_q = torch.from_numpy(fxfy).float()[None, None].to(
            device=self.encoder.device, dtype=self.encoder.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.encoder.dtype == torch.bfloat16):
            out = self.encoder.model.gs_renderer.render(
                mo, w.input_c2w, w.input_fxfycxcy, C2W_q, FXFY_q,
                height=header["height"], width=header["width"],
            )
        ref = out["image"][0, 0].float().clamp(0, 1).cpu().numpy()  # (3, H, W)
        self.stats["resets"] = self.stats.get("resets", 0) + 1
        hdr, body_out = ipc.viewpoint_reset_resp_msg(ref, version)
        return hdr, body_out

    def handle_trajectory_resync(self, header: Dict, body: bytes) -> None:
        """Replace the pose table after a viewpoint reset; invalidate caches."""
        from gs_bridge.asset_server.query import AssetQuerier

        c2w, fxfy = ipc.decode_trajectory_sync(header, body)
        self.trajectory = TrajectorySync(
            c2w_latents=c2w, fxfycxcy=fxfy, num_latents=header["num_latents"]
        )
        self._chunk_pose_cache.clear()
        self.querier.trajectory = self.trajectory
        print(f"[server] trajectory resynced: {self.trajectory.num_latents} latents "
              f"(new-viewpoint rollout)", flush=True)

    # ------------------------------------------------------------------
    # serve loop
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[server] listening on {self.host}:{self.port}", flush=True)
        conn, addr = srv.accept()
        print(f"[server] client connected: {addr}", flush=True)
        try:
            while True:
                header, body = ipc.recv_msg(conn)
                mtype = header.get("type")
                if mtype == "TRAJECTORY_SYNC":
                    self.handle_trajectory_sync(header, body)
                    ipc.send_msg(conn, *ipc.control_msg("ACK"))
                elif mtype == "TRAJECTORY_RESYNC":
                    self.handle_trajectory_resync(header, body)
                    ipc.send_msg(conn, *ipc.control_msg("ACK"))
                elif mtype == "VIEWPOINT_RESET":
                    ipc.send_msg(conn, *self.handle_viewpoint_reset(header, body))
                elif mtype == "CHUNK_UPDATE":
                    resp = self.handle_chunk_update(header, body)
                    if isinstance(resp, tuple):
                        ipc.send_msg(conn, *resp)
                    else:
                        ipc.send_msg(conn, resp, b"")
                elif mtype == "QUERY_REQ":
                    resp = self.handle_query(header)
                    if isinstance(resp, tuple):
                        ipc.send_msg(conn, *resp)
                    else:
                        ipc.send_msg(conn, resp, b"")
                elif mtype == "ATLAS_REQ":
                    ipc.send_msg(conn, *self.handle_atlas(header, body))
                elif mtype == "PING":
                    ipc.send_msg(conn, *ipc.control_msg("PING_RESP"))
                elif mtype == "SHUTDOWN":
                    ipc.send_msg(conn, *ipc.control_msg("ACK"))
                    print("[server] shutdown requested", flush=True)
                    break
                else:
                    print(f"[server] unknown message type: {mtype}", flush=True)
        except ConnectionError as e:
            print(f"[server] connection closed: {e}", flush=True)
        finally:
            conn.close()
            srv.close()
            print(f"[server] stats: {self.stats} asset: {self.asset.describe()}", flush=True)


def num_frames_from_latents(L: int) -> int:
    return 4 * L - 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies_repo", default="MoVieS")
    ap.add_argument("--ckpt", default="MoVieS/resources/movies_ckpt.safetensors")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9820)
    ap.add_argument("--capacity", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    server = AssetServer(
        movies_repo=args.movies_repo, ckpt=args.ckpt, host=args.host, port=args.port,
        capacity_windows=args.capacity, device=args.device,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
