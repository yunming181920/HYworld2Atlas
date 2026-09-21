"""hy_driver: the UI's GPU A-side partner process (runs in the HY environment).

Responsibilities:
    - drives the AR rollout chunk by chunk (via rollout_fork / pipeline)
    - receives control commands (pause/continue/restore/motion/obs_pose/quit)
    - pushes generated frames to the UI on the "frame" topic
    - while paused: renders ATLAS observation frames from the asset at the
      UI's observation camera pose (the UI has no model access; this process
      owns the AssetClient)
    - "continue": branches on whether the observation camera moved
        (moved  -> viewpoint_reset hard-cut regeneration, DESIGN.md section 4)
        (still -> resume the original rollout: no reset, just unpause)

ZMQ wiring (ipc endpoints; UI side is player.py):
    PUB "frame"   : b"frame" + flag(1: 0=gen,1=obs) + count(4 LE) + HxWx3 uint8
    SUB "ctrl"    : "ctrl " + json {cmd, ...}
    PUB "status"  : "status " + text (CONFLATE-latest on UI side)

The driver deliberately keeps the UI process free of HY/MoVieS imports, so the
window can run in any python env with pygame+zmq.
"""

import json
import os
import sys
import time
import threading

import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from gs_bridge.hy_client.client import AssetClient
from gs_bridge.ui.keys import LOGICAL_MOTIONS, apply_motion, motion_to_pose_string_entry

FRAME_EP = "ipc:///tmp/gs_bridge_frames"
CTRL_EP = "ipc:///tmp/gs_bridge_ctrl"
STATUS_EP = "ipc:///tmp/gs_bridge_status"


class HybridDriver:
    """Drives generation, handles pause/continue/restore + observation rendering."""

    def __init__(self, pipe, bridge, initial_pose_json, num_latents: int = 512):
        """pipe: HY pipeline (with rollout_fork installed via memory_replace.install);
        bridge: MemorySource; initial_pose_json: dict or pose string for chunk 0..n0."""
        self.pipe = pipe
        self.bridge = bridge
        self.client: AssetClient = bridge.client

        ctx = zmq.Context()
        self.frame_pub = ctx.socket(zmq.PUB)
        self.frame_pub.bind(FRAME_EP)
        self.ctrl_sub = ctx.socket(zmq.SUB)
        self.ctrl_sub.setsockopt(zmq.SUBSCRIBE, b"ctrl")
        self.ctrl_sub.bind(CTRL_EP)
        self.status_pub = ctx.socket(zmq.PUB)
        self.status_pub.bind(STATUS_EP)
        self.ctx = ctx

        # driver state
        self.paused = False
        self.quit = False
        self.frame_count = 0
        self.pending_motions: list[str] = []      # queued logical steps
        self.obs_camera = None                    # (4,4) c2w while paused
        self.obs_intrinsics = np.array([0.75, 0.75 * 832 / 480, 0.5, 0.5])
        self.pause_anchor_pose = None             # generation pose at pause time
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # frame/status publishing
    # ------------------------------------------------------------------

    def push_gen_frames(self, frames: np.ndarray):
        """frames: (F, 3, H, W) float [0,1] from the chunk decode (rollout_fork B)."""
        for f in frames:
            self.frame_count += 1
            self._send_frame(f, is_obs=False)

    def _send_frame(self, frame_rgb_chw: np.ndarray, is_obs: bool):
        img = (np.clip(frame_rgb_chw, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        header = bytes([1 if is_obs else 0]) + self.frame_count.to_bytes(4, "little")
        self.frame_pub.send(b"frame" + header + img.tobytes())

    def set_status(self, text: str):
        self.status_pub.send_string("status " + text)

    # ------------------------------------------------------------------
    # control command handling (runs between chunks + a poll thread while paused)
    # ------------------------------------------------------------------

    def poll_cmds(self, blocking_ms: int = 0):
        try:
            while True:
                if blocking_ms:
                    self.ctrl_sub.setsockopt(zmq.RCVTIMEO, blocking_ms)
                raw = self.ctrl_sub.recv(zmq.NOBLOCK)
                if not raw.startswith(b"ctrl "):
                    continue
                msg = json.loads(raw[5:].decode())
                self._handle_cmd(msg)
        except zmq.Again:
            pass

    def _handle_cmd(self, msg: dict):
        cmd = msg.get("cmd")
        if cmd == "pause":
            self.paused = True
            self.pause_anchor_pose = self._current_generation_pose()
            self.set_status("paused")
        elif cmd == "continue":
            cam = np.array(msg.get("camera", np.eye(4)), dtype=np.float64)
            self._do_continue(cam)
        elif cmd == "restore":
            # restore is purely a UI-side camera animation; driver just unpauses obs
            self.set_status("restoring view")
        elif cmd == "motion":
            self.pending_motions.append(msg["logical"])
        elif cmd == "obs_pose":
            self.obs_camera = np.array(msg["camera"], dtype=np.float64)
        elif cmd == "quit":
            self.quit = True
        else:
            self.set_status(f"unknown cmd {cmd}")

    # ------------------------------------------------------------------
    # continue: resume vs hard-cut regeneration
    # ------------------------------------------------------------------

    def _current_generation_pose(self) -> np.ndarray:
        """Last pose of the generated trajectory (the rollout's frontier)."""
        # the bridge tracks the trajectory; frontier = last latent pose pushed
        # (maintained by the rollout loop via bridge hooks)
        return getattr(self.bridge, "last_generation_pose", np.eye(4))

    def _camera_moved(self, cam: np.ndarray, tol_t=1e-4, tol_deg=0.1) -> bool:
        if self.pause_anchor_pose is None:
            return True
        d = cam[:3, 3] - self.pause_anchor_pose[:3, 3]
        R = self.pause_anchor_pose[:3, :3].T @ cam[:3, :3]
        angle = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        return bool(np.linalg.norm(d) > tol_t or angle > tol_deg)

    def _do_continue(self, cam: np.ndarray):
        """The user decision point (2026-09-21): continue branches on camera state."""
        if not self._camera_moved(cam):
            # camera at anchor: resume the original rollout, nothing invalidated
            self.paused = False
            self.set_status("resumed (original trajectory)")
            return

        # moved: hard-cut regeneration from the new viewpoint
        # 1. build the new trajectory: [current cam pose] + pending motions
        poses = [cam.copy()]
        for logical in self.pending_motions:
            poses.append(apply_motion(poses[-1], LOGICAL_MOTIONS[logical]))
        traj_c2w = np.stack(poses)

        # 2. viewpoint reset: asset rollback to t-2 + reference frame at cam
        #    (MemorySource.viewpoint_reset resyncs the new trajectory server-side)
        t_current = getattr(self.bridge, "current_chunk", 0)
        reference, version = self.bridge.viewpoint_reset(
            new_pose_c2w=cam,
            new_fxfycxcy=self.obs_intrinsics,
            new_viewmats=np.linalg.inv(traj_c2w),   # HY w2c table
            new_Ks=self._hy_ks_from_fxfy(self.obs_intrinsics),
            t_current=t_current,
        )
        if reference is not None:
            self._send_frame(reference, is_obs=True)
        self.pending_motions.clear()

        # 3. restart the rollout from chunk 0 with the reference frame as I2V input
        #    (implemented by the subclass driver loop: see run loop docstring)
        self.restart_rollout_requested = True
        self.restart_reference = reference
        self.paused = False
        self.set_status(f"regenerating from new viewpoint (asset v{version})")

    def _hy_ks_from_fxfy(self, fxfy: np.ndarray) -> np.ndarray:
        """MoVieS fxfycxcy -> HY normalized Ks (inverse of camera_conv conversion)."""
        from gs_bridge.common.camera_conv import movies_fxfycxcy_to_hy_ks
        return movies_fxfycxcy_to_hy_ks(fxfy[None])[0]

    # ------------------------------------------------------------------
    # paused-mode observation rendering (ATLAS)
    # ------------------------------------------------------------------

    def render_observation(self):
        """One ATLAS render at the UI's observation camera; publish as obs frame."""
        if self.obs_camera is None:
            return
        try:
            img = self.client.atlas_view(
                c2w=self.obs_camera,
                fxfycxcy=self.obs_intrinsics,
                height=480, width=832,
                time_fraction=1.0,
            )
            if img is not None:
                self._send_frame(img, is_obs=True)
        except Exception as e:
            self.set_status(f"obs render failed: {e}")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run_paused_loop(self):
        """While paused: render observations at ~10fps and keep polling commands."""
        last_obs = 0.0
        while self.paused and not self.quit:
            self.poll_cmds()
            now = time.perf_counter()
            if now - last_obs >= 0.1:
                self.render_observation()
                last_obs = now
            time.sleep(0.02)

    # hooks for the generation loop (rollout_fork calls these between chunks) ----

    def on_chunk_frames(self, chunk_idx: int, frames: np.ndarray):
        """Called by rollout_fork block B (via bridge) with decoded chunk frames."""
        self.push_gen_frames(frames)
        self.poll_cmds()  # pause takes effect at chunk boundaries
        if self.paused:
            self.run_paused_loop()  # blocks until continue/quit

    def on_rollout_finished(self):
        self.set_status("rollout finished (queue empty) -- feed WASD to extend")

    def on_chunk_start(self, chunk_idx: int) -> list:
        """Gives the rollout the next 4-latent motions (from the pending queue).

        Returns list of logical motion names (len <= chunk_latent_frames) that
        should extend the trajectory for this chunk; the caller (rollout loop
        adapter) turns them into pose updates. Empty list = keep current heading.
        """
        take = self.pending_motions[:4]
        del self.pending_motions[:4]
        return take
