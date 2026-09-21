"""First-person camera state for the UI (observation camera + pose bookkeeping).

The observation camera is a plain c2w driven by the same motion primitives as
HY's trajectory (keys.apply_motion). During PLAYING it tracks the generation
pose (last pose of the generated trajectory); during PAUSED the user flies it
around the asset. "Continue" semantics compare the two (moved or not) exactly
as the user specified: unchanged camera -> resume original rollout; moved
camera -> hard-cut regeneration from the new viewpoint (viewpoint_reset).
"""

import numpy as np

from gs_bridge.ui.keys import apply_motion, LOGICAL_MOTIONS


class FirstPersonCamera:
    def __init__(self, c2w: np.ndarray):
        self.c2w = np.array(c2w, dtype=np.float64).copy()  # current pose
        self.anchor: np.ndarray = self.c2w.copy()          # pose at pause time
        self.moved_steps = 0

    # ------------------------------------------------------------------

    def reset_to(self, c2w: np.ndarray) -> None:
        """Teleport (e.g. at pause time, or after reset)."""
        self.c2w = np.array(c2w, dtype=np.float64).copy()
        self.anchor = self.c2w.copy()
        self.moved_steps = 0

    def step(self, logical: str) -> None:
        """One motion step (WASD / arrow). Same grain as a trajectory latent step."""
        motion = LOGICAL_MOTIONS[logical]
        self.c2w = apply_motion(self.c2w, motion)
        self.moved_steps += 1

    def mark_anchor(self) -> None:
        """Freeze the current pose as the pause anchor (for moved?/restore)."""
        self.anchor = self.c2w.copy()
        self.moved_steps = 0

    # ------------------------------------------------------------------

    def moved_from_anchor(self, tol_translation: float = 1e-4,
                          tol_angle_deg: float = 0.1) -> bool:
        """Whether the user flew the camera away from the pause anchor.

        Used by "continue": unmoved -> resume the original rollout (no reset);
        moved -> hard-cut regeneration (viewpoint_reset, DESIGN.md section 4).
        """
        d = self.c2w[:3, 3] - self.anchor[:3, 3]
    # rotation delta
        R = self.anchor[:3, :3].T @ self.c2w[:3, :3]
        angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        return bool(
            np.linalg.norm(d) > tol_translation
            or np.rad2deg(angle) > tol_angle_deg
        )

    def slerp_to_anchor(self, n_steps: int):
        """Restore animation: yield n_steps poses interpolating back to the anchor.

        Uses the same SE3 interpolation as the pipeline (translation linear +
        quaternion slerp) so the restore path looks exactly like trajectory
        playback in reverse.
        """
        from gs_bridge.common.camera_conv import se3_interpolate_poses

        n_steps = max(n_steps, 2)
        for i in range(n_steps):
            t = (i + 1) / (n_steps - 1) if n_steps > 1 else 1.0
            if t >= 1.0:
                self.c2w = self.anchor.copy()
                self.moved_steps = 0
            else:
                self.c2w = se3_interpolate_poses(self.anchor, self.c2w, t)
            yield self.c2w
        self.c2w = self.anchor.copy()
        self.moved_steps = 0

    # ------------------------------------------------------------------

    def snapshot(self) -> np.ndarray:
        return self.c2w.copy()
