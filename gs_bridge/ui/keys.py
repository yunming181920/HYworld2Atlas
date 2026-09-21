"""Keyboard -> motion mapping, isomorphic with HY-WorldPlay's native controls.

HY's operation language (generate.py:59-143 + generate_custom_trajectory.py):
    W/S: forward/backward   A/D: left/right strafe
    arrows: yaw/pitch rotation
    one keypress step = ONE motion step = 1 latent of trajectory
    (forward 0.08 units, yaw/pitch 3 degrees per step)

The UI keeps the exact same mapping in BOTH states:
    PLAYING: keystrokes record motions into the pending-generation queue
    PAUSED:  keystrokes drive the observation camera through the asset (ATLAS)
"""

import numpy as np

# Speed constants -- byte-identical to hyvideo/generate.py:80-82
FORWARD_SPEED = 0.08   # units per step
YAW_SPEED = np.deg2rad(3)    # radians per step
PITCH_SPEED = np.deg2rad(3)  # radians per step

# pygame key constants are imported lazily to keep this module testable without
# a display; the map below uses pygame.locals names resolved at import time by
# the caller (player.py). Here we define the logical layer only.

# logical action -> motion dict (generate_camera_trajectory_local format)
LOGICAL_MOTIONS = {
    "forward":  {"forward": FORWARD_SPEED},
    "backward": {"forward": -FORWARD_SPEED},
    "left":     {"right": -FORWARD_SPEED},
    "right":    {"right": FORWARD_SPEED},
    "yaw_left": {"yaw": -YAW_SPEED},
    "yaw_right": {"yaw": YAW_SPEED},
    "pitch_up": {"pitch": PITCH_SPEED},
    "pitch_down": {"pitch": -PITCH_SPEED},
}


def key_to_logical(key_name: str):
    """Map a pygame key constant name (e.g. 'K_w') to a logical motion name.

    Supports both letter keys (WASD) and arrows. Returns None for unmapped keys.
    """
    return _KEY_MAP.get(key_name)


# pygame K_* name -> logical action
_KEY_MAP = {
    "K_w": "forward",
    "K_s": "backward",
    "K_a": "left",
    "K_d": "right",
    "K_LEFT": "yaw_left",
    "K_RIGHT": "yaw_right",
    "K_UP": "pitch_up",
    "K_DOWN": "pitch_down",
}


def apply_motion(T: np.ndarray, motion: dict) -> np.ndarray:
    """Apply one motion step to a c2w pose -- replica of generate_camera_trajectory_local.

    T: (4, 4) c2w. Returns the new c2w. Supports the motion dict keys used by
    LOGICAL_MOTIONS (forward/right/yaw/pitch).
    """
    T = T.copy()
    if "yaw" in motion:
        c, s = np.cos(motion["yaw"]), np.sin(motion["yaw"])
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        T[:3, :3] = T[:3, :3] @ R
    if "pitch" in motion:
        c, s = np.cos(motion["pitch"]), np.sin(motion["pitch"])
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        T[:3, :3] = T[:3, :3] @ R
    forward = motion.get("forward", 0.0)
    if forward != 0:
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, forward])
    right = motion.get("right", 0.0)
    if right != 0:
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([right, 0.0, 0.0])
    return T


def motion_to_pose_string_entry(logical: str, count: int = 1) -> str:
    """Logical action -> pose-string command (w-3 / right-4 / up-2 ...)."""
    name_map = {
        "forward": "w", "backward": "s", "left": "a", "right": "d",
        "yaw_left": "left", "yaw_right": "right",
        "pitch_up": "up", "pitch_down": "down",
    }
    return f"{name_map[logical]}-{count}"
