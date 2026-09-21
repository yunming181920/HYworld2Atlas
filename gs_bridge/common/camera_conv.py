"""Camera conventions between HY-WorldPlay and MoVieS + SE3 pose interpolation.

HY side (hyvideo/generate.py pose_to_input):
    viewmats: (L, 4, 4) W2C (inverted from c2w json)
    Ks: (L, 3, 3) normalized as:
        fx' = fx / (2 * cx), fy' = fy / (2 * cy), cx' = cy' = 0.5
    i.e. fx' is focal in units of image WIDTH (since 2*cx ~ width), principal point at center.

MoVieS side (src/utils/geo_util.py):
    C2W: (F, 4, 4) camera-to-world, OpenCV convention
    fxfycxcy: (F, 4) = (fx, fy, cx, cy) each normalized by image width/height respectively:
        fx, cx in units of width; fy, cy in units of height (geo_util.py plucker_ray / fxfycxcy_to_intrinsics).

Conversion (HY Ks -> MoVieS fxfycxcy), assuming principal point centered in HY:
    fx_movies = fx_hy * 2   ... NO. Careful:
    HY: x_norm_pix = fx_hy * X_cam / Z + 0.5, where x_norm_pix in [0,1] of WIDTH.
        => fx_hy is in units of width already.
    MoVieS pixel K: u_px = fx_m * W * X/Z + cx_m * W  (fx_m in width units, cx_m in width units)
    HY pixel K:    u_px = fx_hy * W * X/Z + 0.5 * W
    => fx_m = fx_hy, cx_m = 0.5 (width units). fy_m = fy_hy * (W/H)?? See normalize_k_movies below.
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ---------------------------------------------------------------------------
# Intrinsics: HY Ks <-> MoVieS fxfycxcy
# ---------------------------------------------------------------------------

def hy_ks_to_movies_fxfycxcy(Ks: np.ndarray, aspect: float) -> np.ndarray:
    """Convert HY normalized intrinsics to MoVieS fxfycxcy.

    Args:
        Ks: (L, 3, 3) HY-normalized K (fx/(2cx), fy/(2cy), principal at 0.5).
        aspect: W / H of the HY video (e.g. 832/480).

    Returns:
        (L, 4) MoVieS (fx, fy, cx, cy): fx, cx in width units; fy, cy in height units.

    Math:
        HY pixel K: fx_px = fx_hy * W, cx_px = 0.5 * W (fx_hy in width units)
                    fy_px = fy_hy * H, cy_px = 0.5 * H (fy_hy in height units)
        MoVieS:     fx_m = fx_px / W = fx_hy; cx_m = 0.5
                    fy_m = fy_px / H = fy_hy; cy_m = 0.5
        => direct passthrough for fx, and principal points at 0.5 (HY assumes centered).
        `aspect` is kept for API symmetry / future non-centered cases; with centered
        principal point it is unused.
    """
    Ks = np.asarray(Ks, dtype=np.float64)
    assert Ks.ndim == 3 and Ks.shape[1:] == (3, 3)
    L = Ks.shape[0]
    out = np.zeros((L, 4), dtype=np.float64)
    out[:, 0] = Ks[:, 0, 0]  # fx (width units) -- HY fx already width-normalized
    out[:, 1] = Ks[:, 1, 1]  # fy (height units) -- HY fy already height-normalized
    out[:, 2] = 0.5          # cx (width units), HY centers principal point
    out[:, 3] = 0.5          # cy (height units)
    return out


def movies_fxfycxcy_to_hy_ks(fxfycxcy: np.ndarray, aspect: float) -> np.ndarray:
    """Inverse of hy_ks_to_movies_fxfycxcy. aspect = W/H (unused with centered principal)."""
    f = np.asarray(fxfycxcy, dtype=np.float64)
    assert f.ndim == 2 and f.shape[1] == 4
    L = f.shape[0]
    Ks = np.zeros((L, 3, 3), dtype=np.float64)
    Ks[:, 0, 0] = f[:, 0]
    Ks[:, 1, 1] = f[:, 1]
    Ks[:, 0, 2] = 0.5
    Ks[:, 1, 2] = 0.5
    return Ks


def fov_from_fxfycxcy(fx: float, fy: float) -> Tuple[float, float]:
    """Horizontal/vertical FOV (radians) from normalized focal lengths."""
    return 2.0 * np.arctan(0.5 / fx), 2.0 * np.arctan(0.5 / fy)


# ---------------------------------------------------------------------------
# Extrinsics: HY w2c <-> c2w  (plain matrix inverse, kept explicit for readability)
# ---------------------------------------------------------------------------

def hy_w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """(L, 4, 4) W2C -> C2W via batch inverse."""
    w2c = np.asarray(w2c, dtype=np.float64)
    assert w2c.ndim == 3 and w2c.shape[1:] == (4, 4)
    return np.linalg.inv(w2c)


def movies_c2w_to_hy_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64))


# ---------------------------------------------------------------------------
# SE3 interpolation between latent poses -> per-pixel-frame poses
# ---------------------------------------------------------------------------

def se3_interpolate_poses(c2w_a: np.ndarray, c2w_b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate one SE3 pose pair at parameter t in [0, 1].

    Translation: linear. Rotation: quaternion slerp. Both use the same t.

    The trajectory is built by constant-velocity motion steps (generate.py: w=0.08/frame,
    yaw/pitch=3deg/frame), so linear interpolation between latent poses IS the true
    trajectory (no approximation error) as long as motion is uniform within the span.
    """
    c2w_a = np.asarray(c2w_a, dtype=np.float64)
    c2w_b = np.asarray(c2w_b, dtype=np.float64)
    assert c2w_a.shape == (4, 4) and c2w_b.shape == (4, 4)

    Ra = Rot.from_matrix(c2w_a[:3, :3])
    Rb = Rot.from_matrix(c2w_b[:3, :3])
    R_interp = Rot.from_quat(
        Rot.quaternion_multiply(
            # slerp on quaternions
            _slerp_quat(Ra.as_quat(), Rb.as_quat(), t)
        )
    ) if False else Rot.from_quat(_slerp_quat(Ra.as_quat(), Rb.as_quat(), t))

    trans = c2w_a[:3, 3] * (1.0 - t) + c2w_b[:3, 3] * t
    out = np.eye(4)
    out[:3, :3] = R_interp.as_matrix()
    out[:3, 3] = trans
    return out


def _slerp_quat(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation of quaternions (xyzw, scipy convention).

    Direct implementation to avoid scipy version quirks of Slerp class
    (which requires sorted times arrays); mathematically identical.
    """
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    dot = np.dot(qa, qb)
    # shortest path
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        # nearly parallel: linear + renormalize
        q = qa * (1.0 - t) + qb * t
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta0 = np.sin(theta0)
    s_a = np.sin((1.0 - t) * theta0) / sin_theta0
    s_b = np.sin(t * theta0) / sin_theta0
    return s_a * qa + s_b * qb


def interpolate_frame_poses(c2w_latents: np.ndarray, frame_indices: List[int]) -> np.ndarray:
    """Interpolate per-pixel-frame c2w from the latent pose table.

    Frame f's pose = SE3 interp between latent (l-1) and latent (l) poses at
    t = position of f within latent l's 4-frame span [0, 1].

    Frame 0 uses latent 0's pose directly (no interpolation anchor before it).
    Latent l covers frames [4l-3 .. 4l] (l >= 1); f = 4l -> t = 3/3 = 1.0? We define
    t = (f - (4l - 3)) / 4 in [0.25, 1.0] ... but anchor span should be [latent l-1, latent l].
    Frame 4l (last of latent l) aligns exactly with latent l's pose -> t = 1.0.
    Frame 4l-3 (first of latent l) is 3/4 between latent l-1 and latent l -> t = 0.75? No:
    spacing between latent poses is 4 frames; frame 4l-3 is 3 frames after frame 4l-4
    (= latent l-1's anchor) -> t = 3/4. This means t never hits 0 -- by construction
    frame 0 is the only exact anchor before latent boundaries.

    Simpler equivalent: f in latent l (l >= 1):
        anchor_a = latent l-1 pose, anchor_b = latent l pose
        t = (f - (4*(l-1))) / 4     # 4*(l-1) is latent l-1's anchor frame
        => f = 4l-3 -> t = 1/4; f = 4l -> t = 1.0
    """
    c2w_latents = np.asarray(c2w_latents, dtype=np.float64)
    L = c2w_latents.shape[0]
    out = np.zeros((len(frame_indices), 4, 4), dtype=np.float64)
    for i, f in enumerate(frame_indices):
        l = (f + 3) // 4 if f >= 1 else 0
        if l == 0:
            out[i] = c2w_latents[0]
            continue
        anchor_frame = 4 * (l - 1)
        t = (f - anchor_frame) / 4.0
        out[i] = se3_interpolate_poses(c2w_latents[l - 1], c2w_latents[l], t)
    return out


# ---------------------------------------------------------------------------
# MoVieS canonical normalization (window-local) and its analytic inverse
# ---------------------------------------------------------------------------

def canonical_normalize(c2w_frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize a window's poses to MoVieS canonical frame.

    MoVieS training convention (src/data/base_dataset.py): first camera -> identity,
    plus scale normalization so mean |unprojected point| == camera_norm_unit.
    We apply the pose part analytically; the scale part needs depths, so it is
    applied separately in the encoder (scale factor recorded in the window).

    Args:
        c2w_frames: (F, 4, 4) window poses in HY world frame.

    Returns:
        (c2w_canonical (F, 4, 4), T_hy2canon (4, 4), T_canon2hy (4, 4)):
        c2w_canonical = T_hy2canon @ c2w_frames (row-major left-multiply, extrinsic compose).
        The window stores T_canon2hy so rendered results (depths) can be mapped back.
    """
    c2w_frames = np.asarray(c2w_frames, dtype=np.float64)
    assert c2w_frames.ndim == 3 and c2w_frames.shape[1:] == (4, 4)
    first_inv = np.linalg.inv(c2w_frames[0])
    c2w_can = np.einsum("ij,fjk->fik", first_inv, c2w_frames)
    return c2w_can, first_inv, c2w_frames[0]


def apply_canonical_to_points(points_canonical: np.ndarray, T_canon2hy: np.ndarray) -> np.ndarray:
    """Map points from window-canonical frame back to HY world frame."""
    pts = np.asarray(points_canonical)
    return pts @ T_canon2hy[:3, :3].T + T_canon2hy[:3, 3]
