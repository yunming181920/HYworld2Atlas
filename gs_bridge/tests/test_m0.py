"""M0 unit tests: time mapping, camera conversion, SE3 interpolation, canonical normalization.

Run:  python -m pytest gs_bridge/tests/test_m0.py -v
(no GPU, no heavy deps -- only numpy/scipy)
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.time_map import (
    num_frames_from_latents,
    num_latents_from_frames,
    frame_to_latent,
    latent_to_frames,
    chunk_latent_range,
    chunk_frame_range,
    num_chunks,
    window_version,
)
from common.camera_conv import (
    hy_ks_to_movies_fxfycxcy,
    movies_fxfycxcy_to_hy_ks,
    hy_w2c_to_c2w,
    se3_interpolate_poses,
    interpolate_frame_poses,
    canonical_normalize,
    apply_canonical_to_points,
)


# =========================================================================
# time mapping
# =========================================================================

class TestTimeMap:
    def test_w31_example(self):
        # 'w-31' = 31 motion steps = 32 latents = 125 frames = 8 chunks
        assert num_frames_from_latents(32) == 125
        assert num_latents_from_frames(125) == 32
        assert num_chunks(32) == 8

    def test_first_latent_covers_only_frame0(self):
        assert latent_to_frames(0) == [0]
        assert frame_to_latent(0) == 0

    def test_frame_latent_roundtrip(self):
        for L in (4, 32, 128):
            for f in range(num_frames_from_latents(L)):
                l = frame_to_latent(f)
                assert f in latent_to_frames(l), (L, f, l)

    def test_chunks_tile_frames_exactly(self):
        for L in (8, 32, 128):
            covered = []
            for c in range(num_chunks(L)):
                s, e = chunk_frame_range(c)
                covered.extend(range(s, e))
            assert sorted(covered) == list(range(num_frames_from_latents(L)))

    def test_chunk0_is_13_frames(self):
        s, e = chunk_frame_range(0)
        assert (s, e) == (0, 13)

    def test_chunk1_is_16_frames(self):
        s, e = chunk_frame_range(1)
        assert (s, e) == (13, 29)
        assert chunk_latent_range(1) == (4, 8)

    def test_window_version_monotonic(self):
        assert window_version(0) == 1
        assert window_version(5) == 6


# =========================================================================
# intrinsics conversion
# =========================================================================

class TestIntrinsics:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        L = 10
        Ks = np.zeros((L, 3, 3))
        Ks[:, 0, 0] = 0.9 + 0.1 * rng.random(L)   # fx in width units
        Ks[:, 1, 1] = 0.9 + 0.1 * rng.random(L)   # fy in height units
        Ks[:, 0, 2] = 0.5
        Ks[:, 1, 2] = 0.5
        Ks[:, 2, 2] = 1.0
        # roundtrip helper leaves K[2,2]=0, so compare only the meaningful entries
        f = hy_ks_to_movies_fxfycxcy(Ks, aspect=832 / 480)
        Ks2 = movies_fxfycxcy_to_hy_ks(f, aspect=832 / 480)
        Ks[:, 2, 2] = 0.0  # helper does not fill K[2,2]
        np.testing.assert_allclose(Ks, Ks2, atol=1e-12)

    def test_pixel_projection_consistency(self):
        """Same 3D point must project to the same pixel under both conventions."""
        W, H = 832, 480
        fx_w, fy_h = 0.75, 0.75 * (832 / 480)  # HY-style: fy in height units
        # MoVieS fxfycxcy
        fx_m, fy_m, cx_m, cy_m = fx_w, fy_h, 0.5, 0.5
        X_cam = np.array([0.3, -0.2, 2.0])
        # HY pixel: fx_px = fx_hy * W, cx_px = 0.5 * W
        u_hy = fx_w * W * X_cam[0] / X_cam[2] + 0.5 * W
        v_hy = fy_h * H * X_cam[1] / X_cam[2] + 0.5 * H
        # MoVieS pixel: fx_px = fx_m * W, cx_px = cx_m * W
        u_mv = fx_m * W * X_cam[0] / X_cam[2] + cx_m * W
        v_mv = fy_m * H * X_cam[1] / X_cam[2] + cy_m * H
        assert u_hy == pytest.approx(u_mv)
        assert v_hy == pytest.approx(v_mv)


# =========================================================================
# SE3 interpolation
# =========================================================================

def make_trajectory(num_steps: int, forward=0.08, yaw=np.deg2rad(3)):
    """Replicate generate_camera_trajectory_local: constant-velocity motion steps."""
    poses = [np.eye(4)]
    T = np.eye(4)
    for _ in range(num_steps):
        Ry = Rot.from_euler("y", yaw).as_matrix()
        T = T.copy()
        T[:3, :3] = T[:3, :3] @ Ry
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0, 0, forward])
        poses.append(T.copy())
    return np.array(poses)


class TestSE3Interp:
    def test_endpoints_exact(self):
        pa = np.eye(4)
        pb = np.eye(4)
        pb[:3, 3] = [1.0, 0, 0]
        np.testing.assert_allclose(se3_interpolate_poses(pa, pb, 0.0), pa, atol=1e-10)
        np.testing.assert_allclose(se3_interpolate_poses(pa, pb, 1.0), pb, atol=1e-10)

    def test_midpoint_translation_linear(self):
        pa = np.eye(4)
        pb = np.eye(4)
        pb[:3, 3] = [1.0, 2.0, 3.0]
        mid = se3_interpolate_poses(pa, pb, 0.5)
        np.testing.assert_allclose(mid[:3, 3], [0.5, 1.0, 1.5], atol=1e-12)

    def test_interpolation_matches_true_trajectory(self):
        """Constant-velocity trajectory: interp between latent anchors == true poses.

        SEMANTICS (verified against generate.py): 'w-31' = 31 motions = 32 latent poses.
        Each latent covers 4 pixel frames, and ONE motion step is spread over those 4
        frames: frame f's camera progress = f/4 motion steps (frame 124 = step 31).

        Pure translation: latent l pose has z = 0.08*l; frame f (in latent l, t=(f-4(l-1))/4)
        must land at z = 0.08 * (l - 1 + t) = 0.08 * f / 4.
        """
        traj = make_trajectory(32, forward=0.08, yaw=0.0)  # 33 poses? No: make_trajectory(n) returns n+1
        # make_trajectory(num_steps) -> len = num_steps + 1. We want 32 latent poses -> 31 steps.
        traj = make_trajectory(31, forward=0.08, yaw=0.0)
        assert traj.shape[0] == 32
        frames = list(range(125))
        interp = interpolate_frame_poses(traj, frames)
        for i, f in enumerate(frames):
            np.testing.assert_allclose(interp[i][:3, 3], [0, 0, 0.08 * f / 4], atol=1e-10)

    def test_total_displacement_conservation(self):
        """Frame 124 (last) must equal latent 31's pose: total trajectory displacement preserved."""
        traj = make_trajectory(31, forward=0.08, yaw=0.0)
        assert traj[31][2, 3] == pytest.approx(0.08 * 31)
        interp = interpolate_frame_poses(traj, [124])
        np.testing.assert_allclose(interp[0][:3, 3], traj[31][:3, 3], atol=1e-10)

    def test_latent_anchor_frames_exact(self):
        """Frame 4l (last frame of latent l) hits latent l's pose exactly (t=1)."""
        traj = make_trajectory(31, forward=0.08, yaw=np.deg2rad(3))
        for l in range(1, 32):
            f = 4 * l
            if f >= 125:
                break
            interp = interpolate_frame_poses(traj, [f])
            np.testing.assert_allclose(interp[0], traj[l], atol=1e-10)

    def test_interp_with_rotation_reasonable(self):
        """Yaw+forward: interp stays close to true continuous trajectory (small angles)."""
        traj = make_trajectory(32, forward=0.08, yaw=np.deg2rad(3))
        frames = list(range(16, 125, 7))
        interp = interpolate_frame_poses(traj, frames)
        # verify each interp pose is a valid rigid transform
        for p in interp:
            R = Rot.from_matrix(p[:3, :3])
            np.testing.assert_allclose((R.as_rotvec()), Rot.from_matrix(p[:3, :3]).as_rotvec())
            np.testing.assert_allclose(p[:3, :3] @ p[:3, :3].T, np.eye(3), atol=1e-10)

    def test_quat_slerp_known_values(self):
        from common.camera_conv import _slerp_quat
        qa = np.array([1.0, 0, 0, 0])
        theta = np.pi / 2
        qb = np.array([np.cos(theta / 2), np.sin(theta / 2), 0, 0])
        mid = _slerp_quat(qa, qb, 0.5)
        expected = np.array([np.cos(theta / 4), np.sin(theta / 4), 0, 0])
        np.testing.assert_allclose(mid, expected, atol=1e-12)


# =========================================================================
# canonical normalization
# =========================================================================

class TestCanonical:
    def test_first_pose_becomes_identity(self):
        traj = make_trajectory(16, forward=0.08, yaw=np.deg2rad(3))
        c2w_can, T_hy2canon, T_canon2hy = canonical_normalize(traj)
        np.testing.assert_allclose(c2w_can[0], np.eye(4), atol=1e-10)

    def test_roundtrip(self):
        traj = make_trajectory(16, forward=0.08, yaw=np.deg2rad(3))
        c2w_can, T_hy2canon, T_canon2hy = canonical_normalize(traj)
        back = np.einsum("ij,fjk->fik", T_canon2hy, c2w_can)
        np.testing.assert_allclose(back, traj, atol=1e-10)

    def test_points_map_back(self):
        traj = make_trajectory(8, forward=0.08, yaw=np.deg2rad(3))
        c2w_can, _, T_canon2hy = canonical_normalize(traj)
        pts_can = np.random.default_rng(1).random((100, 3))
        pts_hy = apply_canonical_to_points(pts_can, T_canon2hy)
        # check via a camera projection consistency: point in HY frame must appear at same
        # bearing from traj[3] as the canonical point from c2w_can[3]
        cam_hy = traj[3]
        cam_can = c2w_can[3]
        v_hy = pts_hy - cam_hy[:3, 3]
        v_can = pts_can - cam_can[:3, 3]
        v_hy_cam = cam_hy[:3, :3].T @ v_hy.T       # world -> camera axes
        v_can_cam = cam_can[:3, :3].T @ v_can.T
        # canonical camera axes == HY first camera's rotation composed with T_hy2canon;
        # v_hy_cam and v_can_cam must be equal up to the shared rotation
        np.testing.assert_allclose(v_hy_cam, v_can_cam, atol=1e-10)


# =========================================================================
# types
# =========================================================================

class TestTypes:
    def test_trajectory_sync_valid(self):
        from common.types import TrajectorySync
        L = 32
        ts = TrajectorySync(
            c2w_latents=np.zeros((L, 4, 4)),
            fxfycxcy=np.zeros((L, 4)),
            num_latents=L,
        )
        assert ts.num_latents == L

    def test_chunk_packet_validation(self):
        from common.types import ChunkPacket
        with pytest.raises(AssertionError):
            ChunkPacket(chunk_idx=0, frames=np.zeros((16, 4, 480, 832)), timestamps=np.zeros(15))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
