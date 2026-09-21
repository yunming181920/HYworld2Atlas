"""M3 offline Atlas preview: frozen-time orbit camera around the asset.

Implements idea.txt point 4 (Atlas-style observation): freeze time at the latest
complete asset version, sweep an orbit camera around the reconstructed scene, and
write an mp4. Background updates are NOT modeled here (offline, single encode) --
the online double-buffered variant lives in asset_server/atlas.py.

Usage (MoVieS env, workspace root):
    python -m gs_bridge.offline.atlas_preview \
        --frames_dir <HY frames> --pose_json <HY pose json> \
        [--num_latents 32] [--orbit_yaw 60] [--orbit_n 90] [--out gs_bridge/out/atlas]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from gs_bridge.common.camera_conv import interpolate_frame_poses
from gs_bridge.common.time_map import chunk_frame_range, num_chunks, num_frames_from_latents

from gs_bridge.offline.encode_probe import load_frames, load_pose_json


def orbit_poses(center: np.ndarray, radius: float, n: int, max_yaw_deg: float,
                height_offset: float = 0.3, base_c2w: np.ndarray = np.eye(4)) -> np.ndarray:
    """Turntable orbit around `center`, inheriting look-at orientation from base camera.

    Simple v1: cameras at fixed pitch looking at the scene center, yaw sweeping
    +/- max_yaw_deg around the base camera's viewing axis.
    """
    R = np.deg2rad(max_yaw_deg)
    yaws = np.linspace(-R, R, n)
    # base camera's forward (z axis of c2w, OpenCV convention)
    forward = base_c2w[:3, 2] / np.linalg.norm(base_c2w[:3, 2])
    up_world = np.array([0.0, -1.0, 0.0])  # HY world down is -y (OpenCV y-down in camera)
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    poses = []
    eye0 = base_c2w[:3, 3] + forward * radius * 0.0  # start from base position
    for yaw in yaws:
        # rotate the camera around the center by yaw (right axis through center)
        eye = center + (np.cos(yaw) * (eye0 - center) + np.sin(yaw) * np.cross(up_world, eye0 - center))
        eye = eye + up_world * height_offset * 0
        # look at center
        f = center - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, cam_up)
        if np.linalg.norm(r) < 1e-6:
            r = right
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([r, u, f], axis=1)  # OpenCV: x-right, y-down, z-forward
        c2w[:3, 3] = eye
        poses.append(c2w)
    return np.stack(poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--num_latents", type=int, default=32)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--chunk", type=int, default=-1, help="which chunk to freeze on (-1 = last)")
    ap.add_argument("--orbit_yaw", type=float, default=60.0)
    ap.add_argument("--orbit_n", type=int, default=90)
    ap.add_argument("--orbit_radius", type=float, default=1.0,
                    help="orbit radius in HY world units (scene scale ~2.5 for w-31)")
    ap.add_argument("--out", default="gs_bridge/out/atlas")
    ap.add_argument("--no_ckpt", action="store_true")
    args = ap.parse_args()

    import torch
    import imageio.v2 as iio

    L = args.num_latents
    n_frames = num_frames_from_latents(L)
    frames = load_frames(args.frames_dir, args.height, args.width)
    n_frames = min(n_frames, frames.shape[0])
    c2w_latents, fxfy = load_pose_json(args.pose_json, L)
    frame_poses = interpolate_frame_poses(c2w_latents, list(range(n_frames)))

    from gs_bridge.asset_server.encoder import MoVieSEncoder
    ckpt = None if args.no_ckpt else "MoVieS/resources/movies_ckpt.safetensors"
    enc = MoVieSEncoder(movies_repo="MoVieS", ckpt=ckpt)

    # encode the target chunk
    c = args.chunk if args.chunk >= 0 else num_chunks(L) - 1
    fs, fe = chunk_frame_range(c)
    fe = min(fe, n_frames)
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    win_poses = frame_poses[fs:fe]
    wg = enc.encode_window(
        frames_t[fs:fe], win_poses, fxfy[0], np.arange(fs, fe, dtype=np.float64), chunk_idx=c
    )
    print(f"encoded chunk {c} (frames [{fs},{fe})) -> orbiting")

    # orbit around a scene point ~1 unit ahead of the chunk's mid camera
    mid_pose = win_poses[len(win_poses) // 2]
    center = mid_pose[:3, 3] + mid_pose[:3, 2] * 1.0  # 1 unit along forward
    orbit = orbit_poses(center, args.orbit_radius, args.orbit_n, args.orbit_yaw, base_c2w=mid_pose)

    # freeze time at the last output timestep (latest state of the chunk)
    time_idx = wg.output_timesteps.shape[1] - 1
    mo = dict(wg.model_outputs)
    if wg.pred_motions is not None:
        mo["offset"] = wg.pred_motions[:, time_idx, :, :3, ...]
    if wg.pred_motion_gs is not None:
        mo.update(wg.pred_motion_gs[time_idx])

    os.makedirs(args.out, exist_ok=True)
    video = []
    # orbit poses are in HY world -> window canonical
    T_hy2canon = np.linalg.inv(wg.T_canon2hy)
    for i in range(args.orbit_n):
        c2w_can = T_hy2canon @ orbit[i]
        C2W_q = torch.from_numpy(c2w_can).float()[None, None].to(
            device="cuda", dtype=torch.bfloat16)
        FXFY_q = torch.from_numpy(fxfy[0]).float()[None, None].to(
            device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = enc.model.gs_renderer.render(
                mo, wg.input_c2w, wg.input_fxfycxcy, C2W_q, FXFY_q,
                height=args.height, width=args.width,
            )
        img = out["image"][0, 0].float().clamp(0, 1).cpu().numpy()
        video.append((img.transpose(1, 2, 0) * 255).astype(np.uint8))

    path = os.path.join(args.out, f"atlas_chunk{c}.mp4")
    iio.mimwrite(path, video, fps=30, macro_block_size=1, quality=8)
    print(f"saved {path} ({args.orbit_n} views, yaw +/-{args.orbit_yaw} deg)")


if __name__ == "__main__":
    main()
