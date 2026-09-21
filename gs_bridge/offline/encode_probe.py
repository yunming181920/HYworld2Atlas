"""M1 offline probe: feed HY video + interpolated poses to MoVieS, measure re-projection.

Quality gate for the encoder path (DESIGN.md M1): take an HY-generated video (or any
monocular walkthrough video + its pose json), build the per-frame poses exactly as the
duplex pipeline would (latent table -> SE3 interpolation), encode one window per chunk
with MoVieS, then render the INPUT camera poses back and compute PSNR/SSIM against the
original frames. This isolates encode quality from query pose choice.

Usage (inside the MoVieS conda env, from the workspace root):
    python -m gs_bridge.offline.encode_probe \
        --frames_dir <dir of rgb frames from HY output> \
        --pose_json <HY pose json (extrinsic c2w + K per latent)> \
        [--num_latents 32] [--chunk_stride 1] [--out gs_bridge/out/probe]

Requires: MoVieS repo + ckpt (see BridgeConfig), GPU.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from gs_bridge.common.camera_conv import (
    hy_ks_to_movies_fxfycxcy,
    interpolate_frame_poses,
)
from gs_bridge.common.time_map import (
    chunk_frame_range,
    frame_to_latent,
    num_chunks,
    num_frames_from_latents,
)


def load_frames(frames_dir: str, height: int, width: int) -> np.ndarray:
    import cv2  # MoVieS env has opencv

    files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    assert files, f"no frames in {frames_dir}"
    out = []
    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        out.append(img.astype(np.float32) / 255.0)
    return np.stack(out)  # (F, H, W, 3)


def load_pose_json(pose_json_path: str, num_latents: int):
    """HY pose json -> (c2w_latents (L,4,4) in HY world, fxfycxcy (L,4) MoVieS conv)."""
    with open(pose_json_path) as f:
        pose_json = json.load(f)
    keys = sorted(pose_json.keys(), key=int)
    assert len(keys) == num_latents, f"{len(keys)} poses != {num_latents} latents"

    c2w = np.stack([np.array(pose_json[k]["extrinsic"], dtype=np.float64) for k in keys])
    # HY K -> normalized Ks (replicate generate.py:217-221) -> MoVieS fxfycxcy
    Ks = np.stack([
        np.array(pose_json[k]["K"], dtype=np.float64) for k in keys
    ])
    for i in range(len(Ks)):
        Ks[i][0, 0] /= Ks[i][0, 2] * 2
        Ks[i][1, 1] /= Ks[i][1, 2] * 2
        Ks[i][0, 2] = 0.5
        Ks[i][1, 2] = 0.5
    aspect = None  # fxfycxcy conversion needs no aspect with centered principal point
    from gs_bridge.common.camera_conv import hy_ks_to_movies_fxfycxcy
    fxfy = hy_ks_to_movies_fxfycxcy(Ks, aspect=1.0)
    return c2w, fxfy


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--num_latents", type=int, default=32)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--chunk_stride", type=int, default=1,
                    help="encode every k-th chunk (stride>1 = faster probe)")
    ap.add_argument("--out", default="gs_bridge/out/probe")
    ap.add_argument("--no_ckpt", action="store_true", help="skip ckpt (shape dry-run)")
    args = ap.parse_args()

    import torch  # deferred: MoVieS env only

    L = args.num_latents
    n_frames = num_frames_from_latents(L)
    frames = load_frames(args.frames_dir, args.height, args.width)
    n_frames = min(n_frames, frames.shape[0])
    frames = frames[:n_frames]
    print(f"probe: {frames.shape[0]} frames, {L} latents, {num_chunks(L)} chunks")

    c2w_latents, fxfy = load_pose_json(args.pose_json, L)
    # per-frame poses exactly as the pipeline would build them
    frame_poses = interpolate_frame_poses(c2w_latents, list(range(n_frames)))

    from gs_bridge.asset_server.encoder import MoVieSEncoder
    ckpt = None if args.no_ckpt else "MoVieS/resources/movies_ckpt.safetensors"
    enc = MoVieSEncoder(movies_repo="MoVieS", ckpt=ckpt)

    os.makedirs(args.out, exist_ok=True)
    results = []
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (F,3,H,W)

    for c in range(0, num_chunks(L), args.chunk_stride):
        fs, fe = chunk_frame_range(c)
        fe = min(fe, n_frames)
        if fs >= n_frames:
            break
        win_frames = frames_t[fs:fe]
        win_poses = frame_poses[fs:fe]
        win_ts = np.arange(fs, fe, dtype=np.float64)
        win_fxfy = fxfy[frame_to_latent(fs): frame_to_latent(fe - 1) + 1]

        wg = enc.encode_window(
            win_frames, win_poses, fxfy[0], win_ts, chunk_idx=c
        )
        # re-render each input frame's pose (canonical frame == input poses' frame)
        n_valid = 0
        psnrs = []
        for j in range(fe - fs):
            mo = dict(wg.model_outputs)
            if wg.pred_motions is not None:
                mo["offset"] = wg.pred_motions[:, j, :, :3, ...]
            if wg.pred_motion_gs is not None:
                mo.update(wg.pred_motion_gs[j])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = enc.model.gs_renderer.render(
                    mo,
                    wg.input_c2w,
                    wg.input_fxfycxcy,
                    wg.input_c2w[:, j: j + 1],
                    wg.input_fxfycxcy[:, j: j + 1],
                    height=args.height,
                    width=args.width,
                )
            rec = out["image"][0, 0].float().clamp(0, 1).cpu().numpy()
            gt = frames[fs + j]
            psnrs.append(psnr(rec, gt))
            n_valid += 1
        chunk_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
        results.append((c, chunk_psnr, len(psnrs)))
        print(f"chunk {c:3d}: frames [{fs},{fe}) re-projection PSNR = {chunk_psnr:.2f} dB "
              f"(gaussians ~{enc.gaussian_count(wg)})")

    with open(os.path.join(args.out, "probe_results.txt"), "w") as f:
        for c, p, n in results:
            f.write(f"chunk {c}: psnr {p:.3f} dB over {n} frames\n")
    all_p = [p for _, p, _ in results if np.isfinite(p)]
    if all_p:
        print(f"\nMEAN re-projection PSNR: {np.mean(all_p):.2f} dB  "
              f"(min {np.min(all_p):.2f}, max {np.max(all_p):.2f})")


if __name__ == "__main__":
    main()
