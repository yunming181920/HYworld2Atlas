"""M5 end-to-end entry: HY generation with the gs_bridge memory swap.

Runs INSIDE the HY environment. Mirrors hyvideo/generate.py's setup but routes
through the bridge:
    1. build the pipeline exactly as generate.py does (create_pipeline)
    2. connect the AssetClient and sync the trajectory (from pose_to_input)
    3. install(pipe, client, mode)  -- fork takes over the AR rollout
    4. run pipe(...) as usual; the fork swaps memory latents and pushes chunks
    5. compare output vs the vanilla run (M6 experiment basis)

The asset server (GPU B, MoVieS env) must already be listening:
    python -m gs_bridge.asset_server.server --port 9820

Usage (HY env, workspace root):
    python -m gs_bridge.hy_client.run_with_bridge \
        --image <path> --pose "w-31" --model_type ar \
        [--bridge_mode fov_kept|future] [--vanilla]  # --vanilla = no bridge (baseline)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "HY-WorldPlay"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pose", default="w-31")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--model_type", default="ar", choices=["ar", "bi"])
    ap.add_argument("--num_frames", type=int, default=125)
    ap.add_argument("--bridge_mode", default="fov_kept", choices=["fov_kept", "future"])
    ap.add_argument("--vanilla", action="store_true", help="disable the bridge (baseline)")
    ap.add_argument("--server_host", default="127.0.0.1")
    ap.add_argument("--server_port", type=int, default=9820)
    ap.add_argument("--output", default="gs_bridge/out/hy_bridge")
    args = ap.parse_args()

    # --- HY imports (HY env) ---
    from hyvideo.generate import generate_video  # reuses HY's own entry plumbing
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    # --- build inputs exactly as generate.py: pose -> viewmats/Ks/action ---
    import numpy as np
    from hyvideo.generate import pose_to_input

    latent_num = (args.num_frames + 3) // 4
    w2c_list, Ks_list, action_one_label = pose_to_input(args.pose, latent_num)
    viewmats = torch.tensor(w2c_list).unsqueeze(0)      # (1, L, 4, 4)
    Ks = torch.tensor(Ks_list).float().unsqueeze(0)     # (1, L, 3, 3)

    if not args.vanilla:
        from gs_bridge.hy_client.client import AssetClient
        from gs_bridge.hy_client.memory_replace import install

        client = AssetClient(host=args.server_host, port=args.server_port)
        n = client.sync_trajectory(viewmats[0].numpy(), Ks[0].numpy())
        print(f"[run_with_bridge] trajectory synced: {n} latents")

    # --- run HY generation; install before the call so the rollout picks it up ---
    # generate_video() builds the pipeline internally; we use its hook-less flow by
    # calling it with a pipeline factory override -- v1 documents this as the
    # integration point: the cleanest path is constructing the pipeline here
    # (same args as run.sh) and calling pipe() directly. The exact construction
    # sequence is long; defer to generate.generate_video for v1 by monkey-patching
    # the pipeline class's __call__ to install on first invocation:
    if not args.vanilla:
        _orig_call = HunyuanVideo_1_5_Pipeline.__call__

        def _call_with_install(self, *a, **kw):
            install(self, client, mode=args.bridge_mode)
            return _orig_call(self, *a, **kw)

        HunyuanVideo_1_5_Pipeline.__call__ = _call_with_install

    os.makedirs(args.output, exist_ok=True)
    generate_video(
        model_type=args.model_type,
        prompt=args.prompt,
        image_path=args.image,
        pose=args.pose,  # generate_video re-derives viewmats internally; the fork
                         # consumes the SAME tensors the pipeline builds from this
        num_frames=args.num_frames,
        # remaining kwargs default as in run.sh
    )

    if not args.vanilla:
        HunyuanVideo_1_5_Pipeline.__call__ = _orig_call
        client.close()
    print(f"[run_with_bridge] done -> {args.output}")


if __name__ == "__main__":
    import torch  # noqa: F401 (HY env)
    main()
