"""M7 entry: launch the player UI + driver together.

Two-process layout (see player.py / driver.py docstrings):
    1. driver process (HY env): builds the pipeline exactly like generate.py,
       installs the bridge, runs HybridDriver
    2. UI process (any env with pygame+zmq): PlayerWindow

Run (workspace root):
    # terminal 1 (HY env): python -m gs_bridge.ui.run_ui --driver-only ...
    # terminal 2 (any env): python -m gs_bridge.ui.run_ui --ui-only
    # or both at once (requires the HY env to also have pygame):
    python -m gs_bridge.ui.run_ui --image <path> --pose "w-31"

The asset server (GPU B) must already be listening:
    python -m gs_bridge.asset_server.server
"""

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="I2V reference image (driver mode)")
    ap.add_argument("--pose", default="w-31", help="initial trajectory (driver mode)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--bridge_mode", default="fov_kept", choices=["fov_kept", "future"])
    ap.add_argument("--server_port", type=int, default=9820)
    ap.add_argument("--ui-only", action="store_true")
    ap.add_argument("--driver-only", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    if args.ui_only:
        from gs_bridge.ui.player import main as ui_main
        ui_main()
        return

    if args.driver_only:
        run_driver(args)
        return

    # both: driver as a subprocess, UI in-process
    drv_cmd = [sys.executable, "-m", "gs_bridge.ui.driver_loop",
               "--image", str(args.image), "--pose", args.pose,
               "--bridge_mode", args.bridge_mode,
               "--server_port", str(args.server_port)]
    proc = subprocess.Popen(drv_cmd, cwd=os.path.join(here, "..", ".."))
    try:
        from gs_bridge.ui.player import main as ui_main
        ui_main()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def run_driver(args):
    """Driver-mode entry (HY env): same flow as run_with_bridge + HybridDriver."""
    from gs_bridge.hy_client.client import AssetClient
    from gs_bridge.hy_client.memory_replace import install
    from gs_bridge.ui.driver import HybridDriver

    import torch
    from hyvideo.generate import pose_to_input

    # trajectory for chunk 0.. (pose string -> viewmats/Ks)
    latent_num = 32  # initial rollout length; extended by pending motions
    w2c_list, Ks_list, _ = pose_to_input(args.pose, latent_num)

    # build pipeline as run.sh does (delegated to HY's own create plumbing --
    # the driver_loop module holds the concrete construction, env-dependent)
    from gs_bridge.ui.driver_loop import build_pipeline_and_rollout
    pipe, bridge = build_pipeline_and_rollout(
        image=args.image, prompt=args.prompt, pose=args.pose,
        bridge_mode=args.bridge_mode, server_port=args.server_port,
        initial_w2c=w2c_list, initial_ks=Ks_list,
    )
    driver = HybridDriver(pipe, bridge, initial_pose_json=args.pose)
    bridge.ui_driver = driver  # rollout_fork block B routes frames through this
    driver.set_status("generating")
    pipe(image_path=args.image, prompt=args.prompt, pose=args.pose)
    driver.on_rollout_finished()


if __name__ == "__main__":
    main()
