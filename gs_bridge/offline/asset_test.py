"""M2 offline test: GlobalAsset ring buffer + AssetQuerier with a mocked renderer.

GPU-free structural test: a fake encoder/renderer (correct shapes, zero work) lets us
verify the full M2 data path -- insert/evict/version-cap/query-planning/pose-to-
canonical math -- before any GPU run. Latency numbers need the real renderer (GPU).

Run:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest gs_bridge/tests/test_m2_struct.py -v
      or: python -m gs_bridge.offline.asset_test
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from gs_bridge.common.types import TrajectorySync


class FakeWindowGaussians:
    """Shape-compatible stand-in for WindowGaussians (no torch tensors beyond numpy)."""

    def __init__(self, chunk_idx, n_frames=16, n_out=16):
        self.model_outputs = {"depth": np.zeros((1, n_frames, 1, 224, 224))}
        self.pred_motions = np.zeros((1, n_out, n_frames, 4, 224, 224))
        self.pred_motion_gs = [{"motion_conf": np.zeros((1, n_frames, 1, 224, 224))} for _ in range(n_out)]
        first_c2w = np.eye(4)
        self.T_canon2hy = first_c2w
        self.input_c2w = np.zeros((1, n_frames, 4, 4))
        self.input_fxfycxcy = np.zeros((1, n_frames, 4))
        self.output_timesteps = np.linspace(0, 1, n_out)[None]
        self.chunk_idx = chunk_idx


class FakeRenderer:
    def __init__(self):
        self.calls = 0

    def render(self, mo, in_c2w, in_fxfy, C2W_q, FXFY_q, height, width):
        self.calls += 1
        return {"image": np.zeros((1, 1, 3, height, width)), "depth": np.zeros((1, 1, 1, height, width))}


def main():
    from gs_bridge.asset_server.asset import GlobalAsset
    from gs_bridge.asset_server.query import AssetQuerier

    n_latents = 64
    asset = GlobalAsset(capacity_windows=20)
    renderer = FakeRenderer()
    traj = TrajectorySync(
        c2w_latents=np.tile(np.eye(4), (n_latents, 1, 1)),
        fxfycxcy=np.tile(np.array([0.75, 0.75, 0.5, 0.5]), (n_latents, 1)),
        num_latents=n_latents,
    )
    querier = AssetQuerier(asset, renderer, traj)

    # -- insert 30 windows: only last 20 survive (capacity) --
    for c in range(30):
        asset.insert_window(FakeWindowGaussians(c))
    assert len(asset) == 20, f"capacity eviction failed: {len(asset)}"
    assert asset.version == 30
    assert asset.stats["evicted"] == 10
    print("[ok] FIFO eviction: 30 inserted, 10 evicted, 20 resident")

    # -- version cap: while denoising chunk t=25, cap = 24 (t-1 per time_map) --
    # windows cover chunks 10..29; latents covered: 40..120
    slots = asset.visible_slots(version_cap=24)
    assert all(s.version <= 24 for s in slots)
    assert max(s.version for s in slots) == 24
    print(f"[ok] version cap 24 -> {len(slots)} visible windows (max version 24)")

    # -- covering window lookup --
    slot = asset.find_window_for_latent(50, version_cap=24)  # latent 50 -> chunk 12
    assert slot.chunk_idx == 12, f"covering window wrong: {slot.chunk_idx}"
    # exact hit for in-buffer chunk under version cap
    slot = asset.find_window_for_latent(44, version_cap=24)  # chunk 11, version 12
    assert slot.chunk_idx == 11
    # evicted chunk (0..9 evicted by FIFO): latent 30 (chunk 7) -> fallback to chunk 10
    slot = asset.find_window_for_latent(30, version_cap=24)
    assert slot.chunk_idx == 10, f"fallback window should be chunk 10, got {slot.chunk_idx}"
    print(f"[ok] latent->window: exact (50->12, 44->11); evicted (30->fallback {slot.chunk_idx})")

    # -- query planning (mocked renderer) --
    t0 = time.perf_counter()
    res = querier.query(
        latent_indices=np.array([52, 53, 60, 70]),
        version_cap=24,
        height=480, width=832, with_depth=True,
    )
    dt = time.perf_counter() - t0
    assert res is not None
    assert res.images.shape == (4, 3, 480, 832)
    assert res.depths.shape == (4, 1, 480, 832)
    assert renderer.calls == 4
    print(f"[ok] query 4 views (mock renderer): plan+dispatch {dt*1000:.2f} ms, "
          f"versions used <= {res.version}")

    # -- empty asset under early cap --
    empty = GlobalAsset()
    q2 = AssetQuerier(empty, renderer, traj)
    assert q2.query(np.array([0, 1]), version_cap=0) is None
    print("[ok] empty-asset query returns None")

    # -- nearest time index mapping: latent 52 in chunk 13 (latents 52..55) -> u=0 -> t=0
    mo_out = asset.find_window_for_latent(52, version_cap=24).window.output_timesteps
    n_out = mo_out.shape[1]
    ti = querier._nearest_time_index(
        FakeWindowGaussians(13, n_out=n_out), 52)
    assert ti == 0, f"latent 52 (u=0) should map to t=0, got {ti}"
    ti = querier._nearest_time_index(FakeWindowGaussians(13, n_out=n_out), 55)
    assert ti == n_out - 1, f"latent 55 (u=1) should map to t=n-1, got {ti}"
    print("[ok] latent->window-local time mapping (edges)")

    print("\nAll M2 structural checks passed. (Latency/memory: run on GPU with real renderer.)")


if __name__ == "__main__":
    main()
