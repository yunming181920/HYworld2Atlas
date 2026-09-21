"""Attach the gs_bridge memory source to a running HY pipeline (M5 install point).

The heavy lifting lives in rollout_fork.py (a direct fork of
`_ar_rollout_inner` with two marked blocks). This module only:
    - builds the MemorySource (client + latent prep + mode)
    - swaps `pipe._ar_rollout_inner` for the fork (instance attribute -- the
      original `ar_rollout` looks it up on self, so the fork takes over
      transparently; HY repo files stay untouched)
    - provides MemorySource.push_finished_chunk: decode the finished chunk and
      ship it to the asset server (CHUNK_UPDATE), the duplex update path
"""

import numpy as np
import torch

from gs_bridge.common.time_map import chunk_frame_range
from gs_bridge.hy_client.client import (
    AssetClient,
    select_memory_latents_future,
    version_cap_for_chunk,
)
from gs_bridge.hy_client.latent_prep import LatentPrep


class MemorySource:
    """Bridge state held on the pipeline (pipe.gs_bridge)."""

    def __init__(self, client: AssetClient, pipe, mode: str = "fov_kept",
                 memory_slots: int = 20, render_height: int = 480, render_width: int = 832):
        assert mode in ("fov_kept", "future")
        self.client = client
        self.pipe = pipe
        self.mode = mode
        self.memory_slots = memory_slots
        self.render_height = render_height
        self.render_width = render_width
        self.latent_prep = LatentPrep(pipe)
        self.enabled = True
        self.fallback_count = 0
        self.served_count = 0
        self.pushed_chunks = 0
        # viewpoint-reset bookkeeping: bump on every reset; the running rollout
        # captured the value at entry and stops pushing when it changes (its
        # chunks are old-trajectory content and must not enter the asset).
        self.generation = 0
        self.reset_count = 0

    # ------------------------------------------------------------------
    # query side (called from rollout_fork block A)
    # ------------------------------------------------------------------

    def query_pose_latents(self, chunk_i: int, selected_frame_indices) -> list:
        """Latent indices for rendering; fov_kept keeps HY's FOV selection."""
        if self.mode == "future":
            return select_memory_latents_future(
                chunk_i, chunk_latent_frames=4, memory_slots=self.memory_slots
            )
        return list(selected_frame_indices)

    def render_memory_latents(self, chunk_i: int, latent_indices: list,
                              device, dtype):
        """Asset render for the given latents -> context latents (1, C, L, h, w) | None."""
        cap = version_cap_for_chunk(chunk_i)
        if cap <= 0:
            return None
        render = self.client.query_views(
            version_cap=cap,
            latent_indices=latent_indices,
            height=self.render_height,
            width=self.render_width,
            with_depth=False,
        )
        if render is None:
            return None
        latents = self.latent_prep.encode_views(render.images, device, dtype)
        wanted = self.latent_prep.latent_count(len(latent_indices))
        latents = self.latent_prep.drop_pad_latents(latents, wanted)
        if latents.shape[2] != len(latent_indices):
            latents = latents[:, :, : len(latent_indices)]
        self.served_count += 1
        return latents

    # ------------------------------------------------------------------
    # update side (called from rollout_fork block B)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def push_finished_chunk(self, pipe, chunk_i: int, latents, device) -> None:
        """Decode the finished chunk's latents and push frames to the server.

        Decode cost overlaps the next chunk's denoising on GPU A only if pushed
        asynchronously; v1 decodes synchronously here (correctness first). The
        server encodes while the DiT denoises chunk t+1 -- the duplex pipeline
        already holds without client-side threads.
        """
        start_idx = chunk_i * pipe.chunk_latent_frames
        end_idx = start_idx + pipe.chunk_latent_frames
        chunk_latents = latents[:, :, start_idx:end_idx]

        # HY VAE decode: latents (1,C,L,h,w) -> frames (1,3,F,H,W)
        frames = pipe.vae.decode(chunk_latents.to(pipe.vae.dtype))
        if hasattr(frames, "sample"):
            frames = frames.sample
        frames = (frames[0].float() / 2.0 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        frames_np = frames.permute(1, 0, 2, 3).cpu().numpy().astype(np.float32)  # (F,3,H,W)

        self.client.push_chunk(chunk_idx=chunk_i, frames=frames_np)
        self.pushed_chunks += 1

    # ------------------------------------------------------------------
    # viewpoint reset (idea.txt point 4 -- closed loop; DESIGN.md section 4)
    # ------------------------------------------------------------------

    def viewpoint_reset(self, new_pose_c2w, new_fxfycxcy, new_viewmats, new_Ks,
                        t_current: int):
        """Abandon the current rollout; prepare the server for a new-viewpoint one.

        Semantics (user's decision, 2026-09-21): switching the viewpoint restarts
        generation from t-1. The in-flight chunks (t-1, t) are FULLY discarded --
        their content was conditioned on old-trajectory context and must not
        enter the asset. The asset rolls back to the t-2 trusted boundary
        (version = t_current - 1... careful: while chunk t is being denoised the
        guaranteed-complete boundary is (t-1) + 1 = t; rolling back to the last
        OLD-trajectory version means: version of chunk (t-1) minus one in-flight:

            rollback_version = (t_current - 1)  -> keeps chunks <= t_current - 2,
            i.e. exactly the idea.txt "t-2 asset" under the new viewpoint.

        After rollback the reference frame for the new rollout is rendered at
        `new_pose_c2w` from the rolled-back asset.

        Args:
            new_pose_c2w: (4,4) HY-world c2w of the new viewpoint (first pose of
                the new trajectory).
            new_fxfycxcy: (4,) MoVieS-convention intrinsics at the new viewpoint.
            new_viewmats/new_Ks: the NEW trajectory's w2c/Ks tables (HY convention),
                chunk-numbered from latent 0.
            t_current: chunk index being denoised when the user switched.

        Returns:
            (reference_frame (3,H,W) | None, asset_version). None = empty asset.
        """
        rollback_version = max(t_current - 1, 0)
        reference, version = self.client.viewpoint_reset(
            rollback_version=rollback_version,
            c2w=new_pose_c2w,
            fxfycxcy=new_fxfycxcy,
            height=self.render_height,
            width=self.render_width,
        )
        # new trajectory: server pose caches invalidate, querier re-binds
        self.client.resync_trajectory(new_viewmats, new_Ks)
        # generation bump: the abandoned rollout's remaining pushes are dropped
        # (guard in rollout_fork block B); new chunks number from 0 and insert
        # cleanly above the rolled-back version.
        self.generation += 1
        self.reset_count += 1
        print(f"[gs_bridge] viewpoint reset: rollback->v{version}, "
              f"generation {self.generation}, reference "
              f"{'rendered' if reference is not None else 'EMPTY (asset has no windows)'}")
        return reference, version


def install(pipe, client: AssetClient, mode: str = "fov_kept",
            memory_slots: int = 20) -> MemorySource:
    """Attach the bridge to a constructed pipeline.

    Usage inside the HY entry (after pipeline creation, before pipe(...)):
        client = AssetClient(); client.sync_trajectory(viewmats, Ks)
        install(pipe, client, mode="fov_kept")
    """
    from gs_bridge.hy_client.rollout_fork import ar_rollout_inner_forked

    source = MemorySource(client, pipe, mode=mode, memory_slots=memory_slots)
    pipe.gs_bridge = source
    # instance-attribute override: `ar_rollout` calls self._ar_rollout_inner(...)
    pipe._ar_rollout_inner = ar_rollout_inner_forked.__get__(pipe)
    return source


def uninstall(pipe) -> None:
    """Restore the original rollout (for A/B comparison in one process)."""
    if getattr(pipe, "gs_bridge", None) is not None:
        pipe.gs_bridge.enabled = False
    if hasattr(pipe, "_ar_rollout_inner") and getattr(
        pipe._ar_rollout_inner, "__func__", None
    ) is not None:
        from gs_bridge.hy_client.rollout_fork import ar_rollout_inner_forked
        if pipe._ar_rollout_inner.__func__ is ar_rollout_inner_forked:
            del pipe._ar_rollout_inner  # falls back to the class method
