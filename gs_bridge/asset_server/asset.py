"""Global 4D Gaussian asset: ring buffer of encoded MoVieS windows.

v1 design (DESIGN.md section 0): the asset IS the window ring buffer -- no cross-window
gaussian fusion/dedup (phase 2). Each window is immutable after insertion; queries
select window(s) by version and render via gsplat. FIFO eviction at capacity.

Version semantics (time_map.window_version): after chunk c is encoded, asset version
= c + 1. The duplex pipeline queries with version_cap = t - 1 while denoising chunk t
(idea.txt's "t-2 asset": t-2 chunks are guaranteed complete, the (t-1)th may still be
encoding -- cap excludes it).
"""

import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from gs_bridge.asset_server.encoder import WindowGaussians


@dataclass
class WindowSlot:
    """One ring-buffer slot: an immutable encoded window + query metadata."""
    version: int                    # = chunk_idx + 1
    chunk_idx: int
    window: WindowGaussians
    # latent coverage of this chunk (chunk_latent_range), for query -> window lookup
    latent_start: int
    latent_end: int


class GlobalAsset:
    """Thread-safe ring buffer of MoVieS windows with versioned visibility.

    Thread model: updates arrive from the IPC/server loop; queries can run
    concurrently (windows are immutable; only buffer mutation is locked).
    """

    def __init__(self, capacity_windows: int = 20, chunk_latent_frames: int = 4):
        assert capacity_windows >= 1
        self.capacity = capacity_windows
        self.chunk_latent_frames = chunk_latent_frames
        self._slots: deque[WindowSlot] = deque(maxlen=capacity_windows)
        self._lock = threading.Lock()
        self.version = 0  # latest complete version (= n_encoded_chunks)
        self.stats = {"inserted": 0, "evicted": 0, "queries": 0,
                      "rejected_stale": 0, "rolled_back": 0}

    # ------------------------------------------------------------------
    # update path
    # ------------------------------------------------------------------

    def insert_window(self, window: WindowGaussians) -> int:
        """Insert an encoded chunk window. Returns the new asset version.

        Stale inserts (window version <= self.version after a rollback) are
        rejected: an in-flight encode of a discarded chunk must not re-enter
        the asset (viewpoint-reset race, see rollout_fork block B / server).
        """
        chunk_idx = window.chunk_idx
        latent_start = chunk_idx * self.chunk_latent_frames
        latent_end = latent_start + self.chunk_latent_frames
        slot = WindowSlot(
            version=chunk_idx + 1,
            chunk_idx=chunk_idx,
            window=window,
            latent_start=latent_start,
            latent_end=latent_end,
        )
        with self._lock:
            if slot.version <= self.version:
                self.stats["rejected_stale"] += 1
                return self.version  # discard: chunk predates the current trajectory
            if len(self._slots) == self._slots.maxlen:
                self.stats["evicted"] += 1
            self._slots.append(slot)
            self.stats["inserted"] += 1
            self.version = max(self.version, slot.version)
        return self.version

    # ------------------------------------------------------------------
    # query path
    # ------------------------------------------------------------------

    def visible_slots(self, version_cap: Optional[int] = None) -> List[WindowSlot]:
        """Slots visible under a version cap (None = latest). Snapshot under lock."""
        with self._lock:
            slots = list(self._slots)
        if version_cap is None:
            return slots
        return [s for s in slots if s.version <= version_cap]

    def find_window_for_latent(
        self, latent_idx: int, version_cap: Optional[int] = None
    ) -> Optional[WindowSlot]:
        """The window whose chunk covers `latent_idx` (for per-pose memory queries).

        Falls back to the temporally-nearest visible window if the covering chunk
        was evicted or not yet encoded (early chunks of the rollout).
        """
        slots = self.visible_slots(version_cap)
        if not slots:
            return None
        for s in slots:
            if s.latent_start <= latent_idx < s.latent_end:
                return s
        # fallback: nearest by latent distance
        return min(
            slots,
            key=lambda s: min(
                abs(latent_idx - s.latent_start), abs(latent_idx - s.latent_end)
            ),
        )

    def snapshot_latest(self) -> Optional[WindowSlot]:
        """Latest complete window (for Atlas frozen-time observation)."""
        with self._lock:
            return self._slots[-1] if self._slots else None

    # ------------------------------------------------------------------
    # viewpoint-reset path
    # ------------------------------------------------------------------

    def rollback_to(self, version: int) -> int:
        """Discard all windows with version > `version`; return the new version.

        Viewpoint-reset semantics (DESIGN.md section 4): switching the generation
        viewpoint discards the in-flight chunks (t-1, t) entirely -- their content
        is conditioned on the OLD trajectory's context, so re-entering the asset
        would pollute the new rollout. The asset's trusted boundary rolls back to
        the last version generated along the old trajectory (t-2).

        Idempotent: rolling back to >= current version is a no-op.
        """
        with self._lock:
            if version >= self.version:
                return self.version
            while self._slots and self._slots[-1].version > version:
                self._slots.pop()
                self.stats["rolled_back"] += 1
            self.version = self._slots[-1].version if self._slots else version
            if not self._slots:
                # nothing survives: the requested version becomes the floor
                # (queries under this cap return None until new windows arrive)
                self.version = version
            return self.version

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._slots)

    def describe(self) -> dict:
        with self._lock:
            slots = list(self._slots)
        return {
            "version": self.version,
            "n_windows": len(slots),
            "capacity": self.capacity,
            "chunk_range": (
                (slots[0].chunk_idx, slots[-1].chunk_idx) if slots else None
            ),
            **self.stats,
        }
