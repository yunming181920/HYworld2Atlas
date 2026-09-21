"""Frame <-> latent <-> chunk time mapping for HY-WorldPlay AR video.

Authoritative mapping (locked by unit tests, see tests/test_m0.py):

    Video has L latents and 4L - 3 pixel frames.
    Frame 0 is special: latent 0 covers ONLY frame 0.
    Frame f >= 1 belongs to latent (f + 3) // 4   (i.e. latents 1..L-1 each cover 4 frames).

    TRAJECTORY SEMANTICS (verified against generate.py): pose string 'w-31' produces
    31 motion steps = 32 latent poses. Adjacent latent poses are ONE motion step apart
    (w=0.08 / 3deg); that step is spread over the latent's 4 pixel frames:
    frame f's camera progress = f/4 motion steps (frame 124 == step 31 == latent 31).
    => Pixel-frame poses = SE3 interpolation between adjacent latent poses at
       t = (f - 4*(l-1)) / 4 (see camera_conv.interpolate_frame_poses).

    AR chunk c (chunk_latent_frames = 4) covers latents [4c, 4c+4).
    Chunk 0 = latents 0..3 = frames [0, 13)  (13 frames: frame 0 + 3 latents x 4 frames).
    Chunk c >= 1 = latents [4c, 4c+4) = frames [16c-3, 16c+13) = 16 frames.

Verified against hyvideo/generate.py `parse_pose_string_to_actions`:
    "first command gets 1 extra frame (the special frame 0)"
    and pose_to_input assert: `latent_num * 4 - 3` frames total.
"""

from typing import List, Tuple


def num_frames_from_latents(num_latents: int) -> int:
    """Total pixel frames: 4L - 3 (first latent covers only frame 0)."""
    assert num_latents >= 1
    return 4 * num_latents - 3


def num_latents_from_frames(num_frames: int) -> int:
    """Inverse of num_frames_from_latents."""
    assert num_frames >= 1
    assert (num_frames + 3) % 4 == 0, f"invalid frame count {num_frames}"
    return (num_frames + 3) // 4


def frame_to_latent(frame_idx: int) -> int:
    """Frame -> latent. Frame 0 -> latent 0; frame f>=1 -> (f+3)//4."""
    assert frame_idx >= 0
    if frame_idx == 0:
        return 0
    return (frame_idx + 3) // 4


def latent_to_frames(latent_idx: int) -> List[int]:
    """Latent -> the pixel frames it covers. Latent 0 -> [0]; latent l>=1 -> [4l-3 .. 4l]."""
    assert latent_idx >= 0
    if latent_idx == 0:
        return [0]
    return list(range(4 * latent_idx - 3, 4 * latent_idx + 1))


def chunk_latent_range(chunk_idx: int, chunk_latent_frames: int = 4) -> Tuple[int, int]:
    """Chunk -> [latent_start, latent_end) (end exclusive)."""
    assert chunk_idx >= 0
    start = chunk_idx * chunk_latent_frames
    return start, start + chunk_latent_frames


def chunk_frame_range(chunk_idx: int, chunk_latent_frames: int = 4) -> Tuple[int, int]:
    """Chunk -> [frame_start, frame_end) (end exclusive) in pixel frames.

    chunk 0: latents 0..3 -> frames 0..16 (17 frames: frame 0 + 4 frames x 4 latents... actually
        latent 0 = frame 0, latents 1-3 = frames 1..12, total 13? No -- see below.)
    Derivation: frames covered by latents [s, e) = union of latent_to_frames.
        s = 4c: latent 4c covers frames [16c-3, 16c] (c>=1).
        End: latent e-1 = 4c+3 covers frames up to 16c+12.
    So chunk c >= 1 covers [16c-3, 16c+13) = 16 frames.
    Chunk 0: latents 0..3 -> frame 0 + frames 1..12 -> [0, 13) = 13 frames.
    """
    start_latent, end_latent = chunk_latent_range(chunk_idx, chunk_latent_frames)
    if start_latent == 0:
        frame_start = 0
    else:
        frame_start = 4 * start_latent - 3
    frame_end = 4 * (end_latent - 1) + 1  # last frame of latent (end_latent - 1), inclusive end
    return frame_start, frame_end


def num_chunks(num_latents: int, chunk_latent_frames: int = 4) -> int:
    """Number of AR chunks for a video with num_latents latents."""
    assert num_latents >= 1
    assert num_latents % chunk_latent_frames == 0, (
        f"num_latents {num_latents} not a multiple of chunk_latent_frames {chunk_latent_frames}"
    )
    return num_latents // chunk_latent_frames


def frame_to_chunk(frame_idx: int, chunk_latent_frames: int = 4) -> int:
    """Frame -> the chunk whose latent range covers this frame's latent."""
    return frame_to_latent(frame_idx) // chunk_latent_frames


def window_version(chunk_idx: int) -> int:
    """Asset version after chunk `chunk_idx` was encoded: monotonically = chunk_idx + 1.

    The duplex constraint: while the DiT denoises chunk t, the latest GUARANTEED-complete
    asset version is (t-1) + 1 - 1 = t-1 encoded chunks... i.e. version_cap = t - 2 + 1 = t - 1.
    We keep the arithmetic explicit here; callers pass `version_cap = chunk_idx_being_denoised - 1`
    (see DESIGN.md section 3, "t-2 asset" in idea.txt terms).
    """
    return chunk_idx + 1
