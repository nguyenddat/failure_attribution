from __future__ import annotations

WINDOW_SIZES = [5, 7, 9, 11]


def method_name(window_size: int) -> str:
    return f"fixed_window_all_at_once_w{window_size}"


def make_segments(trajectory: list[dict], window_size: int) -> list[list[dict]]:
    """Overlapping stride-(k/2) segments (50% overlap between consecutive segments).

    Unlike experiment 4's stride-1 center-sliding window, this segments the
    trajectory into chunks of size window_size, advancing by window_size // 2
    each step so consecutive segments share half their steps — a step near a
    segment boundary still gets full-window coverage in the next segment.
    The final segment is clipped (shorter) if it runs past the trajectory end
    — no fabricated padding steps. Segments beyond the trajectory end are
    dropped (they'd be a strict subset of the previous segment).
    """
    stride = max(1, window_size // 2)
    segments = []
    for start in range(0, len(trajectory), stride):
        segment = trajectory[start : start + window_size]
        if not segment:
            break
        segments.append(segment)
        if start + window_size >= len(trajectory):
            break
    return segments
