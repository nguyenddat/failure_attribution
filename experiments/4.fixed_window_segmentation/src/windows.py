from __future__ import annotations

from enum import Enum


class ContextMode(str, Enum):
    BOTH = "both"
    PREV_ONLY = "prev_only"
    NEXT_ONLY = "next_only"


CONTEXT_MODES = [ContextMode.BOTH, ContextMode.PREV_ONLY, ContextMode.NEXT_ONLY]
WINDOW_SIZES = [5, 7, 9, 11]


def method_name(window_size: int, mode: ContextMode) -> str:
    return f"fixed_window_w{window_size}_{mode.value}"


def before_after_slices(
    trajectory: list[dict], current_step: int, window_size: int, mode: ContextMode
) -> tuple[list[dict], list[dict]]:
    """Kernel-style fixed window centered on current_step.

    window_size k counts the current step as one slot of the window
    (odd k => symmetric split for BOTH). Context budget (excluding the
    current step) is k - 1 steps for every mode, so the three modes stay
    cost-comparable. Boundaries are clipped, not padded (valid-convolution
    style) — no fabricated steps near the start/end of a trajectory.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    budget = window_size - 1

    if mode is ContextMode.BOTH:
        half = budget // 2
        before = trajectory[max(0, current_step - half) : current_step]
        after = trajectory[current_step + 1 : current_step + 1 + half]
        return before, after

    if mode is ContextMode.PREV_ONLY:
        before = trajectory[max(0, current_step - budget) : current_step]
        return before, []

    if mode is ContextMode.NEXT_ONLY:
        after = trajectory[current_step + 1 : current_step + 1 + budget]
        return [], after

    raise ValueError(f"Unsupported context mode: {mode}")
