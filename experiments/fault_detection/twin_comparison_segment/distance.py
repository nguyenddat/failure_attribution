"""Block 1 of twin-comparison segmentation: adjacent-pair distance + Dt threshold.

1 trajectory = {n agent behaviors} <-> 1 video = {n frames}. For every
adjacent pair of agent behaviors, compute an embedding-based distance D.
Ds = {D of every adjacent pair in the trajectory} is the input to threshold
detection:

- If the histogram of Ds shows a peak clearly separated from the main mass
  (a small, isolated cluster of large distances), that separation point is
  Dt (transition-break threshold).
- Otherwise Ds is bimodal (low-D "same context" cluster vs high-D
  "transition" cluster) rather than a single Gaussian, so Dt is the Otsu
  valley that best splits the two clusters (maximizes between-cluster
  variance over the histogram).
"""

from __future__ import annotations

import numpy as np

from experiments.embedding_models import get_model


def adjacent_distances(steps: list[str], model_name: str = "embedding_3_small") -> list[float]:
    """Cosine distance between embeddings of every adjacent agent-behavior pair."""
    embeddings = get_model(model_name).embed_documents(steps)
    vectors = [np.array(vec) for vec in embeddings]

    distances = []
    for a, b in zip(vectors, vectors[1:]):
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            distances.append(1.0)
        else:
            distances.append(1.0 - float(np.dot(a, b) / (norm_a * norm_b)))
    return distances


def otsu_threshold(distances: list[float], bins: int = 30) -> float:
    """Bin boundary that maximizes between-cluster variance (Otsu 1979).

    Splits Ds into a low-D cluster and a high-D cluster without assuming a
    single-mode (Gaussian) shape - the natural fit for the observed bimodal
    "same context" vs "transition" split.
    """
    arr = np.array(distances)
    counts, edges = np.histogram(arr, bins=bins)
    counts = counts.astype(float)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    total_weight = counts.sum()
    total_sum = float(np.sum(counts * bin_centers))

    weight_low = 0.0
    sum_low = 0.0
    best_variance = -1.0
    best_edge = float(edges[len(edges) // 2])

    for i in range(len(counts) - 1):
        weight_low += counts[i]
        sum_low += counts[i] * bin_centers[i]
        weight_high = total_weight - weight_low
        if weight_low == 0 or weight_high == 0:
            continue

        mean_low = sum_low / weight_low
        mean_high = (total_sum - sum_low) / weight_high
        between_variance = weight_low * weight_high * (mean_low - mean_high) ** 2

        if between_variance > best_variance:
            best_variance = between_variance
            best_edge = float(edges[i + 1])

    return best_edge


def detect_dt(
    distances: list[float],
    bins: int = 30,
    isolated_peak_max_fraction: float = 0.2,
) -> tuple[float, str]:
    """Return (Dt, method) where method is "isolated_peak" or "otsu"."""
    arr = np.array(distances)
    counts, edges = np.histogram(arr, bins=bins)
    total = counts.sum()

    # Scan bins right-to-left for a gap (empty bin) that separates a small,
    # isolated upper cluster (a "sharp peak") from the rest of the mass.
    for i in range(len(counts) - 1, 0, -1):
        if counts[i] > 0 and counts[i - 1] == 0:
            upper_count = counts[i:].sum()
            if 0 < upper_count <= total * isolated_peak_max_fraction:
                return float(edges[i]), "isolated_peak"

    return otsu_threshold(distances, bins=bins), "otsu"
