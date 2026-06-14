"""Queue waiting-time estimation from ZIP density maps.

Core idea: queue ≈ people in a high-density region + historical throughput.
No tracking, no bounding boxes — purely density-mass distribution in space.

Strategies for irregular queues (no fixed ROI):
  - threshold:   percentile threshold → largest connected component
  - morphology:  binary dilation + opening → largest connected component
  - ema:         exponential moving average of density maps (production-ready)

Waiting time via Little's Law: W = L / λ, where L is queue_count and λ is
the estimated service rate derived from the count derivative over a sliding
window of recent frames.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_opening, label
from torch import Tensor, nn


class SumPool2d(nn.Module):
    """Spatial sum-pooling for aggregating density over local windows.

    Equivalent to AvgPool2d with ``divisor_override=1`` so the output is
    the *sum* inside each kernel neighbourhood rather than the mean.
    """

    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, divisor_override=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


# ---------------------------------------------------------------------------
# Low-level queue-region extraction
# ---------------------------------------------------------------------------


def _ensure_numpy(density_map: Tensor) -> np.ndarray:
    """Squeeze batched/channel dims and move to cpu numpy (H, W)."""
    d = density_map.detach()
    while d.dim() > 2:
        d = d.squeeze(0)
    return d.cpu().numpy()


def extract_queue_region(
    density_map: Tensor,
    percentile: float = 75.0,
    smooth_kernel: int = 15,
) -> tuple[float, np.ndarray]:
    """Extract the queue count via percentile-threshold + largest CC.

    The queue is assumed to be the largest contiguous blob of high-density
    pixels.  Works best when the queue forms a single dominant cluster.

    Args:
        density_map: ``(H, W)``, ``(1, H, W)``, or ``(1, 1, H, W)`` tensor.
        percentile: threshold percentile computed on *non-zero* density values.
        smooth_kernel: averaging kernel size for pre-smoothing (must be odd).

    Returns:
        ``(queue_count, binary_mask)`` where *queue_count* is the sum of raw
        density inside the largest component and *binary_mask* is a ``(H, W)``
        bool array.
    """
    d = density_map.detach()
    # Smooth
    if d.dim() == 2:
        d = d.unsqueeze(0).unsqueeze(0)
    elif d.dim() == 3:
        d = d.unsqueeze(0)
    d_smooth = (
        F.avg_pool2d(d, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)
        .squeeze()
        .cpu()
        .numpy()
    )

    nonzero = d_smooth[d_smooth > 0]
    if len(nonzero) == 0:
        return 0.0, np.zeros_like(d_smooth, dtype=bool)

    thresh = np.percentile(nonzero, percentile)
    binary = (d_smooth >= thresh).astype(np.uint8)

    labeled, num_features = label(binary)
    if num_features == 0:
        return 0.0, binary.astype(bool)

    # Largest connected component
    largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    mask = labeled == largest_label

    # Sum on the *raw* (unsmoothed) density for accurate count
    raw = d.squeeze().cpu().numpy()
    queue_count = float(raw[mask].sum())
    return queue_count, mask


def extract_irregular_queue(
    density_map: Tensor,
    min_density: float = 0.01,
    dilate_iters: int = 10,
    open_iters: int = 3,
) -> float:
    """Extract queue count via morphological clustering.

    Binary-thresholds the density map, then dilates to connect nearby
    fragments (useful for serpentine / winding queues) and opens to remove
    spurious noise.  Returns the total density inside the largest resulting
    component.

    Args:
        density_map: ``(H, W)``, ``(1, H, W)``, or ``(1, 1, H, W)`` tensor.
        min_density: absolute density threshold for binarisation.
        dilate_iters: number of binary-dilation iterations.
        open_iters: number of binary-opening iterations.

    Returns:
        Queue count (float).
    """
    raw = _ensure_numpy(density_map)
    binary = raw > min_density

    connected = binary_dilation(binary, iterations=dilate_iters)
    opened = binary_opening(connected, iterations=open_iters)

    labeled, num_features = label(opened)
    if num_features == 0:
        return 0.0

    largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    mask = labeled == largest_label

    return float(raw[mask].sum())


# ---------------------------------------------------------------------------
# Adaptive (EMA) ROI — production-ready
# ---------------------------------------------------------------------------


class AdaptiveQueueROI:
    """Learn the queue region online via exponential moving average.

    Over the first few dozen seconds the system *learns* where the queue
    tends to be without any manual annotation, then uses that accumulated
    map as a stable pseudo-ROI frame-to-frame.  This avoids the jittery
    masks produced by per-frame thresholding.

    Args:
        alpha: EMA smoothing factor (0 < alpha ≤ 1).  Smaller → more stable.
        quantile: quantile threshold on the running mean for the binary mask.
    """

    def __init__(self, alpha: float = 0.05, quantile: float = 0.70) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.quantile = quantile
        self.running_mean: Tensor | None = None

    def update(self, density_map: Tensor) -> float:
        """Update the running mean and return the current queue count.

        Args:
            density_map: ``(H, W)`` or ``(1, H, W)`` or ``(1, 1, H, W)`` tensor.
        """
        d = density_map.detach()
        while d.dim() > 2:
            d = d.squeeze(0)
        # Work in float32 on the same device as input
        d = d.to(dtype=torch.float32)

        if self.running_mean is None:
            self.running_mean = d.clone()
        else:
            self.running_mean = (
                self.alpha * d + (1.0 - self.alpha) * self.running_mean
            )

        # Stable mask from the accumulated mean
        flat = self.running_mean.flatten()
        nonzero = flat[flat > 0]
        if nonzero.numel() == 0:
            return 0.0

        thresh = torch.quantile(nonzero, self.quantile)
        mask = self.running_mean >= thresh

        # Count on the *current* density inside the stable ROI
        return float(d[mask].sum().item())

    def reset(self) -> None:
        """Forget the accumulated mean (e.g. on scene change)."""
        self.running_mean = None


# ---------------------------------------------------------------------------
# Waiting-time estimator
# ---------------------------------------------------------------------------


class QueueWaitEstimator:
    """Estimate queue waiting time from a stream of ZIP density maps.

    Combines automatic queue-region extraction with dynamic service-rate
    estimation via Little's Law::

        W = L / λ

    where *L* is the current queue count and *λ* is the service rate
    (people/second), estimated from the count derivative over a sliding
    history window.

    Args:
        strategy: queue-extraction strategy (see above).
        fps: camera / inference frames per second (used to convert frame
            deltas to wall-clock time for rate estimation).
        window_seconds: size of the sliding history window in seconds.
        service_rate: fallback service rate (people/sec) when history is
            insufficient.
        percentile: threshold percentile for ``"threshold"`` strategy.
        min_density: absolute threshold for ``"morphology"`` strategy.
        ema_alpha: EMA smoothing factor for ``"ema"`` strategy.
        ema_quantile: quantile threshold for ``"ema"`` strategy mask.

    Typical usage::

        estimator = QueueWaitEstimator(strategy="ema", fps=10)
        for frame in video:
            den_map = zip_model(frame)      # (1, H, W) or (1, 1, H, W)
            wait_s = estimator.update(den_map)
            if wait_s is not None:
                print(f"Estimated wait: {wait_s:.1f}s")
    """

    def __init__(
        self,
        strategy: Literal["threshold", "morphology", "ema"] = "ema",
        fps: float = 10.0,
        window_seconds: float = 30.0,
        service_rate: float = 0.5,
        percentile: float = 75.0,
        min_density: float = 0.01,
        ema_alpha: float = 0.05,
        ema_quantile: float = 0.70,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")

        self.strategy = strategy
        self.fps = fps
        self.service_rate = service_rate
        self.percentile = percentile
        self.min_density = min_density

        maxlen = max(int(fps * window_seconds), 2)
        self._history: deque[tuple[float, float]] = deque(maxlen=maxlen)

        self._roi: AdaptiveQueueROI | None = None
        if strategy == "ema":
            self._roi = AdaptiveQueueROI(alpha=ema_alpha, quantile=ema_quantile)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, density_map: Tensor) -> float | None:
        """Ingest a new density map and return the estimated waiting time.

        Returns ``None`` when the history is too short to estimate a rate
        (first few frames).  Otherwise returns waiting time in seconds.
        """
        queue_count = self._extract_count(density_map)
        now = time.time()
        self._history.append((now, queue_count))
        return self._estimate_wait(queue_count)

    def reset(self) -> None:
        """Clear accumulated history and (for "ema") the learned ROI."""
        self._history.clear()
        if self._roi is not None:
            self._roi.reset()

    @property
    def history(self) -> list[tuple[float, float]]:
        """Return a snapshot of the (timestamp, count) history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_count(self, density_map: Tensor) -> float:
        if self.strategy == "threshold":
            count, _mask = extract_queue_region(
                density_map, percentile=self.percentile
            )
            return count

        if self.strategy == "morphology":
            return extract_irregular_queue(
                density_map, min_density=self.min_density
            )

        # "ema"
        assert self._roi is not None
        return self._roi.update(density_map)

    def _estimate_wait(self, current_count: float) -> float | None:
        if len(self._history) < 2:
            return None

        dt = self._history[-1][0] - self._history[0][0]
        if dt <= 0:
            return None

        # People who left the queue = oldest count - newest count
        delta = self._history[0][1] - self._history[-1][1]
        dynamic_rate = max(delta / dt, 0.05)  # floor at 0.05 ppl/s

        if dynamic_rate <= 0:
            return None

        return current_count / dynamic_rate


# ---------------------------------------------------------------------------
# Convenience: single-call waiting time from a density map
# ---------------------------------------------------------------------------


def estimate_waiting_time(
    density_map: Tensor,
    strategy: Literal["threshold", "morphology", "ema"] = "ema",
    fps: float = 10.0,
    service_rate: float = 0.5,
) -> QueueWaitEstimator:
    """Create and seed a ``QueueWaitEstimator`` with one frame.

    Returns the estimator so the caller can continue calling ``.update()``.
    The first call always returns ``None`` (insufficient history).
    """
    est = QueueWaitEstimator(strategy=strategy, fps=fps, service_rate=service_rate)
    est.update(density_map)
    return est
