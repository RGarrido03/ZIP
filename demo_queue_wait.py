#!/usr/bin/env python3
"""Demo: queue waiting-time estimation with full visualizations.

Generates a synthetic queue scenario (people arriving, queuing, being served),
runs the QueueWaitEstimator on the density stream, and produces:
  - queue_wait_dashboard.png   — 4-panel dashboard at final frame
  - queue_wait_strategies.png  — side-by-side strategy comparison
  - queue_wait_video.mp4       — annotated video of queue over time
  - ema_convergence.gif        — how the EMA ROI stabilises
  - queue_wait_timeseries.png  — count & wait time over time

Requires: opencv-python (for video output).  Falls back to image-only if absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# Ensure the project root is on the path
_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


from models.queue_viz import (
    plot_dashboard,
    plot_strategy_comparison,
    render_ema_convergence,
    render_queue_overlay,
    render_wait_video,
)
from models.queue_wait import (
    AdaptiveQueueROI,
    QueueWaitEstimator,
    extract_irregular_queue,
    extract_queue_region,
)


# ---------------------------------------------------------------------------
# Synthetic queue simulator
# ---------------------------------------------------------------------------


class SyntheticQueue:
    """Simulate a queue in a 2D arena using Gaussian blobs for people.

    The scene layout (left → right):
      Entrance zone  →  Queue zone  →  Service point
         x: 0-24         x: 25-99       x: 100-127

    People arrive at the entrance, drift through the queue zone, and
    disappear when served.  The output is a density map (H, W).
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        arrival_rate: float = 0.5,   # avg people arriving per frame
        service_rate: float = 0.3,   # avg people served per frame
        blob_sigma: float = 2.0,     # Gaussian sigma per person
        seed: int = 42,
    ):
        self.H, self.W = height, width
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.blob_sigma = blob_sigma
        self.rng = np.random.default_rng(seed)

        # Each person: (x, y) position in pixels
        self._people: list[tuple[float, float]] = []

        # Build a grid of coordinates for fast blob rendering
        ys, xs = np.mgrid[0:height, 0:width]
        self._ygrid = ys.astype(np.float32)
        self._xgrid = xs.astype(np.float32)

    def step(self) -> np.ndarray:
        """Advance one frame and return the density map (H, W)."""
        # 1. Serve people at the front (rightmost people)
        n_serve = self.rng.poisson(self.service_rate)
        n_serve = min(n_serve, len(self._people))
        if n_serve > 0:
            # Remove the rightmost people (closest to service point)
            self._people.sort(key=lambda p: p[0], reverse=True)
            self._people = self._people[n_serve:]

        # 2. Move remaining people rightward (toward service point)
        new_people: list[tuple[float, float]] = []
        for x, y in self._people:
            # Drift toward service point with some jitter
            dx = self.rng.normal(1.5, 0.3)
            dy = self.rng.normal(0.0, 0.5)
            new_x = x + dx
            new_y = y + dy
            # Keep within bounds (don't overshoot service point)
            if new_x < self.W - 5:
                new_y = np.clip(new_y, 5, self.H - 5)
                new_people.append((new_x, new_y))
            # else: person reached service point and is removed
        self._people = new_people

        # 3. New arrivals at entrance zone
        n_arrive = self.rng.poisson(self.arrival_rate)
        for _ in range(n_arrive):
            y = self.rng.uniform(10, self.H - 10)
            x = self.rng.uniform(2, 20)
            self._people.append((x, y))

        # 4. Render density map from Gaussian blobs
        dmap = np.zeros((self.H, self.W), dtype=np.float32)
        for x, y in self._people:
            # Compute Gaussian contribution around (x, y)
            dist2 = (self._xgrid - x) ** 2 + (self._ygrid - y) ** 2
            dmap += np.exp(-dist2 / (2 * self.blob_sigma**2))

        return dmap

    @property
    def queue_count(self) -> int:
        """Ground-truth number of people in the queue zone (x: 25-99)."""
        return sum(1 for x, _ in self._people if 25 <= x < 100)


# ---------------------------------------------------------------------------
# Scenario: varying arrival rate to test different queue behaviours
# ---------------------------------------------------------------------------


def generate_scenario(
    n_frames: int = 300,
    height: int = 128,
    width: int = 128,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate a density stream with three phases:
      0-99:   moderate arrivals (queue builds)
      100-199:  high arrivals (queue grows)
      200-299:  low arrivals (queue drains)
    """
    sim = SyntheticQueue(height=height, width=width, seed=seed)
    stream: list[np.ndarray] = []
    for i in range(n_frames):
        if i < 100:
            sim.arrival_rate = 0.4
            sim.service_rate = 0.3
        elif i < 200:
            sim.arrival_rate = 0.8
            sim.service_rate = 0.3
        else:
            sim.arrival_rate = 0.1
            sim.service_rate = 0.35
        stream.append(sim.step())
    return stream


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path("demo_output")
    out_dir.mkdir(exist_ok=True)

    print("Generating synthetic queue scenario (300 frames)...")
    density_stream = generate_scenario(n_frames=300)
    print(f"  {len(density_stream)} frames, shape={density_stream[0].shape}")

    # ------------------------------------------------------------------
    # Run QueueWaitEstimator on the stream (EMA strategy)
    # ------------------------------------------------------------------
    print("Running QueueWaitEstimator (ema)...")
    est = QueueWaitEstimator(strategy="ema", fps=10, window_seconds=30,
                             ema_alpha=0.05, ema_quantile=0.70)
    wait_times: list[float | None] = []
    masks: list[np.ndarray | None] = []
    queue_counts: list[float] = []

    for i, dmap in enumerate(density_stream):
        dmap_t = torch.from_numpy(dmap)
        w = est.update(dmap_t)
        wait_times.append(w)
        queue_counts.append(est.history[-1][1] if est.history else 0.0)

        # Get the EMA mask for visualization
        if est._roi is not None and est._roi.running_mean is not None:
            rm = est._roi.running_mean.numpy()
            flat = rm.flatten()
            nonzero = flat[flat > 0]
            if len(nonzero) > 0:
                thresh = np.quantile(nonzero, 0.70)
                masks.append(rm >= thresh)
            else:
                masks.append(np.zeros_like(rm, dtype=bool))
        else:
            masks.append(None)

    print(f"  Final wait estimate: {wait_times[-1]:.1f}s" if wait_times[-1] is not None
          else "  (no estimate yet)")

    # ------------------------------------------------------------------
    # 1. Single-frame overlay (frame 200 — peak queue)
    # ------------------------------------------------------------------
    print("Rendering single-frame overlay...")
    peak_idx = 200
    ov = render_queue_overlay(
        density_stream[peak_idx],
        mask=masks[peak_idx],
        wait_time=wait_times[peak_idx],
        queue_count=queue_counts[peak_idx],
        frame_idx=peak_idx,
    )
    from PIL import Image
    Image.fromarray(ov).save(out_dir / "queue_overlay_frame200.png")
    print(f"  → {out_dir / 'queue_overlay_frame200.png'}")

    # ------------------------------------------------------------------
    # 2. Dashboard (final frame)
    # ------------------------------------------------------------------
    print("Rendering dashboard...")
    dash = plot_dashboard(
        history=est.history,
        current_density=density_stream[-1],
        current_mask=masks[-1],
        wait_times=wait_times,
        title="Queue Wait-Time Dashboard — Synthetic Scenario",
    )
    Image.fromarray(dash).save(out_dir / "queue_wait_dashboard.png")
    print(f"  → {out_dir / 'queue_wait_dashboard.png'}")

    # ------------------------------------------------------------------
    # 3. Strategy comparison (peak frame, all 3 strategies)
    # ------------------------------------------------------------------
    print("Rendering strategy comparison...")
    dmap_peak_np = density_stream[peak_idx]
    dmap_peak_t = torch.from_numpy(dmap_peak_np)
    _, thresh_mask = extract_queue_region(dmap_peak_t, percentile=75.0, smooth_kernel=11)
    morph_count = extract_irregular_queue(dmap_peak_t, min_density=0.01)
    # Build morph mask from scipy (same logic as extract_irregular_queue)
    from scipy.ndimage import binary_dilation, binary_opening, label
    raw = dmap_peak_np.copy()
    binary = raw > 0.01
    connected = binary_dilation(binary, iterations=10)
    opened = binary_opening(connected, iterations=3)
    labeled, nf = label(opened)
    if nf > 0:
        largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        morph_mask = labeled == largest_label
    else:
        morph_mask = np.zeros_like(raw, dtype=bool)

    comp = plot_strategy_comparison(
        dmap_peak_np,
        threshold_mask=thresh_mask,
        morphology_mask=morph_mask,
        ema_mask=masks[peak_idx] if peak_idx < len(masks) else None,
    )
    Image.fromarray(comp).save(out_dir / "queue_wait_strategies.png")
    print(f"  → {out_dir / 'queue_wait_strategies.png'}")

    # ------------------------------------------------------------------
    # 4. EMA convergence GIF
    # ------------------------------------------------------------------
    print("Rendering EMA convergence animation...")
    ema_path = render_ema_convergence(
        density_stream[:60],  # first 60 frames (~6 seconds)
        alpha=0.05,
        quantile=0.70,
        output_path=str(out_dir / "ema_convergence.gif"),
        fps=10,
        max_frames=60,
    )
    print(f"  → {ema_path}")

    # ------------------------------------------------------------------
    # 5. Time-series plot
    # ------------------------------------------------------------------
    print("Rendering time-series plot...")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = np.arange(len(queue_counts)) / 10.0  # 10 fps

    ax1.plot(t, queue_counts, color="#1f77b4", linewidth=1.2)
    ax1.fill_between(t, 0, queue_counts, alpha=0.12, color="#1f77b4")
    ax1.set_ylabel("Queue Count (people)")
    ax1.set_title("Queue Count Over Time")
    ax1.grid(True, alpha=0.3)

    valid_w = [(tw if tw is not None else np.nan) for tw in wait_times]
    ax2.plot(t, valid_w, color="#ff7f0e", linewidth=1.2)
    ax2.fill_between(t, 0, [w if w is not None and not np.isnan(w) else 0 for w in valid_w],
                     alpha=0.12, color="#ff7f0e")
    ax2.axhline(y=0, color="gray", linewidth=0.5)
    # Mark phase transitions
    for x_pos in [10, 20]:
        ax2.axvline(x=x_pos, color="gray", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Wait Time (s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Estimated Wait Time Over Time")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "queue_wait_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'queue_wait_timeseries.png'}")

    # ------------------------------------------------------------------
    # 6. Video (optional — requires opencv-python)
    # ------------------------------------------------------------------
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("\n⚠ opencv-python not installed — skipping video.  Install with:")
        print("    pip install opencv-python")
        print(f"\nAll image outputs in: {out_dir}/")
        return

    print("Rendering annotated video...")
    video_path = render_wait_video(
        density_stream,
        masks=masks,
        wait_times=wait_times,
        queue_counts=queue_counts,
        output_path=str(out_dir / "queue_wait_video.mp4"),
        fps=10,
    )
    print(f"  → {video_path}")
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
