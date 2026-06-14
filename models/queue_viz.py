"""Visualization utilities for queue waiting-time estimation.

Produces annotated frames and dashboards from ZIP density maps and
QueueWaitEstimator outputs.  Works with synthetic or real density maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from scipy.ndimage import find_objects

matplotlib.use("Agg")  # headless-safe


# ---------------------------------------------------------------------------
# Colour maps & constants
# ---------------------------------------------------------------------------

_DENSITY_CMAP = plt.cm.inferno
_MASK_ALPHA = 0.35
_MASK_COLOR = (0.0, 1.0, 0.5)  # neon green
_WAIT_CMAP = plt.cm.RdYlGn_r  # red=high wait, green=low
_HISTORY_COLOR = "#1f77b4"
_RATE_COLOR = "#ff7f0e"
_THRESH_COLOR = (0.172, 0.627, 0.172)   # #2ca02c
_MORPH_COLOR = (0.839, 0.153, 0.157)    # #d62728
_EMA_COLOR = (0.584, 0.404, 0.741)      # #9467bd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _ensure_numpy(density_map: np.ndarray | "torch.Tensor") -> np.ndarray:  # noqa: F821
    """Accept torch tensor or numpy array, return (H, W) float64."""
    try:
        import torch

        if isinstance(density_map, torch.Tensor):
            arr = density_map.detach().cpu().numpy()
        else:
            arr = np.asarray(density_map)
    except ImportError:
        arr = np.asarray(density_map)
    while arr.ndim > 2:
        arr = arr.squeeze(0)
    return arr.astype(np.float64)


def _mask_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return a polygon approximating the mask boundary, or None if empty."""
    from matplotlib.contour import QuadContourSet

    if not mask.any():
        return None
    # Use a simple contour at 0.5 level on float mask
    try:
        cs: QuadContourSet = plt.contour(
            mask.astype(np.float32), levels=[0.5], colors="none"
        )
        paths = []
        for col in cs.collections:
            for path in col.get_paths():
                paths.append(path.vertices)
        plt.close(cs.figure)
        if paths:
            return np.concatenate(paths, axis=0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------


def render_queue_overlay(
    density_map: np.ndarray | "torch.Tensor",  # noqa: F821
    mask: np.ndarray | None = None,
    wait_time: float | None = None,
    queue_count: float | None = None,
    frame_idx: int = 0,
    background: np.ndarray | None = None,
    *,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 80,
) -> np.ndarray:
    """Return an RGB frame (H, W, 3) uint8 with density heatmap, mask overlay,
    and metrics text.

    Args:
        density_map: ``(H, W)`` density values.
        mask: ``(H, W)`` bool array — detected queue region.
        wait_time: current wait-time estimate in seconds (for text overlay).
        queue_count: current queue count (for text overlay).
        frame_idx: frame number (for text overlay).
        background: ``(H, W, 3)`` uint8 image to blend under the heatmap.
        figsize: matplotlib figure size in inches.
        dpi: output resolution.

    Returns:
        uint8 RGB numpy array.
    """
    den = _ensure_numpy(density_map)
    H, W = den.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Background
    if background is not None:
        bg = np.asarray(background)
        if bg.ndim == 3 and bg.shape[2] == 3:
            ax.imshow(bg, extent=(0, W, H, 0))

    # Density heatmap
    vmax = max(den.max(), 1e-6)
    ax.imshow(den, cmap=_DENSITY_CMAP, alpha=0.7, extent=(0, W, H, 0),
              norm=Normalize(vmin=0, vmax=vmax))

    # Mask overlay
    if mask is not None and mask.any():
        mask_float = mask.astype(np.float32)
        # Semi-transparent green where mask is active
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[mask, 1] = 1.0  # green channel
        overlay[mask, 3] = _MASK_ALPHA
        ax.imshow(overlay, extent=(0, W, H, 0))

        # Contour
        contour = _mask_contour(mask)
        if contour is not None:
            ax.plot(contour[:, 0], contour[:, 1], color=_MASK_COLOR,
                    linewidth=1.5, alpha=0.9)

    # Text overlay
    lines: list[str] = []
    if frame_idx is not None:
        lines.append(f"Frame {frame_idx}")
    if queue_count is not None:
        lines.append(f"Queue: {queue_count:.1f} people")
    if wait_time is not None:
        lines.append(f"Wait: {wait_time:.1f}s")
    if lines:
        text = "\n".join(lines)
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6,
                      edgecolor="none"),
            color="white",
        )

    ax.set_axis_off()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return rgb


# ---------------------------------------------------------------------------
# Dashboard: multi-panel time-series view
# ---------------------------------------------------------------------------


def plot_dashboard(
    history: list[tuple[float, float]],
    current_density: np.ndarray | "torch.Tensor" | None = None,  # noqa: F821
    current_mask: np.ndarray | None = None,
    wait_times: list[float] | None = None,
    background: np.ndarray | None = None,
    *,
    figsize: tuple[float, float] = (16, 9),
    dpi: int = 100,
    title: str = "Queue Wait-Time Dashboard",
) -> np.ndarray:
    """Produce a 4-panel dashboard:

    ┌─────────────────────┬─────────────────────┐
    │   density + mask    │   queue count (t)   │
    ├─────────────────────┼─────────────────────┤
    │   wait time (t)     │   service rate (t)  │
    └─────────────────────┴─────────────────────┘

    Args:
        history: list of ``(timestamp, queue_count)`` from the estimator.
        current_density: latest density map.
        current_mask: latest queue mask.
        wait_times: pre-computed wait times (optional; derived from history
            if None).
        background: background image for the density panel.
        figsize, dpi: figure dimensions.
        title: overall title.

    Returns:
        uint8 RGB numpy array.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ((ax_density, ax_count), (ax_wait, ax_rate)) = axs

    # --- Panel 1: density + mask ---
    if current_density is not None:
        den = _ensure_numpy(current_density)
        H, W = den.shape
        if background is not None:
            bg = np.asarray(background)
            if bg.ndim == 3 and bg.shape[2] == 3:
                ax_density.imshow(bg, extent=(0, W, H, 0))
        vmax = max(den.max(), 1e-6)
        ax_density.imshow(den, cmap=_DENSITY_CMAP, alpha=0.7,
                          extent=(0, W, H, 0),
                          norm=Normalize(vmin=0, vmax=vmax))
        if current_mask is not None and current_mask.any():
            overlay = np.zeros((H, W, 4), dtype=np.float32)
            overlay[current_mask, 1] = 1.0
            overlay[current_mask, 3] = _MASK_ALPHA
            ax_density.imshow(overlay, extent=(0, W, H, 0))
        ax_density.set_title("Density + Queue Mask")
    else:
        ax_density.text(0.5, 0.5, "No density map", ha="center", va="center",
                        transform=ax_density.transAxes)
        ax_density.set_title("Density + Queue Mask")
    ax_density.set_axis_off()

    # --- Panel 2: queue count over time ---
    if history:
        times = [t - history[0][0] for t, _ in history]
        counts = [c for _, c in history]
        ax_count.plot(times, counts, color=_HISTORY_COLOR, linewidth=1.5)
        ax_count.fill_between(times, 0, counts, alpha=0.15, color=_HISTORY_COLOR)
        ax_count.scatter(times[-1], counts[-1], color=_HISTORY_COLOR, s=40, zorder=5)
        ax_count.set_ylabel("Queue Count (people)")
        ax_count.set_xlabel("Time (s)")
        ax_count.set_title("Queue Count Over Time")
        ax_count.grid(True, alpha=0.3)
    else:
        ax_count.set_title("Queue Count (no data)")

    # --- Panel 3: wait time over time ---
    if wait_times is not None and len(wait_times) > 0:
        valid = [w for w in wait_times if w is not None]
        if valid:
            t_wait = np.arange(len(wait_times)) / 10.0  # assume 10 fps
            ax_wait.plot(t_wait[:len(wait_times)], wait_times,
                         color=_RATE_COLOR, linewidth=1.5)
            ax_wait.fill_between(
                t_wait[:len(wait_times)], 0,
                [w if w is not None else 0 for w in wait_times],
                alpha=0.15, color=_RATE_COLOR,
            )
            ax_wait.axhline(y=0, color="gray", linewidth=0.5)
            ax_wait.set_ylabel("Wait Time (s)")
            ax_wait.set_xlabel("Time (s)")
            ax_wait.set_title("Estimated Wait Time")
            ax_wait.grid(True, alpha=0.3)
    else:
        ax_wait.set_title("Wait Time (no data)")

    # --- Panel 4: derived service rate ---
    if history and len(history) >= 2:
        times_arr = np.array([t - history[0][0] for t, _ in history])
        counts_arr = np.array([c for _, c in history])
        rates = []
        rate_times = []
        for i in range(1, len(history)):
            dt = times_arr[i] - times_arr[i - 1]
            if dt > 0:
                delta = counts_arr[i - 1] - counts_arr[i]
                rate = max(delta / dt, 0.05)
                rates.append(rate)
                rate_times.append(times_arr[i])
        if rates:
            ax_rate.plot(rate_times, rates, color="#2ca02c", linewidth=1.5)
            ax_rate.axhline(y=0.05, color="red", linewidth=0.8, linestyle="--",
                            alpha=0.5, label="floor (0.05)")
            ax_rate.set_ylabel("Rate (ppl/s)")
            ax_rate.set_xlabel("Time (s)")
            ax_rate.set_title("Estimated Service Rate")
            ax_rate.legend(fontsize=8)
            ax_rate.grid(True, alpha=0.3)
    else:
        ax_rate.set_title("Service Rate (no data)")

    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return rgb


# ---------------------------------------------------------------------------
# Strategy comparison: side-by-side masks
# ---------------------------------------------------------------------------


def plot_strategy_comparison(
    density_map: np.ndarray | "torch.Tensor",  # noqa: F821
    threshold_mask: np.ndarray | None = None,
    morphology_mask: np.ndarray | None = None,
    ema_mask: np.ndarray | None = None,
    *,
    figsize: tuple[float, float] = (18, 6),
    dpi: int = 100,
) -> np.ndarray:
    """Side-by-side comparison of the three extraction strategies on one frame.

    Args:
        density_map: the density map.
        threshold_mask: mask from ``extract_queue_region``.
        morphology_mask: mask from ``extract_irregular_queue``.
        ema_mask: mask from ``AdaptiveQueueROI`` (running-mean mask).

    Returns:
        uint8 RGB numpy array.
    """
    den = _ensure_numpy(density_map)
    H, W = den.shape
    vmax = max(den.max(), 1e-6)
    masks = [
        ("Threshold + CC", threshold_mask, _THRESH_COLOR),
        ("Morphology", morphology_mask, _MORPH_COLOR),
        ("EMA (production)", ema_mask, _EMA_COLOR),
    ]

    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    for ax, (title, mask, color) in zip(axs, masks):
        ax.imshow(den, cmap=_DENSITY_CMAP, extent=(0, W, H, 0),
                  norm=Normalize(vmin=0, vmax=vmax))
        if mask is not None and mask.any():
            overlay = np.zeros((H, W, 4), dtype=np.float32)
            overlay[mask, :3] = np.array(color)
            overlay[mask, 3] = _MASK_ALPHA
            ax.imshow(overlay, extent=(0, W, H, 0))
            contour = _mask_contour(mask)
            if contour is not None:
                ax.plot(contour[:, 0], contour[:, 1], color=color,
                        linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return rgb


# ---------------------------------------------------------------------------
# EMA convergence animation (gif)
# ---------------------------------------------------------------------------


def render_ema_convergence(
    density_stream: Sequence[np.ndarray | "torch.Tensor"],  # noqa: F821
    alpha: float = 0.05,
    quantile: float = 0.70,
    output_path: str | Path = "ema_convergence.gif",
    *,
    fps: int = 10,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 80,
    max_frames: int = 100,
) -> Path:
    """Animate how the EMA running mean stabilises over time.

    Args:
        density_stream: sequence of density maps (one per frame).
        alpha: EMA smoothing factor.
        quantile: quantile for the binary mask.
        output_path: where to save the GIF.
        fps: frames per second in the output.
        figsize, dpi: figure dimensions.
        max_frames: cap on frame count.

    Returns:
        Path to the saved GIF.
    """
    from models.queue_wait import AdaptiveQueueROI

    output_path = Path(output_path)
    roi = AdaptiveQueueROI(alpha=alpha, quantile=quantile)

    # Pre-compute all states
    frames_data: list[dict] = []
    for i, dmap in enumerate(density_stream):
        if i >= max_frames:
            break
        den = _ensure_numpy(dmap)
        # Convert to torch for ROI update if needed
        import torch
        dmap_t = torch.from_numpy(np.asarray(dmap)) if not isinstance(dmap, torch.Tensor) else dmap
        roi.update(dmap_t)
        running = roi.running_mean.cpu().numpy() if roi.running_mean is not None else den
        flat = running.flatten()
        nonzero = flat[flat > 0]
        thresh = float(np.quantile(nonzero, quantile)) if len(nonzero) > 0 else 0.0
        mask = running >= thresh
        frames_data.append({
            "density": den.copy(),
            "running_mean": running.copy(),
            "mask": mask.copy(),
            "frame": i,
        })

    if not frames_data:
        raise ValueError("Empty density stream")

    fig, (ax_den, ax_ema) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    H, W = frames_data[0]["density"].shape
    vmax = max(f["density"].max() for f in frames_data) or 1e-6

    den_im = ax_den.imshow(
        np.zeros((H, W)), cmap=_DENSITY_CMAP,
        extent=(0, W, H, 0), norm=Normalize(vmin=0, vmax=vmax),
        animated=True,
    )
    ema_im = ax_ema.imshow(
        np.zeros((H, W)), cmap=_DENSITY_CMAP,
        extent=(0, W, H, 0), norm=Normalize(vmin=0, vmax=vmax),
        animated=True,
    )
    ax_den.set_title("Current Density")
    ax_ema.set_title("EMA Running Mean")
    for ax in (ax_den, ax_ema):
        ax.set_axis_off()
    fig.tight_layout()

    def update(frame_data):
        den_im.set_array(frame_data["density"])
        ema_im.set_array(frame_data["running_mean"])
        return den_im, ema_im

    ani = FuncAnimation(fig, update, frames=frames_data, blit=True,
                        interval=1000 // fps)
    ani.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Full video renderer
# ---------------------------------------------------------------------------


def render_wait_video(
    density_stream: Sequence[np.ndarray | "torch.Tensor"],  # noqa: F821
    masks: Sequence[np.ndarray | None] | None = None,
    wait_times: Sequence[float | None] | None = None,
    queue_counts: Sequence[float] | None = None,
    output_path: str | Path = "queue_wait.mp4",
    background: np.ndarray | None = None,
    *,
    fps: int = 10,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 80,
) -> Path:
    """Render an MP4 video with density+mask overlay and metrics.

    Args:
        density_stream: sequence of density maps.
        masks: optional per-frame bool masks (same length).
        wait_times: optional per-frame wait times.
        queue_counts: optional per-frame queue counts.
        output_path: output ``.mp4`` file.
        background: single background image for all frames.
        fps: output video FPS.
        figsize, dpi: per-frame figure dimensions.

    Returns:
        Path to the saved video.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video output.  "
            "Install with: pip install opencv-python"
        )

    output_path = Path(output_path)
    n = len(density_stream)

    # Align lengths
    if masks is None:
        masks = [None] * n
    if wait_times is None:
        wait_times = [None] * n
    if queue_counts is None:
        queue_counts = [None] * n

    first_frame = render_queue_overlay(
        density_stream[0],
        mask=masks[0] if len(masks) > 0 else None,
        wait_time=wait_times[0] if len(wait_times) > 0 else None,
        queue_count=queue_counts[0] if len(queue_counts) > 0 else None,
        frame_idx=0,
        background=background,
        figsize=figsize,
        dpi=dpi,
    )
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    try:
        writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        for i in range(1, n):
            frame = render_queue_overlay(
                density_stream[i],
                mask=masks[i] if i < len(masks) else None,
                wait_time=wait_times[i] if i < len(wait_times) else None,
                queue_count=queue_counts[i] if i < len(queue_counts) else None,
                frame_idx=i,
                background=background,
                figsize=figsize,
                dpi=dpi,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return output_path
