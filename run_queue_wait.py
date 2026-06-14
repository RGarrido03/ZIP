#!/usr/bin/env python3
"""End-to-end queue waiting-time estimation on a video file.

Usage:
    # Quick test with synthetic data (no video file needed):
    uv run python run_queue_wait.py --demo

    # Real video without model (motion-proxy for pipeline testing):
    uv run python run_queue_wait.py \\
        --video path/to/queue.mp4 --output annotated.mp4

    # Real video with live preview window:
    uv run python run_queue_wait.py \\
        --video path/to/queue.mp4 --live-preview

    # With ZIP model (recommended for production):
    uv run python run_queue_wait.py \\
        --video path/to/queue.mp4 \\
        --model-info checkpoints/.../ckpt.pth \\
        --model-name mamba3_tiny \\
        --device cuda \\
        --output annotated.mp4

Outputs:
    annotated.mp4      — video with density heatmap + queue mask + wait time
    *_dashboard.png    — 4-panel summary dashboard
    *_timeseries.png   — count & wait time over all frames

Progress: a tqdm bar shows frame progress, current queue count, and estimated
wait time.  Pass ``--live-preview`` to also see the annotated frame in an
OpenCV window during processing (press 'q' to stop early).
"""

from __future__ import annotations

import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

# Ensure project root on path
_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Frame reader
# ---------------------------------------------------------------------------


def read_video_frames(video_path: str | Path) -> list[np.ndarray]:
    """Read all frames from a video as uint8 RGB arrays (H, W, 3)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return frames


# ---------------------------------------------------------------------------
# Density map extraction (plug your ZIP model here)
# ---------------------------------------------------------------------------


def density_from_zip(
    frames: list[np.ndarray],
    model,
    device: str = "cpu",
    target_size: int = 224,
) -> list[np.ndarray]:
    """Run ZIP model on every frame, return list of (H', W') density maps."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    model = model.to(device)
    model.eval()

    # Match the normalization used during training (CLIP stats)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    density_maps: list[np.ndarray] = []
    to_tensor = transforms.ToTensor()

    for i, frame in enumerate(frames):
        # Convert to tensor, resize, normalize
        img = to_tensor(frame)  # (3, H, W) float [0,1]
        img = F.interpolate(
            img.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        img = (img - mean.unsqueeze(0)) / std.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            den_map = model(img)  # (1, 1, H', W') or similar

        # Squeeze to (H, W) numpy
        dm = den_map.squeeze().cpu().numpy()
        if dm.ndim > 2:
            dm = dm.squeeze(0)
        density_maps.append(dm.astype(np.float64))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames")

    return density_maps


def density_from_motion(
    frames: list[np.ndarray],
    resize: int = 128,
    sigma: float = 5.0,
) -> list[np.ndarray]:
    """Fallback: use frame-difference magnitude as a density proxy.

    This is NOT a person counter — it just lets you test the pipeline
    without a trained model.  Replace with ``density_from_zip`` for real use.
    """
    from scipy.ndimage import gaussian_filter

    prev_gray = None
    density_maps: list[np.ndarray] = []

    for frame in frames:
        # Convert to grayscale, resize
        if frame.ndim == 3:
            gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = frame.astype(np.float64)

        # Resize
        h, w = gray.shape
        scale = resize / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        # Simple resize via slicing + averaging
        gray_small = gray[:: max(1, h // resize), :: max(1, w // resize)]
        # Pad or crop to exact size
        if gray_small.shape[0] > resize:
            gray_small = gray_small[:resize, :resize]
        elif gray_small.shape[0] < resize:
            pad_h = resize - gray_small.shape[0]
            pad_w = resize - gray_small.shape[1]
            gray_small = np.pad(gray_small, ((0, pad_h), (0, pad_w)))

        if prev_gray is not None:
            diff = np.abs(gray_small.astype(np.float64) - prev_gray.astype(np.float64))
            # Smooth to create blob-like densities
            dmap = gaussian_filter(diff, sigma=sigma)
            # Normalize to reasonable range
            dmap = dmap / (dmap.max() + 1e-8)
        else:
            dmap = np.zeros((resize, resize), dtype=np.float64)

        prev_gray = gray_small
        density_maps.append(dmap)

    return density_maps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = ArgumentParser(description="Queue waiting-time estimation on video")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="queue_annotated.mp4",
                        help="Output video path (default: queue_annotated.mp4)")
    parser.add_argument("--model-info", type=str,
                        help="Path to ZIP model checkpoint (.pt)")
    parser.add_argument("--model-name", type=str, default="mamba3_tiny",
                        help="Model architecture name (default: mamba3_tiny)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Block size for the model (default: 16)")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Input image size for the model (default: 224)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--strategy", type=str, default="ema",
                        choices=["threshold", "morphology", "ema"],
                        help="Queue extraction strategy (default: ema)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Video FPS for rate estimation (default: 10)")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data (no video file needed)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video output (only produce dashboard + timeseries)")
    parser.add_argument("--live-preview", action="store_true",
                        help="Show live OpenCV window during processing (press 'q' to stop)")
    parser.add_argument("--progress-every", type=int, default=1,
                        help="Update progress bar every N frames (default: 1)")
    args = parser.parse_args()

    if not _has_cv2():
        print("ERROR: opencv-python required.  Install with:")
        print("  uv pip install opencv-python")
        sys.exit(1)

    import cv2
    import torch
    from PIL import Image

    from models.queue_viz import plot_dashboard, render_queue_overlay
    from models.queue_wait import QueueWaitEstimator

    out_dir = Path(args.output).parent or Path(".")
    out_dir.mkdir(exist_ok=True)
    base_name = Path(args.output).stem

    # ------------------------------------------------------------------
    # 1. Get frames
    # ------------------------------------------------------------------
    if args.demo:
        print("=== DEMO MODE: synthetic queue scenario ===")
        import matplotlib.pyplot as plt
        from demo_queue_wait import SyntheticQueue

        sim = SyntheticQueue(height=128, width=128, seed=42)
        frames = []
        density_stream = []
        for i in range(300):
            if i < 100:
                sim.arrival_rate = 0.4
            elif i < 200:
                sim.arrival_rate = 0.8
            else:
                sim.arrival_rate = 0.1
            dmap = sim.step()
            density_stream.append(dmap)
            # Create a fake RGB frame from the density map
            rgb = (plt.cm.inferno(dmap / max(dmap.max(), 1e-6))[:, :, :3] * 255).astype(np.uint8)
            frames.append(rgb)
        print(f"  {len(frames)} frames generated")
    elif args.video:
        print(f"Reading video: {args.video}")
        frames = read_video_frames(args.video)
        print(f"  {len(frames)} frames, shape={frames[0].shape}")

        # Extract density maps
        if args.model_info:
            print(f"Loading ZIP model from {args.model_info}...")
            from models import get_model
            from models.ebc.model import EBC

            # Try loading via get_model; fall back to manual assembly
            model = get_model(
                model_name=args.model_name,
                model_info_path=args.model_info,
                block_size=args.block_size,
                bins=[(0,0),(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,15),(15,25),(25,50)],
                bin_centers=[0,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,12.5,20,37.5],
                zero_inflated=True,
                input_size=args.input_size,
            )
            # Load weights if they weren't auto-loaded
            ckpt = torch.load(args.model_info, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                try:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                except Exception as e:
                    print(f"  Warning: partial weight load — {e}")

            print("  Extracting density maps with ZIP model...")
            density_stream = density_from_zip(
                frames, model, device=args.device, target_size=args.input_size,
            )
        else:
            print("  No model provided — using motion-based density proxy")
            print("  (For real results, pass --model-info <checkpoint.pt>)")
            density_stream = density_from_motion(frames, resize=128)
    else:
        print("ERROR: pass --video <path> or --demo")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Run QueueWaitEstimator (with progress)
    # ------------------------------------------------------------------
    from tqdm import tqdm

    print(f"Running QueueWaitEstimator (strategy={args.strategy})...")
    if args.live_preview:
        print("  Live preview ON — press 'q' in the window to stop early")
        cv2.namedWindow("Queue Wait Estimator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Queue Wait Estimator", 800, 500)

    est = QueueWaitEstimator(
        strategy=args.strategy,
        fps=args.fps,
        window_seconds=30,
    )
    wait_times: list[float | None] = []
    masks: list[np.ndarray | None] = []
    queue_counts: list[float] = []
    total = len(density_stream)

    pbar = tqdm(total=total, unit="fr", desc="Estimating", dynamic_ncols=True)
    try:
        for i, dmap_np in enumerate(density_stream):
            dmap_t = torch.from_numpy(dmap_np.astype(np.float32))
            w = est.update(dmap_t)
            wait_times.append(w)
            queue_counts.append(est.history[-1][1] if est.history else 0.0)

            # Get EMA mask (cheap — only for viz, could skip if not needed)
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

            # Update progress bar
            if (i + 1) % args.progress_every == 0 or i == total - 1:
                w_str = f"{w:.1f}s" if w is not None else "…"
                pbar.set_postfix({
                    "count": f"{queue_counts[-1]:.1f}",
                    "wait": w_str,
                })
                pbar.update(args.progress_every if (i + 1) % args.progress_every == 0
                            else (total % args.progress_every or args.progress_every))

            # Live preview window
            if args.live_preview and frames and (i % max(1, args.fps // 2) == 0 or i == total - 1):
                preview = render_queue_overlay(
                    dmap_np,
                    mask=masks[-1],
                    wait_time=w,
                    queue_count=queue_counts[-1],
                    frame_idx=i,
                    background=frames[min(i, len(frames) - 1)],
                )
                cv2.imshow("Queue Wait Estimator", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n  Stopped early by user")
                    break
    finally:
        pbar.close()
        if args.live_preview:
            cv2.destroyWindow("Queue Wait Estimator")

    final_wait = wait_times[-1]
    print(f"  Final: count={queue_counts[-1]:.1f}, wait={final_wait:.1f}s"
          if final_wait is not None else "  Final: (insufficient history)")

    # ------------------------------------------------------------------
    # 3. Render dashboard + timeseries
    # ------------------------------------------------------------------
    print("Rendering dashboard...")
    dash = plot_dashboard(
        history=est.history,
        current_density=density_stream[-1],
        current_mask=masks[-1],
        wait_times=wait_times,
        title=f"Queue Wait-Time — {base_name}",
    )
    dash_path = str(out_dir / f"{base_name}_dashboard.png")
    Image.fromarray(dash).save(dash_path)
    print(f"  → {dash_path}")

    # Time series
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = np.arange(len(queue_counts)) / args.fps
    ax1.plot(t, queue_counts, linewidth=1.2)
    ax1.fill_between(t, 0, queue_counts, alpha=0.12)
    ax1.set_ylabel("Queue Count")
    ax1.set_title("Queue Count Over Time")
    ax1.grid(True, alpha=0.3)
    valid_w = [(tw if tw is not None else np.nan) for tw in wait_times]
    ax2.plot(t, valid_w, color="#ff7f0e", linewidth=1.2)
    ax2.fill_between(t, 0, [w if w is not None and not np.isnan(w) else 0 for w in valid_w], alpha=0.12, color="#ff7f0e")
    ax2.set_ylabel("Wait Time (s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Estimated Wait Time")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    ts_path = str(out_dir / f"{base_name}_timeseries.png")
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    print(f"  → {ts_path}")

    # ------------------------------------------------------------------
    # 4. Render annotated video
    # ------------------------------------------------------------------
    if not args.no_video:
        import cv2
        print(f"Rendering annotated video ({len(density_stream)} frames)...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Determine output resolution from first overlay frame
        sample = render_queue_overlay(
            density_stream[0],
            mask=masks[0],
            wait_time=wait_times[0],
            queue_count=queue_counts[0],
            frame_idx=0,
            background=frames[0],
        )
        vh, vw = sample.shape[:2]
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (vw, vh))

        try:
            for i in tqdm(range(len(density_stream)), desc="Rendering", unit="fr",
                          dynamic_ncols=True):
                frame_rgb = render_queue_overlay(
                    density_stream[i],
                    mask=masks[i] if i < len(masks) else None,
                    wait_time=wait_times[i] if i < len(wait_times) else None,
                    queue_count=queue_counts[i] if i < len(queue_counts) else None,
                    frame_idx=i,
                    background=frames[i] if i < len(frames) else None,
                )
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        print(f"  → {args.output}")
    print("\nDone.")


if __name__ == "__main__":
    main()
