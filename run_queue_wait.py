#!/usr/bin/env python3
"""End-to-end queue waiting-time estimation on a video file — STREAMING.

Processes video frame-by-frame without loading everything into RAM.
Suitable for long / high-resolution videos on memory-constrained machines.

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

Memory: frames are decoded, processed, and discarded one at a time.  Only
scalar metrics (count, wait) are retained for the final dashboard.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Generator

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
# Streaming frame sources
# ---------------------------------------------------------------------------


def stream_video_frames(video_path: str | Path) -> Generator[np.ndarray, None, tuple[int, int, float]]:
    """Yield frames one at a time.  Returns ``(total_frames, width, height, fps)``.

    Usage::

        gen = stream_video_frames("video.mp4")
        total, w, h, fps = yield from gen  # or gen.send(None) / next(gen)
        for frame in gen:
            …
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    info = (total, w, h, fps)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

    return info  # only reached via GeneratorExit, but signals the info


def count_video_frames(video_path: str | Path) -> tuple[int, int, int, float]:
    """Return (total_frames, width, height, fps) without decoding all frames."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()
    return total, w, h, fps


# ---------------------------------------------------------------------------
# Per-frame density extractors (stateful generators)
# ---------------------------------------------------------------------------


def make_density_from_zip(
    model,
    device: str = "cpu",
    target_size: int = 224,
):
    """Return a callable ``fn(frame) -> density_map`` using the ZIP model."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    model = model.to(device)
    model.eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3, 1, 1)
    to_tensor = transforms.ToTensor()

    def extract(frame: np.ndarray) -> np.ndarray:
        img = to_tensor(frame)  # (3, H, W) float [0,1]
        img = F.interpolate(
            img.unsqueeze(0), size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        img = (img.to(device) - mean.unsqueeze(0)) / std.unsqueeze(0)
        with torch.no_grad():
            den_map = model(img)
        dm = den_map.squeeze().cpu().numpy()
        if dm.ndim > 2:
            dm = dm.squeeze(0)
        return dm.astype(np.float64)

    return extract


def make_density_from_motion(resize: int = 128, sigma: float = 5.0):
    """Return a stateful callable ``fn(frame) -> density_map`` using frame diffs."""
    from scipy.ndimage import gaussian_filter

    prev: np.ndarray | None = None

    def extract(frame: np.ndarray) -> np.ndarray:
        nonlocal prev
        if frame.ndim == 3:
            gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = frame.astype(np.float64)

        # Fast resize
        h, w = gray.shape
        gray_small = gray[:: max(1, h // resize), :: max(1, w // resize)]
        if gray_small.shape[0] > resize:
            gray_small = gray_small[:resize, :resize]
        elif gray_small.shape[0] < resize:
            pad_h = resize - gray_small.shape[0]
            pad_w = resize - gray_small.shape[1]
            gray_small = np.pad(gray_small, ((0, pad_h), (0, pad_w)))

        if prev is not None:
            diff = np.abs(gray_small.astype(np.float64) - prev.astype(np.float64))
            dmap = gaussian_filter(diff, sigma=sigma)
            dmap = dmap / (dmap.max() + 1e-8)
        else:
            dmap = np.zeros((resize, resize), dtype=np.float64)

        prev = gray_small
        return dmap

    return extract


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = ArgumentParser(description="Queue waiting-time estimation on video (streaming)")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="queue_annotated.mp4",
                        help="Output video path (default: queue_annotated.mp4)")
    parser.add_argument("--model-info", type=str,
                        help="Path to ZIP model checkpoint (.pt)")
    parser.add_argument("--model-name", type=str, default="mamba3_micro",
                        help="Model architecture (default: mamba3_micro; also: mamba3_tiny, mamba3_pico)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Block size for the model (default: 16)")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Input image size for the model (default: 224)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, or auto (default: auto — uses cuda if available)")
    parser.add_argument("--strategy", type=str, default="ema",
                        choices=["threshold", "morphology", "ema"],
                        help="Queue extraction strategy (default: ema)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Processing FPS for rate estimation (default: 10)")
    parser.add_argument("--output-fps", type=float, default=0,
                        help="Output video FPS (0=same as input; lower=subsample — faster)")
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
    from tqdm import tqdm

    from models.queue_viz import plot_dashboard, render_queue_overlay_fast
    from models.queue_wait import QueueWaitEstimator


    # Resolve device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda but CUDA not available — falling back to cpu")
        args.device = "cpu"
    print(f"Device: {args.device}")

    out_dir = Path(args.output).parent or Path(".")
    out_dir.mkdir(exist_ok=True)
    base_name = Path(args.output).stem

    # ------------------------------------------------------------------
    # 1. Set up the density extractor and frame source
    # ------------------------------------------------------------------
    if args.demo:
        print("=== DEMO MODE: synthetic queue scenario ===")
        import matplotlib.pyplot as plt
        from demo_queue_wait import SyntheticQueue

        sim = SyntheticQueue(height=128, width=128, seed=42)
        total_frames = 300
        video_fps = 10.0
        frame_h, frame_w = 128, 128

        def frame_stream():
            for i in range(total_frames):
                if i < 100:
                    sim.arrival_rate = 0.4
                elif i < 200:
                    sim.arrival_rate = 0.8
                else:
                    sim.arrival_rate = 0.1
                yield sim.step()  # density map IS the "frame" in demo mode
        extract_density = None  # density comes directly from the stream
        frames_gen = frame_stream()
        use_rgb_frames = False
        print(f"  {total_frames} frames, {frame_w}x{frame_h}")

    elif args.video:
        total_frames, frame_w, frame_h, video_fps = count_video_frames(args.video)
        print(f"Video: {total_frames} frames, {frame_w}x{frame_h}, {video_fps:.1f} fps")

        def frame_stream():
            import cv2
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {args.video}")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()


        frames_gen = frame_stream()
        use_rgb_frames = True

        if args.model_info:
            print(f"Loading ZIP model from {args.model_info}...")

            # Check if the checkpoint has a 'config' key (model_info format)
            # or is a raw training checkpoint (only model_state_dict)
            ckpt = torch.load(args.model_info, map_location="cpu", weights_only=False)
            if "config" in ckpt:
                # Standard model_info format — get_model handles everything
                from models import get_model
                model = get_model(
                    model_name=args.model_name,
                    model_info_path=args.model_info,
                    block_size=args.block_size,
                    bins=[(0,0),(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,15),(15,25),(25,50)],
                    bin_centers=[0,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,12.5,20,37.5],
                    zero_inflated=True,
                    input_size=args.input_size,
                )
            else:
                # Training checkpoint — construct model from CLI args, load state_dict
                from models import get_model
                model = get_model(
                    model_name=args.model_name,
                    model_info_path="/tmp/__nonexistent_zip.pt",
                    block_size=args.block_size,
                    bins=[(0,0),(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,15),(15,25),(25,50)],
                    bin_centers=[0,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,12.5,20,37.5],
                    zero_inflated=True,
                    input_size=args.input_size,
                )
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            extract_density = make_density_from_zip(
                model, device=args.device, target_size=args.input_size,
            )
            print("  Model ready")
        else:
            print("  No model — using motion-based density proxy")
            extract_density = make_density_from_motion(resize=min(frame_w, frame_h, 256))

    else:
        print("ERROR: pass --video <path> or --demo")
        sys.exit(1)

    # Compute output subsampling
    out_fps = args.output_fps if args.output_fps > 0 else (video_fps if not args.demo else args.fps)
    render_every = max(1, int(round((video_fps if not args.demo else args.fps) / out_fps)))
    if render_every > 1:
        print(f"  Output subsampling: 1 every {render_every} frames ({out_fps:.1f} fps output)")

    # ------------------------------------------------------------------
    # 2. Streaming estimation + video write
    # ------------------------------------------------------------------
    print(f"Running QueueWaitEstimator (strategy={args.strategy})...")
    if args.live_preview:
        print("  Live preview ON — press 'q' in the window to stop early")
        cv2.namedWindow("Queue Wait Estimator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Queue Wait Estimator", 800, 500)

    est = QueueWaitEstimator(
        strategy=args.strategy,
        fps=video_fps if not args.demo else args.fps,
        window_seconds=30,
    )
    wait_times: list[float | None] = []
    queue_counts: list[float] = []
    last_density: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_frame: np.ndarray | None = None

    # Video writer (initialized lazily on first frame)
    writer: "cv2.VideoWriter | None" = None  # noqa: F821
    pbar = tqdm(total=total_frames, unit="fr", desc="Estimating", dynamic_ncols=True)
    try:
        for i, item in enumerate(frames_gen):
            if use_rgb_frames:
                rgb_frame = item
                den = extract_density(rgb_frame) if extract_density else item
            else:
                rgb_frame = (plt.cm.inferno(item / max(item.max(), 1e-6))[:, :, :3] * 255).astype(np.uint8)
                den = item

            # Update estimator
            dmap_t = torch.from_numpy(den.astype(np.float32))
            w = est.update(dmap_t)
            wait_times.append(w)
            queue_counts.append(est.history[-1][1] if est.history else 0.0)

            # Video writer init or write (subsampled if render_every > 1)
            if not args.no_video and (i % render_every == 0 or i == total_frames - 1):
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    sample = render_queue_overlay_fast(
                        den, mask=_get_ema_mask(est), wait_time=w,
                        queue_count=queue_counts[-1], frame_idx=i,
                        background=rgb_frame,
                    )
                    vh, vw = sample.shape[:2]
                    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (vw, vh))
                    writer.write(sample)
                else:
                    mask = _get_ema_mask(est)
                    frame_bgr = render_queue_overlay_fast(
                        den, mask=mask, wait_time=w,
                        queue_count=queue_counts[-1], frame_idx=i,
                        background=rgb_frame,
                    )
                    writer.write(frame_bgr)

            last_density = den
            last_frame = rgb_frame
            last_mask = _get_ema_mask(est)

            # Progress
            if (i + 1) % args.progress_every == 0 or i == total_frames - 1:
                w_str = f"{w:.1f}s" if w is not None else "…"
                pbar.set_postfix({"count": f"{queue_counts[-1]:.1f}", "wait": w_str})
                pbar.update(args.progress_every if (i + 1) % args.progress_every == 0
                            else (total_frames - i))

            # Live preview
            if args.live_preview and (i % max(1, int((video_fps if not args.demo else args.fps) // 4)) == 0):
                preview = render_queue_overlay_fast(
                    den, mask=last_mask, wait_time=w,
                    queue_count=queue_counts[-1], frame_idx=i,
                    background=rgb_frame,
                )
                cv2.imshow("Queue Wait Estimator", preview)  # already BGR
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  Stopped early by user")
                    break
    finally:
        pbar.close()
        if writer is not None:
            writer.release()
        if args.live_preview:
            cv2.destroyWindow("Queue Wait Estimator")

    final_wait = wait_times[-1] if wait_times else None
    if final_wait is not None:
        print(f"  Final: count={queue_counts[-1]:.1f}, wait={final_wait:.1f}s")
    else:
        print("  Final: insufficient history for wait estimate")
    print("Rendering dashboard...")
    dash = plot_dashboard(
        history=est.history,
        current_density=last_density,
        current_mask=last_mask,
        wait_times=wait_times,
        title=f"Queue Wait-Time — {base_name}",
    )
    dash_path = str(out_dir / f"{base_name}_dashboard.png")
    Image.fromarray(dash).save(dash_path)
    print(f"  → {dash_path}")

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = np.arange(len(queue_counts)) / (video_fps if not args.demo else args.fps)
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

    print("\nDone.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ema_mask(est: "QueueWaitEstimator") -> np.ndarray | None:  # noqa: F821
    if est._roi is not None and est._roi.running_mean is not None:
        rm = est._roi.running_mean.numpy()
        nonzero = rm[rm > 0]
        if len(nonzero) > 0:
            thresh = np.quantile(nonzero, 0.70)
            return rm >= thresh
        return np.zeros_like(rm, dtype=bool)
    return None





if __name__ == "__main__":
    main()
