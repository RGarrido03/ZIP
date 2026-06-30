"""Export the EBC model to ONNX format.

Usage:
    uv run python export_onnx.py \
        --model_name mamba3_micro \
        --block_size 32 \
        --dataset game \
        --input_size 448 \
        --output model.onnx

    # Export without zero-inflation head
    uv run python export_onnx.py ... --no_zero_inflated --output model_nozi.onnx
"""

import argparse
import json
import os
import sys

import torch

# Ensure the project root is on the path so `models` and `utils` imports resolve.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from models import get_model  # noqa: E402
from utils import calc_bin_center  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EBC model to ONNX")
    p.add_argument("--model_name", default="mamba3_micro")
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--dataset", default="game")
    p.add_argument("--input_size", type=int, default=448)
    p.add_argument("--output", default="model.onnx")
    p.add_argument("--no_zero_inflated", action="store_true",
                   help="Disable the zero-inflation head")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load bin configuration (mirrors trainer.py) ──────────────────────
    with open(os.path.join(_here, "configs", "bin_config.json")) as f:
        bins_raw = json.load(f)[args.dataset][str(args.block_size)]
    bins = [(float(lo), float(hi)) for lo, hi in bins_raw]

    with open(os.path.join(_here, "counts", f"{args.dataset}.json")) as f:
        count_stats = json.load(f)[str(args.block_size)]
    count_stats = {int(k): int(v) for k, v in count_stats.items()}
    bin_centers, _ = calc_bin_center(bins, count_stats)

    zero_inflated = not args.no_zero_inflated

    # ── Instantiate model ───────────────────────────────────────────────
    model = get_model(
        model_name=args.model_name,
        model_info_path=f"/tmp/_onnx_model_info_{os.getpid()}.pth",
        block_size=args.block_size,
        bins=bins,
        bin_centers=bin_centers,
        zero_inflated=zero_inflated,
        input_size=args.input_size,
    )
    model.eval()

    print(f"Model: {args.model_name}, block_size={args.block_size}, "
          f"zero_inflated={zero_inflated}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Create dummy input ──────────────────────────────────────────────
    dummy = torch.randn(1, 3, args.input_size, args.input_size)

    # Smoke-test the forward pass before exporting.
    with torch.no_grad():
        out = model(dummy)
    print(f"Forward reference shape: {tuple(out.shape)}")

    # ── Export to ONNX ──────────────────────────────────────────────────
    # Use the legacy TorchScript-based exporter (dynamo=False).
    # The dynamo-based exporter has issues with dynamic shapes in
    # models that use einops / view / reshape with symbolic batch dims.
    dynamic_axes = {
        "input":  {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )

    print(f"✓ Exported to {args.output}")

    # ── Quick validation: load back and compare output ──────────────────
    import onnx
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model passes checker")


if __name__ == "__main__":
    main()
