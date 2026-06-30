import torch
from torch import nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from argparse import ArgumentParser
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import standardize_dataset_name
from models import get_model
from utils import get_config, get_dataloader, setup, cleanup
from evaluate import evaluate


parser = ArgumentParser(description="Test a trained model on a dataset.")
# Parameters for model
parser.add_argument("--weight_path", type=str, required=True, help="The name of the weight to use.")
parser.add_argument("--output_filename", type=str, default=None, help="The name of the result file.")

# Parameters for evaluation
parser.add_argument("--dataset", type=str, required=True, help="The dataset to evaluate on.")
parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="The split to evaluate on.")
parser.add_argument("--input_size", type=int, default=224, help="The size of the input image.")
parser.add_argument("--sliding_window", action="store_true", help="Use sliding window strategy for evaluation.")
parser.add_argument("--max_input_size", type=int, default=4096, help="The maximum size of the input image in evaluation. Images larger than this will be processed using sliding window by force to avoid OOM.")
parser.add_argument("--max_num_windows", type=int, default=8, help="The maximum number of windows to be simultaneously processed.")
parser.add_argument("--resize_to_multiple", action="store_true", help="Resize the image to the nearest multiple of the input size.")
parser.add_argument("--stride", type=int, default=None, help="The stride for sliding window strategy.")
parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision for evaluation.")
parser.add_argument("--device", type=str, default="cuda", help="The device to use for evaluation.")
parser.add_argument("--num_workers", type=int, default=8, help="The number of workers for the data loader.")
parser.add_argument("--local_rank", type=int, default=-1, help="The local rank for distributed training.")


def run(local_rank: int, nprocs: int, args: ArgumentParser):
    print(f"Rank {local_rank} process among {nprocs} processes.")
    setup(local_rank, nprocs)
    print(f"Initialized successfully. Training with {nprocs} GPUs.")
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda:0"
    print(f"Using device: {device}.")

    ddp = nprocs > 1
    _ = get_config(vars(args).copy(), mute=False)

    model = get_model(model_info_path=args.weight_path).to(device)
    model = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(model), device_ids=[local_rank], output_device=local_rank) if ddp else model
    model = model.to(device)
    model.eval()

    args.output_filename = f"{model.model_name}_{args.weight_path.split('/')[-1].split('.')[0]}" if args.output_filename is None else args.output_filename

    dataloader = get_dataloader(args, split=args.split)
    scores = evaluate(
        model=model,
        data_loader=dataloader,
        sliding_window=args.sliding_window,
        max_input_size=args.max_input_size,
        window_size=args.input_size,
        stride=args.stride,
        max_num_windows=args.max_num_windows,
        amp=args.amp,
        local_rank=local_rank,
        nprocs=nprocs,
    )

    if local_rank == 0:
        for k, v in scores.items():
            print(f"{k}: {v}")

        result_dir = os.path.join(current_dir, "results", args.dataset, args.split)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"{args.output_filename}.txt"), "w") as f:
            for k, v in scores.items():
                f.write(f"{k}: {v}\n")
    
    cleanup(ddp)


if __name__ == "__main__":
    args = parser.parse_args()
    args.dataset = standardize_dataset_name(args.dataset)

    if args.dataset in ["sha", "shb", "qnrf", "nwpu", "game"]:
        assert args.split == "val", f"Split {args.split} is not available for dataset {args.dataset}."

    # Sliding window prediction will be used if args.sliding_window is True, or when the image size is larger than args.max_input_size
    args.stride = args.stride or args.input_size
    assert os.path.exists(args.weight_path), f"Weight path {args.weight_path} does not exist."
    args.in_memory_dataset = False

    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")
    if args.nprocs > 1:
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, args))
    else:
        run(0, 1, args)
