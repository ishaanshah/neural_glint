import os
import typer
from argparse import ArgumentParser
from utils.exp import GPUScheduler
from utils.fs import create_dir
from utils.sat import open_sat

app = typer.Typer(pretty_exceptions_show_locals=False)

parser = ArgumentParser()
parser.add_argument("sat_path")
parser.add_argument("out_path")
parser.add_argument("exp_name")
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--log-batch-size", default=18, type=int)
parser.add_argument("--iters", default=20000, type=int)
parser.add_argument("--fp-sampling", default="exp", choices=["exp", "uniform"])
parser.add_argument("--nn-depth", default=2, type=int)
parser.add_argument("--nn-width", default=64, type=int)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--start-stride", type=int, default=8)

args, encoder_config = parser.parse_known_args()

sat, _, __ = open_sat(args.sat_path, mmap_mode="r")
ntheta = sat.shape[2]
idx = list(range(ntheta))
stdouts = []
jobs = []
create_dir(args.out_path, False)
for i in idx:
    model_path = os.path.join(args.out_path, f"model_{i:02d}.pt")
    job = {
        "args": [
            "python",
            os.path.join("scripts", "sat_to_nh", "fit.py"),
            args.sat_path,
            model_path,
            str(i),
            "--lr", f"{args.lr}",
            "--log-batch-size", str(args.log_batch_size),
            "--iters", str(args.iters),
            "--fp-sampling", args.fp_sampling,
            "--nn-depth", str(args.nn_depth),
            "--nn-width", str(args.nn_width),
            "--exp-name", args.exp_name,
            "--start-stride", str(args.start_stride)
        ] + encoder_config
    }
    if args.wandb:
        job["args"].append("--wandb")

    jobs.append(job)
    stdouts.append(os.path.join(args.out_path, f"stdout_{i:02d}.log"))

model_path = os.path.join(args.out_path, f"model_theta.pt")
job = {
    "args": [
        "python",
        os.path.join("scripts", "sat_to_nh", "fit.py"),
        args.sat_path,
        model_path,
        "-1",
        "--lr", f"{args.lr}",
        "--log-batch-size", str(args.log_batch_size),
        "--iters", str(args.iters),
        "--fp-sampling", args.fp_sampling,
        "--nn-depth", str(args.nn_depth),
        "--nn-width", str(args.nn_width),
        "--start-stride", str(args.start_stride)
    ] + encoder_config
}
if args.wandb:
    job["args"].append("--wandb")

jobs.append(job)
stdouts.append(os.path.join(args.out_path, f"stdout_theta.log"))

sched = GPUScheduler(jobs, stdouts)
sched.run()