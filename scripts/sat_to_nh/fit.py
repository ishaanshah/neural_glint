import mitsuba as mi
mi.set_variant("cuda_rgb", "cuda_ad_rgb")

import torch
import numpy as np
import wandb
from argparse import ArgumentParser
from tqdm import tqdm
from torch.optim import Adam
from utils.sat import TorchSAT, open_sat
from utils.models import PredHist

device = torch.device("cuda")

parser = ArgumentParser()
parser.add_argument("sat_path")
parser.add_argument("out_path")
parser.add_argument("idx", type=int)
parser.add_argument("--cont", action="store_true")
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--log-batch-size", default=18, type=int)
parser.add_argument("--iters", default=20000, type=int)
parser.add_argument("--fp-sampling", default="exp", choices=["exp", "uniform"])
parser.add_argument("--nn-depth", default=2, type=int)
parser.add_argument("--nn-width", default=64, type=int)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--exp-name", default="")
parser.add_argument("--log-freq", type=int, default=10)
parser.add_argument("--start-stride", type=int, default=8)

args, encoder_config = parser.parse_known_args()

# Load SAT
if args.idx == -2:
    # Train the entire histogram
    sat, _, __ = open_sat(args.sat_path)
    sat = torch.from_numpy(sat.astype(np.int32)).to(device)
    idx = 0
elif args.idx == -1:
    # Train for all theta
    sat, _, __ = open_sat(args.sat_path)
    sat = torch.from_numpy(sat.sum(-1).astype(np.int32)).to(device)
    idx = "theta"
elif args.idx >= 0:
    # Train for phi along given theta
    sat, _, __ = open_sat(args.sat_path, "r")
    sat = torch.from_numpy(sat[:,:,args.idx].astype(np.int32)).to(device)
    idx = args.idx
else:
    raise ValueError("Invalid value for 'idx'")

if args.wandb:
    if not args.exp_name:
        raise ValueError("'exp_name' is required when logging to WandB")
    wandb.init(project="glint_rendering", group=args.exp_name, job_type=f"layer_{idx}", name=f"{args.exp_name}_{idx}")

skip = False
if torch.all(sat == 0):
    skip = True
sat = sat.reshape(sat.shape[0], sat.shape[1], -1)

if args.fp_sampling == "uniform":
    fp_sampler = lambda n: torch.rand((n, ), device=device) + args.start_stride / sat_res
else:
    fp_distr = torch.distributions.Exponential(torch.tensor(16.0, device=device))
    fp_sampler = lambda n: torch.clip(fp_distr.sample((n, )) + args.start_stride / sat_res, 0, 1)
sat_res = sat.shape[0]

# Create PyTorch module
sat = TorchSAT(sat)

if args.cont:
    model_data = torch.load(args.out_path)
    model_type = model_data["model_type"]
    assert model_type == "pred_hist", f"Incorrect model type, required 'pred_hist', got '{model_type}'"
    model_config = model_data["config"]
else:
    encoder_config = {
        "res": sat_res,
        "start_stride": args.start_stride,
    }
    model_config = {
        "depth": args.nn_depth,
        "width": args.nn_width,
        "output_dim": sat.sat.shape[-1],
        "encoder_config": encoder_config
    }

model = PredHist(**model_config)

if args.cont:
    model.load_state_dict(model_data["weights"])

model.to(device)

# If all entries of this layer's SAT are 0, we don't need to train
if skip:
    torch.save({
        "model_type": "pred_hist",
        "config": model_config,
        "weights": model.state_dict()
    }, args.out_path)
    exit(0)

batch_size = 2**args.log_batch_size
bar = tqdm(range(args.iters))
optimizer = Adam(model.parameters(), args.lr)
min_loss = 1e10
for i in bar:
    x = torch.rand([batch_size, 2], device=device)

    # Sample footprint size
    if args.fp_sampling == "exp" and torch.rand((1, )) < 0.05:
        fp_size = torch.rand((batch_size, 1), device=device)
        fp_size += args.start_stride / sat_res
        fp_size = torch.clamp(fp_size, 0, 1)
    else:
        fp_size = fp_sampler(batch_size).unsqueeze(-1)

    # Sample anisotropic factor
    x = torch.cat([x, fp_size, fp_size], dim=-1)

    gt = sat(x) + 1e-6
    gt_sum = gt.sum(-1, keepdim=True)
    gt = gt / gt_sum

    y = model(x)
    loss = torch.sum(y * torch.log(y / gt), -1)
    loss = torch.mean(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.wandb and i % args.log_freq == 0:
        wandb.log({
            f"loss_{idx}": loss
        })

    if i % 100 == 0:
        if loss < min_loss:
            min_loss = loss.item()
            torch.save({
                "model_type": "pred_hist",
                "config": model_config,
                "weights": model.state_dict()
            }, args.out_path)

    bar.set_description(f"Loss: {loss.item():.6f}")

if loss < min_loss:
    torch.save({
        "model_type": "pred_hist",
        "config": model_config,
        "weights": model.state_dict()
    }, args.out_path)