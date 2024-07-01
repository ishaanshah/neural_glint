"""
Builds a Summed Area Table which is required
for fast query of histogram during training
or for GxD [Gamboa et al. 2018].
"""
import mitsuba as mi
import os
import numpy as np
import drjit as dr
from utils.sat import calc_optimal_bins, compute_sat
from utils.fs import get_name
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("texture")
parser.add_argument("ntheta", type=int)
parser.add_argument("nphi", type=int)
parser.add_argument("--out_dir", type=str, default=os.path.join("data","sat"))

args = parser.parse_args()

texture = mi.TensorXf(mi.Bitmap(args.texture)).numpy()
texture *= 2
texture -= 1
texture /= np.linalg.norm(texture, axis=-1)[...,None]
h, w, _ = texture.shape
normals = mi.Vector3f(texture.reshape(-1, 3))

texture_name = get_name(args.texture)[1]

bin_edges, bin_centers = None, None
bin_centers, bin_edges = calc_optimal_bins(dr.acos(normals.z).numpy(), args.ntheta)

SAT = compute_sat(texture, args.ntheta, args.nphi, bin_edges)

out_path = os.path.join(args.out_dir, texture_name, f"{args.ntheta}_{args.nphi}_sat")
os.makedirs(out_path, exist_ok=True)
np.save(os.path.join(out_path, "sat.npy"), SAT)
np.save(os.path.join(out_path, "bin_centers.npy"), bin_centers)
np.save(os.path.join(out_path, "bin_edges.npy"), bin_edges)
