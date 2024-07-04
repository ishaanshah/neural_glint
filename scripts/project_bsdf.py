"""
This script projects canonically oriented Gaussian lobes to Zonal Harmonics
for different roughness values. These are then rotated on the fly to obtain
the Spherical Harmonic coefficients oriented along a given direction.
"""
import mitsuba as mi
mi.set_variant("cuda_rgb", "cuda_ad_rgb")

import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from utils.sh.gaussian import Gaussian

import ipdb

parser = ArgumentParser()
parser.add_argument("--min-alpha", default=0.01, type=int)
parser.add_argument("--max-alpha", default=0.99, type=int)
parser.add_argument("--steps", default=1000, type=int)
parser.add_argument("--norders", default=100, type=int)
args = parser.parse_args()

master_coeffs = np.zeros((args.norders, args.steps))
idx = np.arange(args.norders)
idx = idx*idx + idx
i = 0
for alpha in tqdm(np.linspace(args.min_alpha, args.max_alpha, args.steps)):
    bsdf = Gaussian(args.norders, alpha=alpha, rotate_on_fly=True)
    coeff_path = bsdf.cache_path()
    coeffs = np.load(coeff_path)
    master_coeffs[:,i] = coeffs[idx][:,0]
    i += 1

np.save(os.path.join("data", "sh", "gaussian", "coeffs_1000.npy"), master_coeffs)