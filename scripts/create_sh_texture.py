"""
Pre-computes the normalized Legendre polynomials which are required
for obtaining the Spherical Harmonics value in a given direcition.
"""
import mitsuba as mi
mi.set_variant("cuda_rgb", "cuda_ad_rgb")

import numpy as np
import drjit as dr
import os
from utils.fs import create_dir
from argparse import ArgumentParser
from utils import sh
from tqdm import trange

parser = ArgumentParser()
parser.add_argument("--norders", default=100, type=int)
parser.add_argument("--ntheta", default=2048, type=int)
parser.add_argument("--out_dir", "-o", default=os.path.join("data", "sh"))

args = parser.parse_args()

num_coeffs = sh.get_num_coeff(args.norders)
lcoeffs = np.zeros((args.ntheta, num_coeffs), dtype=np.float32)

# Compute Legendre polynomials distributed proportional by cos(theta)
# These will be used for projecting canonically oriented distributions
# Example: Beckmann lobe or Gaussian lobe oriented along (0, 0, 1)
theta = dr.linspace(mi.Float, 0, dr.pi, args.ntheta)
wz = dr.cos(theta)
for i in trange(num_coeffs):
    l, m = sh.get_lm(i)
    lcoeffs[:,i] = sh.eval_legendre(wz, l[0], abs(m[0])) * sh.k(l[0], m[0])

create_dir(args.out_dir)
np.save(os.path.join(args.out_dir, "lcoeffs.npy"), lcoeffs)