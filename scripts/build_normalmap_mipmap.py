"""
This script build the acceleration structure needed for fast evaluation
of NDF. It build a minmax structure in the uv-st space of the normal map.
"""
import mitsuba as mi
mi.set_variant("cuda_ad_rgb", "cuda_rgb")

import numpy as np
import torch
import os
from utils.fs import get_name
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("normalmap")
args = parser.parse_args()

nm = mi.TensorXf(mi.Bitmap(args.normalmap)).torch().cuda()
nm = nm * 2 - 1
nm /= torch.linalg.norm(nm, dim=-1, keepdim=True)

height, width = nm.shape[:2]

assert height == width, "Texture should be a square"
assert (height & (height - 1) == 0), "Texture size should be a power of 2"

num_miplevels = int(np.log2(height)) + 1
mips = [torch.concatenate([nm[...,:2], nm[...,:2]], -1)]
for i in range(1, num_miplevels):
    mip_size = 2**(num_miplevels - i - 1)
    mip = torch.zeros((mip_size, mip_size, 4)).cuda()

    mip[...,:2] = torch.minimum(
        torch.minimum(mips[-1][::2,::2,:2], mips[-1][1::2,1::2,:2]),
        torch.minimum(mips[-1][1::2,::2,:2], mips[-1][::2,1::2,:2])
    )

    mip[...,2:] = torch.maximum(
        torch.maximum(mips[-1][::2,::2,2:], mips[-1][1::2,1::2,2:]),
        torch.maximum(mips[-1][1::2,::2,2:], mips[-1][::2,1::2,2:])
    )

    mips.append(mip.cpu())

mips = [mip.cpu().numpy() for mip in mips]

np.save(os.path.join("data", "sat", get_name(args.normalmap)[1], "mip_hierarchy.npy"), np.array(mips, dtype=object), allow_pickle=True)