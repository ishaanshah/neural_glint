import mitsuba as mi
mi.set_variant("cuda_rgb")
mi.set_log_level(mi.LogLevel.Info)

import os
import plugins as _
import torch
import numpy as np
import time
from typing import Tuple
from utils.denoiser import Denoiser
from plugins.gxd_integrator import render_gxd
from plugins.sh_glint_integrator import render_fastdot
from plugins.ratio_estimator import ratio_estimator
from utils.fs import write_image
from utils.sat import TorchSAT, open_sat
from utils.models import TorchWrapper, PredHist
from utils.render import render_multi_pass
from argparse import ArgumentParser

device = "cuda:0"

###### Argument Parsing
parser = ArgumentParser()
parser.add_argument("out_path", type=str)
parser.add_argument("--alpha", default=0.01, type=float)
parser.add_argument("--resx", default=1920, type=int)
parser.add_argument("--resy", default=1080, type=int)
parser.add_argument("--lookup", action="store_true")
parser.add_argument("--write-mask", action="store_true")
subparsers = parser.add_subparsers(title="method", dest="method", required=True)

gt_parser = subparsers.add_parser("gt")
gt_parser.add_argument("--spp", default=4000, type=int)

sat_parser = subparsers.add_parser("sat")
sat_parser.add_argument("--gxd-spp", default=1, type=int)
sat_parser.add_argument("--bg-spp", default=128, type=int)
sat_parser.add_argument("--shadow-spp", default=128, type=int)

ours_parser = subparsers.add_parser("ours")
ours_parser.add_argument("--gxd-spp", default=1, type=int)
ours_parser.add_argument("--bg-spp", default=128, type=int)
ours_parser.add_argument("--shadow-spp", default=128, type=int)

args = parser.parse_args()
######

###### Data
base_dir = os.path.join("scenes", "car")
scene_path = os.path.join(base_dir, "scene.xml")
mipmap_path = os.path.join("data", "sat", "flakes", "mip_hierarchy.npy")
sat_path = os.path.join("data", "sat", "flakes", f"9_32_sat")
model_path = os.path.join("data", "sat", "flakes", f"9_32_model")
uv_scale = 40
envmap_path = os.path.join(os.path.dirname(scene_path), "envmaps", "uffizi.exr")
envmap_scale = 2

denoiser = Denoiser(mi.ScalarVector2u(args.resx, args.resy), 3, True, True, True)
######

###### Rendering functions
def compose_img(
    hist_mod: torch.nn.Module,
    bin_centers: np.ndarray,
    alpha: float,
    resx: int,
    resy: int,
    bg_spp: int,
    shadow_spp: int,
    gxd_spp: int,
) -> Tuple[np.ndarray,np.ndarray,float]:
    common_params = {
        "alpha": alpha,
        "resx": resx,
        "resy": resy,
    }
    # Render with SH
    scene = mi.load_file(scene_path, spp=1, mode="-SH", sample_center=True, **common_params)

    sh, fg_pixels, hist_time, prod_time  = render_fastdot(
        scene, hist_mod, envmap_path, 
        bin_centers=bin_centers, envmap_scale=envmap_scale,
        resx=resx, resy=resy, uv_scale=uv_scale, sh_order=40,
        fast_rotation=not args.lookup, alpha_common=alpha,
    )

    # Render with GxD
    gxd, gxd_time = render_gxd(
        scene, mipmap_path,
        spp=gxd_spp, uv_scale=uv_scale,
        **common_params
    )

    # Render background
    scene = mi.load_file(
        scene_path,
        spp=bg_spp, sample_center=False,
        rfilter="box", mode="", **common_params
    )

    t0 = time.time()
    bg = render_multi_pass(mi.render, resx, resy, scene, bg_spp)
    bg_time = time.time() - t0

    # Render shadows
    shadowed, unshadowed, ratio_time = ratio_estimator(scene, spp=shadow_spp)

    # Denoise everything
    denoised, denoise_time = denoiser.denoise(scene, [bg, shadowed, unshadowed])
    bg, shadowed, unshadowed = denoised
    ratio = np.where(unshadowed > 1e-6, shadowed / unshadowed, 0)

    result = np.zeros((resy, resx, 3), np.float32)
    result[~fg_pixels] = bg[~fg_pixels]
    result[fg_pixels] = gxd[fg_pixels] + (sh * ratio)[fg_pixels]

    total_time = sum(hist_time) + sum(prod_time) + gxd_time + bg_time + ratio_time + denoise_time

    return result, fg_pixels.astype(np.float32), total_time

def gt(
    out_path: str,
    alpha: float,
    resx: int,
    resy: int,
    spp: int,
):
    scene = mi.load_file(
        scene_path,
        resx=resx, resy=resy, spp=spp,
        sample_center=False, mode="",
        alpha=alpha
    )
    img = render_multi_pass(mi.render, resx, resy, scene, spp)
    write_image(out_path, img)

def sat(
    out_path: str,
    alpha: float,
    resx: int,
    resy: int,
    bg_spp: int,
    shadow_spp: int,
    gxd_spp: int,
):
    sat, bin_centers, _ = open_sat(sat_path)
    sat = sat.astype(np.int32)
    sat = sat.reshape(sat.shape[0], sat.shape[1], -1)
    sat = TorchSAT(torch.from_numpy(sat).to(device))

    args = {
        "hist_mod": sat,
        "bin_centers": bin_centers,
        "alpha": alpha,
        "resx": resx,
        "resy": resy,
        "bg_spp": bg_spp,
        "shadow_spp": shadow_spp,
        "gxd_spp": gxd_spp,
    }

    img, _, __ = compose_img(**args)
    write_image(out_path, img)

def ours(
    out_path: str,
    alpha: float,
    resx: int,
    resy: int,
    bg_spp: int,
    shadow_spp: int,
    gxd_spp: int,
):
    _, bin_centers, __ = open_sat(sat_path)

    sat = TorchWrapper(sat_path, model_path, PredHist, full_hist=False, use_sat=False)
    sat.to(device)

    args = {
        "hist_mod": sat,
        "bin_centers": bin_centers,
        "alpha": alpha,
        "resx": resx,
        "resy": resy,
        "bg_spp": bg_spp,
        "shadow_spp": shadow_spp,
        "gxd_spp": gxd_spp,
    }

    img, _, __ = compose_img(**args)
    write_image(out_path, img)
#######

####### Run
common_args = {
    "out_path": args.out_path,
    "resx": args.resx,
    "resy": args.resy,
    "alpha": args.alpha,
}
if args.method == "gt":
    gt(**common_args, spp=args.spp)
elif args.method == "sat":
    sat(**common_args, gxd_spp=args.gxd_spp, bg_spp=args.bg_spp, shadow_spp=args.shadow_spp)
else:
    ours(**common_args, gxd_spp=args.gxd_spp, bg_spp=args.bg_spp, shadow_spp=args.shadow_spp)
#######