"""
This script project the environment map to Spherical Harmonics
so that they are suitable for Half-Space integration.
"""
import mitsuba as mi
mi.set_variant("cuda_rgb", "cuda_ad_rgb")
mi.set_log_level(mi.LogLevel.Info)

from argparse import ArgumentParser
from utils.sh.envmap import EnvmapHalf

parser = ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--method", choices=["mc", "simpsons"], type=str, default="mc")
parser.add_argument("--nsamples", default=int(1e6), type=int)
parser.add_argument("--norders", default=100, type=int)
parser.add_argument("--ntheta", default=64, type=int)
parser.add_argument("--nphi", default=128, type=int)
args = parser.parse_args()

envmap: mi.Emitter = mi.load_dict({
    "type": "envmap",
    "filename": args.filename
})

EnvmapHalf(args.norders, args.ntheta, args.nphi, envmap=args.filename, nsamples=args.nsamples, force_eval=True, project_method=args.method)