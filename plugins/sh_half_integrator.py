import os
import mitsuba as mi
import drjit as dr
import torch
import numpy as np
import fast_dot
import time
from utils.render import generate_rays
from utils.sh.gaussian import Gaussian
from utils.sh.envmap import EnvmapHalf
from typing import Tuple

def render_half_fastdot(
    scene: mi.Scene,
    envmap_path: str,    # Envmap details
    envmap_scale: float=1,
    envmap_proj_meth: str="mc",
    sh_order: int=60,
    resx: int=1024,
    resy: int=1024,
    fast_rotation: bool=True,
) -> Tuple[np.ndarray, np.ndarray]:
    max_idx = (sh_order+1)**2

    # Start rendering
    start_time = time.time()

    rays, _, pos = generate_rays(scene, 1)
    si: mi.SurfaceInteraction3f = scene.ray_intersect(rays)

    # Find out which pixels use SH dot product
    bsdf = si.bsdf()
    sh_pixels = bsdf.has_attribute("glint_idx")
    # Prepare G buffers
    wi_world = si.to_world(si.wi)
    normal = si.sh_frame.n
    alpha = bsdf.eval_attribute_1("alpha", si)

    # Evaluate this beforehand to avoid running the
    # ray-tracing kernel for each theta
    dr.eval(wi_world, si.wi, normal, si.p, alpha, sh_pixels)

    sh_pixels = sh_pixels.torch().reshape(resy, resx).cuda()
    alpha = alpha.torch().reshape(resy, resx).cuda()
    wi_world = wi_world.torch().reshape(resy, resx, -1).cuda()
    normal = normal.torch().reshape(resy, resx, -1).cuda()

    # We want to keep loading time separate from runtime
    # but we need to start tracing to find which alpha coeffs to load.
    # Hence, track the time needed for loading and subtract it from
    # the total runtime of the algorithm.
    load_start_time = time.time()
    if fast_rotation:
        bsdf_coeffs = torch.from_numpy(np.load(os.path.join("data", "sh", "gaussian", "coeffs_1000.npy"))).cuda()
        bsdf_coeffs = bsdf_coeffs[:max_idx]
        l_coeffs = torch.from_numpy(np.load("data/sh/lcoeffs.npy")).to("cuda:0")
        l_coeffs = l_coeffs.transpose(0, 1)
        l_coeffs = l_coeffs.contiguous()
        l_coeffs = l_coeffs[:max_idx]
    else:
        bsdf_coeffs = Gaussian(sh_order, alpha=alpha[sh_pixels==1].mean(), rotate_on_fly=False)
        bsdf_coeffs = torch.from_numpy(np.load(bsdf_coeffs.cache_path())).to("cuda:0")
        bsdf_coeffs = bsdf_coeffs.permute(2, 0, 1, 3)   # Make SH the first index
        bsdf_coeffs = bsdf_coeffs.contiguous()  # This is important as the previous operation returns a view of the tensor
        bsdf_coeffs = bsdf_coeffs[:max_idx]

    envmap = EnvmapHalf(sh_order, envmap=envmap_path, project_method=envmap_proj_meth)
    emitter_coeffs = torch.from_numpy(np.load(envmap.cache_path()))
    emitter_coeffs = emitter_coeffs.permute(2, 0, 1, 3)   # Make SH the first index
    emitter_coeffs = emitter_coeffs.contiguous()  # This is important as the previous operation returns a view of the tensor
    emitter_coeffs = emitter_coeffs[:max_idx].to("cuda:0")  # Send to device at end to avoid memory overflow
    load_time = time.time() - load_start_time

    result = torch.zeros((resy, resx, 3), device="cuda:0")
    if fast_rotation:
        fast_dot.render_half_fast_rotation(normal, wi_world, alpha, bsdf_coeffs, emitter_coeffs, l_coeffs, result, sh_order)
    else:
        fast_dot.render_half_lookup(normal, wi_world, bsdf_coeffs, emitter_coeffs, result, sh_order)
    result *= envmap_scale

    # FG decoupling
    ctx = mi.BSDFContext()
    wo = mi.reflect(si.wi)
    fg = bsdf.eval(ctx, si, wo).torch().reshape(resy, resx, 3)
    result = result * fg

    # Put it into the film
    result = mi.Color3f(result.reshape(-1, 3))
    result = [result.x, result.y, result.z, mi.Float(1)]

    # Develop the film
    film = scene.sensors()[0].film()
    # Image block
    block = film.create_block()
    # Offset is the currect location of the block
    # In case of GPU, the block covers the entire image, hence offset is 0
    block.set_offset(film.crop_offset())

    ################################
    # Save image
    ################################
    block.put(pos, result)
    film.put_block(block)
    result = film.develop().numpy()

    end_time = time.time()
    mi.Log(mi.LogLevel.Info, f"Time taken: {end_time-start_time-load_time:.2f}")

    return result, sh_pixels.cpu().numpy()