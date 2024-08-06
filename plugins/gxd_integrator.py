import mitsuba as mi
import drjit as dr
import torch
import numpy as np
import gxd
import time
from utils.sat import open_mipmap
from typing import Tuple
from utils.render import render_manual

import ipdb

def render_gxd(
    scene: mi.Scene,
    mipmap_path: str,
    alpha: float,   # TODO: Allow SVR
    uv_scale: 20,
    resx: int=1024,
    resy: int=1024,
    spp: int=1,
) -> Tuple[np.ndarray, float]:
    mipmap = open_mipmap(mipmap_path)
    mipmap = torch.tensor(mipmap, device="cuda:0")

    def render(rays: mi.RayDifferential3f, scene: mi.Scene):
        si: mi.SurfaceInteraction3f = scene.ray_intersect(rays)
        si.compute_uv_partials(rays)

        # We don't swap here because swap is done in GxD CUDA kernel
        duv = (dr.abs(si.duv_dx) + dr.abs(si.duv_dy)) * uv_scale
        duv.x = dr.minimum(duv.x, 1)
        duv.y = dr.minimum(duv.y, 1)
        cuv = si.uv * uv_scale
        glint_pixels = si.bsdf().has_attribute("glint_idx")
        bsdf = si.bsdf()

        time = 0
        result = mi.Color3f(0)
        emitters = scene.emitters()
        for emitter in emitters:
            if mi.has_flag(emitter.flags(), mi.EmitterFlags.DeltaPosition):
                ds, intensity = emitter.sample_direction(si, mi.Point2f(0))

                # Find half normal
                wo = si.to_local(ds.d)
                wh = dr.normalize(wo + si.wi)
                fg = bsdf.eval(mi.BSDFContext(), si, wo)

                occluded = scene.ray_test(si.spawn_ray_to(ds.p))

                L_fg = fg * intensity
                active = dr.and_(si.wi.z > 0,  wo.z > 0)
                active = dr.and_(active, ~occluded)
                active = dr.and_(active, glint_pixels)
                dr.eval(wh, cuv, duv, L_fg, active, occluded)

                duv = duv.torch().reshape(resy, resx, -1, 2)
                cuv = cuv.torch().reshape(resy, resx, -1, 2)
                wh = wh.torch().reshape(resy, resx, -1, 3)
                active_torch = active.torch().reshape(resy, resx, -1)
                output = torch.zeros((resy, resx, wh.shape[2]), device="cuda:0")
                time += gxd.eval_accel(wh, cuv, duv, active_torch, mipmap, output, alpha)

                result += mi.Color3f(L_fg.torch() * output.reshape(-1, 1))

        mi.Log(mi.LogLevel.Info, f"Time taken: {time / 1000:.3f}s")

        return result

    sample_center = spp == 1
    t0 = time.time()
    result = render_manual(render, scene, spp, random_offset=not sample_center)

    return result.numpy(), time.time() - t0