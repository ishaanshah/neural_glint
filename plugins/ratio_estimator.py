import mitsuba as mi
import drjit as dr
import numpy as np
import time
from utils.render import generate_rays

def ratio_estimator(scene: mi.Scene, spp: int=1024):
    start_time = time.time()
    rays, _, pos = generate_rays(scene, spp, random_offset=True)
    si: mi.SurfaceInteraction3f = scene.ray_intersect(rays)

    ctx = mi.BSDFContext()    

    active = si.is_valid()

    rng = mi.PCG32(dr.width(active))

    unshadowed = dr.select(active, mi.Color3f(0), 1)
    shadowed = dr.select(active, mi.Color3f(0), 1)

    bsdf: mi.BSDF = si.bsdf(rays)

    # Emitter Sampling
    ds, emitter_val = scene.sample_emitter_direction(si, mi.Point2f(rng.next_float32(active), rng.next_float32(active)), False, active)
    active_e = active & dr.neq(ds.pdf, 0.0)
    wo = si.to_local(ds.d)
    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
    # Note that we are give 0 weight to delta lights
    # This is because it doesn't make sense to use ratio estimator
    # for delta lights.
    mis = dr.select(ds.delta, 0, ds.pdf / (ds.pdf + bsdf_pdf))
    occluded = scene.ray_test(si.spawn_ray(ds.d), active_e)
    unshadowed += dr.select(active_e, emitter_val * bsdf_val * mis, 0)
    shadowed += dr.select(active_e & ~occluded, bsdf_val * emitter_val * mis, 0)

    # BSDF Sampling
    # WARNING: Delta BSDFs are not handled
    bs, bsdf_val = bsdf.sample(ctx, si, rng.next_float32(active), mi.Point2f(rng.next_float32(active), rng.next_float32(active)), active)
    active_b = active & dr.any(dr.neq(bsdf_val, 0))
    wo = si.to_world(bs.wo)
    occluded = scene.ray_test(si.spawn_ray(wo), active_b)
    envmap: mi.Emitter = scene.environment()
    si_bsdf = dr.zeros(mi.SurfaceInteraction3f, dr.width(active_b))
    si_bsdf.wi = -wo
    emitter_val = envmap.eval(si_bsdf, active_b)
    emitter_pdf = scene.pdf_emitter_direction(si, mi.DirectionSample3f(scene, si_bsdf, si), active_b)
    mis = bs.pdf / (bs.pdf + emitter_pdf)
    unshadowed += dr.select(active_b, bsdf_val * emitter_val * mis, 0)
    shadowed += dr.select(~occluded & active_b, bsdf_val * emitter_val * mis, 0)

    film: mi.Film  = scene.sensors()[0].film()

    # Develop the film for unshadowed
    result = [unshadowed.x, unshadowed.y, unshadowed.z, mi.Float(1)]
    film.clear()
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
    unshadowed = film.develop().numpy()

    # Develop the film for unshadowed
    result = [shadowed.x, shadowed.y, shadowed.z, mi.Float(1)]
    film.clear()
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
    shadowed = film.develop().numpy()

    if np.any(np.isnan(shadowed)):
        count = np.count_nonzero(np.isnan(shadowed).any(-1))
        mi.Log(mi.LogLevel.Warn, f"{count} NaN found in 'shadowed'")
    if np.any(np.isnan(unshadowed)):
        count = np.count_nonzero(np.isnan(unshadowed).any(-1))
        mi.Log(mi.LogLevel.Warn, f"{count} NaN found in 'unshadowed'")

    end_time = time.time()

    return shadowed, unshadowed, end_time - start_time