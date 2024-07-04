import mitsuba as mi
import drjit as dr
import time
import numpy as np
import os
from utils.fs import write_image, linear_to_srgb # Imported for backward compat
from tqdm import trange
from typing import Tuple, Callable

def get_spp_per_pass(res_x: int, res_y: int, spp: int) -> Tuple[int, int]:
    spp_per_pass = int(spp)
    samples_per_pass = res_x * res_y * spp_per_pass
    wavefront_size_limit = int(2**32)
    if samples_per_pass > wavefront_size_limit:
        spp_per_pass = spp_per_pass // int((samples_per_pass + wavefront_size_limit - 1) / wavefront_size_limit)
        n_passes = spp // spp_per_pass
        samples_per_pass = res_x * res_y * spp_per_pass
        mi.Log(mi.LogLevel.Warn, f"Too many samples requested, splitting the job into {n_passes} passes with {spp_per_pass} spp")
        return spp_per_pass, n_passes
    else:
        return spp_per_pass, 1

def render_multi_pass(
    render_func: Callable,
    res_x: int, res_y: int,
    scene: mi.Scene, spp: int,
    save_path: str=None, save_aov: bool=True,
    average_passes: bool=True, seed: int=0
) -> np.ndarray:
    t0 = time.time()
    samples_per_pass, n_passes = get_spp_per_pass(res_x, res_y, spp)
    aovs = []
    film: mi.Film = scene.sensors()[0].film()
    for pass_ in range(n_passes):
        # Render
        render_func(scene=scene, spp=samples_per_pass, seed=pass_+seed)
        bmp = film.bitmap()

        # Exctract AOV names and values
        result = []
        components = bmp.split()
        for component in components:
            name = component[0]
            num_channels = 0
            result.append(mi.TensorXf(component[1]).numpy())
            for channel in component[1].struct_():
                if pass_ == 0:
                    aovs.append(name+"."+channel.name)
                num_channels += 1
        result = np.concatenate(result, axis=-1)

        if pass_ == 0:
            final_result = result
        else:
            final_result += result 

    if average_passes:
        final_result = final_result / n_passes

    if save_path:
        if ".exr" in save_path and save_aov:
            bmp = mi.Bitmap(final_result, channel_names=aovs)
        else:
            bmp = mi.Bitmap(final_result[...,:3])
        write_image(save_path, bmp)

    final_result = final_result[...,:3]
    mi.Log(mi.LogLevel.Info, f"Time taken: {time.time() - t0:.2f}s", )

    return final_result

def generate_rays(
        scene: mi.Scene,
        spp: int = None,
        random_offset: bool=False,
        pixel_offset: mi.Point2f=mi.Point2f(0.5),
        seed: int=0
    ) -> Tuple[mi.Ray3f,mi.Float,mi.Point2f]:
    # Sensor & Film
    sensor = scene.sensors()[0]
    film = sensor.film()
    film_size = film.crop_size() # [width, height] image size
    if film.sample_border():
        # For correctness, we need to sample extra pixels on the border
        # Otherwise, convolution will have black pixels at the border
        # film.rfilter().border_size() is mult. by 2 to account for left/top & right/bottom borders
        film_size += 2 * film.rfilter().border_size()

    sampler: mi.Sampler = sensor.sampler()
    spp = spp or sampler.sample_count()
    film.prepare([]) # Allocate GPU mem

    # Wavefront setup
    wavefront_size = film_size.x * film_size.y * spp

    sampler.set_sample_count(spp)
    sampler.set_samples_per_wavefront(spp) # There are 'spp' number of passes
    sampler.seed(seed, wavefront_size)

    idx = dr.arange(mi.UInt32, 0, wavefront_size)
    idx = idx // mi.UInt32(spp)

    pos = mi.Vector2u(0)
    pos.y = idx // film_size.x
    pos.x = -film_size.x * pos.y + idx

    if film.sample_border():
        pos = pos - film.rfilter().border_size()

    pos = pos + film.crop_offset()

    diff_scale_factor = dr.rsqrt(spp)

    ################################
    # Camera rays
    ################################
    scale = 1.0 / mi.ScalarVector2f(film.crop_size())
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale

    if random_offset:
        sample_pos = pos + sampler.next_2d()
    else:
        sample_pos = pos + pixel_offset

    adjusted_pos = sample_pos * scale + offset # Float in range [0, 1] in each dimension [width, height]

    aperture_sample = mi.Point2f(0)
    if sensor.needs_aperture_sample():
        aperture_sample = sampler.next_2d()

    time = sensor.shutter_open()
    if sensor.shutter_open_time() > 0.0:
        time = time + sampler.next_1d() * sensor.shutter_open_time()
    
    wavelength_sample = mi.Float(0)

    ray, ray_weight = sensor.sample_ray_differential(
        time, wavelength_sample, adjusted_pos, aperture_sample
    )
    if ray.has_differentials and random_offset:
        ray.scale_differential(diff_scale_factor)

    return ray, ray_weight, sample_pos