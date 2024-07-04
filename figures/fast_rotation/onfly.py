import mitsuba as mi
mi.set_variant("cuda_rgb", "cuda_ad_rgb")
mi.set_log_level(mi.LogLevel.Info)

import os
import numpy as np
import plugins as _
from figures.fast_rotation.scene import config
from plugins.ratio_estimator import ratio_estimator
from plugins.sh_half_integrator import render_half_fastdot

sh_order = config["sh_order"]
scene_path = os.path.join("scenes", "teapot", "scene.xml")
envmap_path = os.path.join("scenes", "teapot", "envmaps", "indoor_1.hdr")

for i in range(2):
    scene = mi.load_file(
        scene_path,
        resx=config["resx"], resy=config["resy"],
        spp=config["spp"], alpha=config[f"alpha_{i}"],
        glint="-glint"
    )
    render, sh_pixels = render_half_fastdot(
        scene,
        envmap_path,
        resx=config["resx"], resy=config["resy"],
        fast_rotation=True
    )
    sh_pixels = sh_pixels.astype(bool)

    scene = mi.load_file(
        scene_path,
        resx=config["resx"], resy=config["resy"],
        spp=config["spp"], alpha=config[f"alpha_{i}"],
        glint=""
    )
    shadowed, unshadowed, _ = ratio_estimator(scene)
    ratio = np.where(unshadowed > 1e-6, shadowed / unshadowed, 0)
    bg = mi.render(scene, spp=64).numpy()

    render *= ratio
    render[~sh_pixels] = bg[~sh_pixels]

    mi.Bitmap(render).write(os.path.join("renders", "fast_rotation", f"onfly_{i}.exr"))