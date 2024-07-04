import mitsuba as mi
mi.set_variant("cuda_rgb")
mi.set_log_level(mi.LogLevel.Info)

import os
import plugins as _
from utils.render import render_multi_pass
from figures.fast_rotation.scene import config

scene_path = os.path.join("scenes", "teapot", "scene.xml")

for i in range(2):
    scene = mi.load_file(
        scene_path,
        resx=config["resx"], resy=config["resy"],
        spp=config["spp"], alpha=config[f"alpha_{i}"], glint=""
    )

    render = render_multi_pass(
        mi.render,
        config["resx"], config["resy"],
        scene, config["spp"],
        os.path.join("renders", "fast_rotation", f"gt_{i}.exr")
    )