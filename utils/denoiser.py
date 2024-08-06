import mitsuba as mi
import time
import numpy as np
from typing import List, Tuple

class Denoiser:
    def __init__(self, size: mi.Vector2u, num_images: int, enable_normals: bool, enable_albedo: bool, enable_temporal: bool):
        self.bmps = [None] * num_images
        self.prev_bmps = [None] * num_images
        
        # Initialize 
        self.denoiser = mi.OptixDenoiser(size, enable_albedo, enable_normals, enable_temporal)

        if enable_albedo and enable_normals:
            aov = "albedo:albedo,normals:sh_normal"
        elif enable_normals:
            aov = "normals:sh_normal"
        elif enable_albedo:
            aov = "albedo:albedo"
        
        self.integrator = None
        if enable_albedo or enable_normals:
            self.integrator = mi.load_dict({
                "type": "aov",
                "aovs": aov,
                "integrator": {
                    "type": "path"
                },
                "sample_center": True
            })

    def denoise(self, scene: mi.Scene, noisy: List[np.ndarray], flow: np.ndarray=None, sensor_idx: int=0) -> Tuple[List[np.ndarray],float]:
        assert len(noisy) == len(self.bmps)

        t0 = time.time()
        sensor: mi.Sensor = scene.sensors()[sensor_idx]
        to_sensor = sensor.world_transform().inverse()

        if flow is None:
            flow = np.zeros((*noisy[0].shape[:2], 2), dtype=np.float32)

        normal = []
        albedo = []
        if not self.integrator is None:
            mi.render(scene, integrator=self.integrator, spp=1)

            bmp = sensor.film().bitmap()
            bmps = bmp.split()
            for name, bmp in bmps:
                if name == "normals":
                    normal = mi.TensorXf(bmp)
                elif name == "albedo":
                    albedo = mi.TensorXf(bmp)

        for i, img in enumerate(noisy):
            # TODO: Temporal stuff
            if self.prev_bmps[i] is None:
                prev_bmp = img
            else:
                prev_bmp = self.prev_bmps[i]
            self.prev_bmps[i] = self.denoiser(
                img,
                albedo=albedo, normals=normal,
                previous_denoised=prev_bmp, flow=flow,
                to_sensor=to_sensor
            ).numpy()

        return self.prev_bmps, time.time() - t0