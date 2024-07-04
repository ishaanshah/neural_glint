import mitsuba as mi

import os
import drjit as dr
from utils.sh.base import SHBase
from utils.fs import split_path
from typing import Tuple

class EnvmapHalf(SHBase):
    """
    The environment map represented in Spherical Harmonics basis suitable for
    half-basis integration. The function is stored as a 4D function with
    the envmap being viewed from different directions as the reduciton parameter.
    """
    def __init__(
        self,
        order: int,
        ntheta: int=64,
        nphi: int=128,
        nsamples: int=int(1e6),
        force_eval: bool=False,
        project_on_fly: bool=False,
        **kwargs
    ) -> None:
        envmap = kwargs["envmap"]
        _, envmap_name, _ = split_path(envmap)

        self.envmap: mi.Emitter = mi.load_dict({
            "type": "envmap",
            "filename": envmap
        })

        self.cache_dir = os.path.join("data", "sh", "envmaps", envmap_name)
        self.dim = 4
        self.nchannels = 3
        self.upper_hemisphere = False
        self.zonal_only = False
        self.fd_step_size = 1e-3
        self.project_on_fly = project_on_fly

        super().__init__(order, ntheta, nphi, nsamples, force_eval, project_on_fly=project_on_fly, **kwargs)

    def eval(self, w1: mi.Vector3f, w2: mi.Vector3f=None) -> mi.Float|mi.Vector3f:
        # w2 is the view direction (wo)
        # The view direction points towards normal, i.e dot(wo, n) >= 0
        si = dr.zeros(mi.SurfaceInteraction3f)
        
        # w1 is the half vector (wh)
        # Find incoming direction by reflecting about half vector
        si.wi = -mi.reflect(w2, w1)

        result = self.envmap.eval(si)
        # Multiply by Jacobian for half vector space integration
        result *= dr.maximum(4*dr.dot(dr.normalize(w1), dr.normalize(w2)), 0)

        return result

    def sample(self, eta: mi.Point2f, w: mi.Vector3f) -> Tuple[mi.Vector3f, mi.Float]:
        si = dr.zeros(mi.SurfaceInteraction3f)
        ds, _ = self.envmap.sample_direction(si, eta)

        if self.project_on_fly:
            wh = dr.normalize(ds.d + w)
            return wh, ds.pdf * dr.maximum(4*dr.dot(wh, ds.d), 0)

        nsamples = dr.width(eta)
        nb = dr.width(w)

        wi = ds.d
        wi = dr.tile(wi, nb)
        w = dr.repeat(w, nsamples)
        wh = dr.normalize(wi + w)

        pdf = ds.pdf
        dr.eval(pdf)    # If we dont do this, drjit stalls on this for some reason
        pdf = dr.tile(pdf, nb)

        return wh, pdf * dr.maximum(4*dr.dot(wh, wi), 0)

    def pol2cart(self, pol: mi.Point2f):
        return mi.Vector3f(
            dr.sin(pol.x)*dr.sin(pol.y),
            dr.cos(pol.x),
            dr.sin(pol.x)*dr.cos(pol.y),
        )

    def glob_str(self) -> str:
        return f"coeffs_half_{self.project_method}_{self.ntheta}_{self.nphi}_*.npy"