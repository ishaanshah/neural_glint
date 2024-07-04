import os
import drjit as dr
import mitsuba as mi
from typing import Tuple
from utils.sh.base import SHBase

class Gaussian(SHBase):
    """
    The spherical gaussian function represented in Spherical Harmonics basis.
    The function is stored as a 4D function with the 'cos' function oriented
    in various directions.
    """
    def __init__(
        self,
        order: int,
        ntheta: int=64,
        nphi: int=128,
        nsamples: int=int(1e6),
        force_eval: bool=False,
        **kwargs
    ) -> None:
        self.alpha = kwargs["alpha"]

        self.cache_dir = os.path.join("data", "sh", "gaussian")
        self.dim = 4
        self.nchannels = 1
        self.upper_hemisphere = True
        self.zonal_only = True

        super().__init__(order, ntheta, nphi, nsamples, force_eval, **kwargs)

    def eval(self, w1: mi.Vector3f, w2: mi.Vector3f=None, **kwargs) -> mi.Float|mi.Vector3f:
        if self.project_on_fly:
            alpha = kwargs["alpha"]
        else:
            alpha = self.alpha

        if self.project_on_fly:
            frame = mi.Frame3f(w2)
            w1 = frame.to_local(w1)
        
        alpha_uv = float(alpha*alpha)
        result = dr.rcp(2 * dr.pi * alpha_uv) * \
                 dr.exp(-0.5 * dr.rcp(alpha_uv) * (dr.sqr(w1.x) + dr.sqr(w1.y)))

        return result

    def sample(self, eta: mi.Point2f, w: mi.Vector3f=None, **kwargs) -> Tuple[mi.Vector3f, mi.Float]:
        if self.project_on_fly:
            alpha = kwargs["alpha"]
        else:
            alpha = self.alpha

        # Sample canonical Beckmann
        D = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, alpha, sample_visible=False)
        w_query, pdf = D.sample(mi.Vector3f(0, 0, 1), eta)

        # Rotate the sample to normal orientation
        if self.project_on_fly:
            frame = mi.Frame3f(w)
            w_query = frame.to_world(w_query)

        return w_query, pdf
    
    def glob_str(self) -> str:
        if self.rotate_on_fly:
            return f"coeffs_{self.alpha:.4f}_*.npy"
        return f"coeffs_{self.ntheta}_{self.nphi}_{self.alpha:.4f}_*.npy"