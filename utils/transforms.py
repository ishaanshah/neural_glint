import mitsuba as mi
import drjit as dr
import torch
from torch import Tensor

def pol2cart(pol: mi.Point2f|Tensor) -> mi.Vector3f|Tensor:
    """
    Convert from polar to cartesian

    Args:
        pol.x: θ
        pol.y: ϕ
    """
    if isinstance(pol, Tensor):
        lib = torch
        theta = pol[...,0]
        phi = pol[...,1]
    else:
        lib = dr
        theta = pol.x
        phi = pol.y

    sin_theta = lib.sin(theta)
    cos_theta = lib.cos(theta)
    sin_phi = lib.sin(phi)
    cos_phi = lib.cos(phi)

    if isinstance(pol, Tensor):
        return torch.stack([
            sin_theta*cos_phi,
            sin_theta*sin_phi,
            cos_theta
        ], -1)
    else:
        return mi.Vector3f(
            sin_theta*cos_phi,
            sin_theta*sin_phi,
            cos_theta
        )

def cart2pol(cart: mi.Vector3f|Tensor, clip_upper: bool=False) -> mi.Point2f|Tensor:
    """
    Convert from cartesian to polar
    """
    if isinstance(cart, Tensor):
        lib = torch
        x = cart[...,0]
        y = cart[...,1]
        z = cart[...,2]
    else:
        lib = dr
        x = cart.x
        y = cart.y
        z = cart.z

    theta = lib.clip(lib.acos(z), 0, lib.pi/2 if clip_upper else lib.pi)
    phi = lib.clip(lib.atan2(y, x), -lib.pi, lib.pi)

    if isinstance(cart, Tensor):
        return torch.stack([theta, phi], -1)
    else:
        return mi.Point2f(theta, phi)