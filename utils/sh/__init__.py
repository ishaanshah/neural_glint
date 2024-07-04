"""
This file contains various utility functions needed for projection functions to Spherical Harmonics
"""
import mitsuba as mi
import drjit as dr
import numpy as np
from torch import Tensor
from tqdm import tqdm
from typing import Tuple

import ipdb

### Utils
def get_num_coeff(order: int) -> int:
    return (order+1)**2

def get_idx(l: mi.Int|Tensor, m: mi.Int|Tensor) -> mi.Int|Tensor:
    return l * l + l + m

def get_lm(idx: int) -> Tuple[int, int]:
    idx = mi.Int(idx)
    l = mi.Int(dr.sqrt(mi.Float(idx)))
    m = idx - l*l - l
    return l, m

### Evaluation
def k(l: int, m: int) -> mi.Float64:
    """
    Evaluate the normalization factor for Legendre polynomial of degree 'l'
    and order 'm'

    Args:
        l: The degree of polynomial
        m: The order of polynomial
    Returns:
        klm: The normalization factor for p_(l,m)
    """
    fact = np.prod(np.arange(l+abs(m), l-abs(m), -1, dtype=np.float128))
    klm = np.sqrt((2*l + 1) * dr.inv_four_pi * np.reciprocal(fact))
    return mi.Float64(klm.astype(np.float64))

def eval_legendre(x: mi.Float, l: int, m: int) -> mi.Float64:
    """
    Evaluate Legendre polynomials at 'x'

    Args:
        x: The variable to evaluate
        l: The degree of polynomial
        m: The order of polynomial
    Returns:
        plm: The polynomial evaluated at 'x'
    """
    assert 0 <= m and m <= l, f"'m' should be between '0' and 'l (={l})'"
    
    xd = mi.Float64(x) # Represent x in double precision to avoid implicit casting

    if l == 0:
        return mi.Float64(1)
    
    # m = 0
    if m == 0:
        if l == 1:
            return xd

        plm2 = mi.Float64(1)    # p_(l-2), initialize with p_0 = 1
        plm1 = mi.Float64(xd)   # p_(l-1), initialize with p_1 = x
        pl0 = mi.Float64(0)     # p_l
        cur_l = mi.UInt(2)
        loop = mi.Loop("Eval SH (m = 0)", lambda: (cur_l, pl0, plm2, plm1))
        while loop(cur_l <= l):
            pl0 = ((2*cur_l - 1)*xd*plm1 - (cur_l-1)*plm2) / mi.Float64(cur_l)
            plm2 = plm1
            plm1 = pl0
            cur_l += 1

        return pl0
    
    # m = l
    pll = mi.Float64(1)
    somx2 = dr.sqrt(dr.maximum(mi.Float64(0), (1 - xd) * (1 + xd)))
    fact = mi.Float64(1)
    idx = mi.UInt(1)
    loop = mi.Loop("Eval SH (m = l)", lambda: (fact, pll, idx))
    while loop(idx <= m):
        pll *= -fact * somx2
        fact += mi.Float64(2)
        idx += 1

    if l == m:
        return pll

    # m = l-1
    pllm1 = mi.Float64(xd * (2*m+1)) * pll
    if m == l-1:
        return pllm1

    plm = mi.Float64(0)
    idx = mi.Int(m+2)
    loop = mi.Loop("Eval SH", lambda: (plm, pllm1, pll, idx))
    while loop(idx <= l):
        plm = (((2 * idx - 1) * xd * pllm1) - ((idx + m - 1) * pll)) / (idx - m)
        pll = pllm1
        pllm1 = plm
        idx += 1

    return plm

def eval_sh(
    w: mi.Vector3f, l: int, m: int,
    texture: mi.Texture2f, max_idx: int,
    sin_texture: mi.Texture1f=None, cos_texture: mi.Texture1f=None
) -> mi.Float64:
    """
    Evaluate SH of degree l, order m at directions ω. The function use double
    precision to avoid precision issues for higher orders. If `texture` parameter
    is set, then the result is returned by performing bilinear interpolation.

    Args:
        w: The directions to evaluate SH at, directions should be normalized
        l: The degree of SH
        m: The order of SH
        texture: Use precomputed legendre polynomial values from texture
        max_idx: Maximum order SH values stored in the provided texture
        texture_sin: The texture consisting of sin values
        texture_cos = The texture consisting of cos values
    Returns:
        shlm: Spherical harmonic evaluation at 'x'
    """
    idx = get_idx(l, m)
    shlm = texture.eval(mi.Point2f((idx+0.5) / max_idx, dr.acos(w.z) / dr.pi))[0]   # Cos parametrization
    # shlm = texture.eval(mi.Point2f((idx+0.5) / max_idx, (w.z+1)/2))[0]
    
    angle = dr.abs(m) * dr.atan2(w.y, w.x)
    angle -= mi.Int(angle / dr.two_pi) * dr.two_pi

    # TODO: Allow cos texture
    # if not cos_texture is None:
    #     shlm *= cos_texture.eval(angle / dr.two_pi)[0]
    # if not sin_texture is None:
    #     shlm *= sin_texture.eval(angle / dr.two_pi)[0]
    shlm *= dr.select(m > 0, dr.cos(angle) * dr.sqrt_two, 1)
    shlm *= dr.select(m < 0, dr.sin(angle) * dr.sqrt_two, 1)

    return shlm

def rotate_zh(coeffs: np.ndarray, w: mi.Vector3f, texture: mi.Texture2f, max_idx: int) -> np.ndarray:
    rot_coeffs = np.zeros((dr.width(w), *coeffs.shape))
    sh_order = int(np.sqrt(coeffs.shape[0]))
    bar = tqdm(total=coeffs.shape[0], desc="Computing SH rotation")
    for l in range(sh_order):
        nl = np.sqrt((4*np.pi) / (2*l + 1))
        for m in range(-l, l+1):
            idx = get_idx(l, m)
            sh_coeffs = eval_sh(w, l, m, texture, max_idx)
            rot_coeffs[:,idx] = (nl * sh_coeffs * coeffs[l*l+l])[...,None]
            bar.update(1)

    return rot_coeffs