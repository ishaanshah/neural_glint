import os
import mitsuba as mi
import drjit as dr
import torch
import numpy as np
import fast_dot
import time
from utils.render import generate_rays
from utils.sh.gaussian import Gaussian
from utils.sh.envmap import EnvmapHalf
from utils.sat import bin_to_normal
from tqdm import tqdm

def load_coeffs(
    envmap_path: str,
    envmap_proj_meth: str,
    sh_order: int,
    fast_rotation: bool,
    alpha_common: float,
):
    max_idx = (sh_order+1)**2
    if fast_rotation:
        bsdf_coeffs = torch.from_numpy(np.load(os.path.join("data", "sh", "gaussian", "coeffs_1000.npy"))).cuda()
        bsdf_coeffs = bsdf_coeffs[:max_idx]
        l_coeffs = torch.from_numpy(np.load("data/sh/lcoeffs.npy")).to("cuda:0")
        l_coeffs = l_coeffs.transpose(0, 1)
        l_coeffs = l_coeffs.contiguous()
        l_coeffs = l_coeffs[:max_idx]
    else:
        bsdf_coeffs = Gaussian(sh_order, alpha=alpha_common, rotate_on_fly=False)
        bsdf_coeffs = torch.from_numpy(np.load(bsdf_coeffs.cache_path())).to("cuda:0")
        bsdf_coeffs = bsdf_coeffs.permute(2, 0, 1, 3)   # Make SH the first index
        bsdf_coeffs = bsdf_coeffs.contiguous()  # This is important as the previous operation returns a view of the tensor
        bsdf_coeffs = bsdf_coeffs[:max_idx]
        l_coeffs = None

    envmap = EnvmapHalf(sh_order, envmap=envmap_path, project_method=envmap_proj_meth)
    emitter_coeffs = torch.from_numpy(np.load(envmap.cache_path()))
    emitter_coeffs = emitter_coeffs.permute(2, 0, 1, 3)   # Make SH the first index
    emitter_coeffs = emitter_coeffs.contiguous()  # This is important as the previous operation returns a view of the tensor
    emitter_coeffs = emitter_coeffs[:max_idx].to("cuda:0")  # Send to device at end to avoid memory overflow

    return bsdf_coeffs, emitter_coeffs, l_coeffs

@torch.no_grad()
def render_fastdot(
    scene: mi.Scene,
    hist_mod: torch.nn.Module,
    envmap_path: str,    # Envmap details
    envmap_scale: float=1,
    envmap_proj_meth: str="mc",
    clearcoat_weight: float=0,
    uv_scale: int=20,
    ntheta: int=9,
    nphi: int=32,
    sh_order: int=60,
    resx: int=1024,
    resy: int=1024,
    bin_centers: np.ndarray=None,
    fast_rotation: bool=True,
    alpha_common: float=0.05,
    clearcoat_alpha_common: float=0.05,
    render_clearcoat: bool=False
):
    if bin_centers is None:
        bin_centers = (np.pi / 2) * ((np.arange(ntheta, dtype=np.float32) + 0.5) / ntheta)
    
    if not render_clearcoat:
        clearcoat_weight = 0

    # Load SH Coefficients of environment map
    bsdf_coeffs, emitter_coeffs, l_coeffs = load_coeffs(envmap_path, envmap_proj_meth, sh_order, fast_rotation, alpha_common)

    rays, _, pos = generate_rays(scene, 1)
    si: mi.SurfaceInteraction3f = scene.ray_intersect(rays)

    si.compute_uv_partials(rays)

    # Find out which pixels are glinty
    bsdf = si.bsdf()
    glint_pixels = bsdf.has_attribute("glint_idx")
    # Prepare G buffers
    alpha = bsdf.eval_attribute_1("alpha", si)
    alpha_mul = bsdf.eval_attribute_1("alpha_mul", si)
    clearcoat_alpha = bsdf.eval_attribute_1("clearcoat_alpha", si)
    clearcoat_alpha = dr.clamp(clearcoat_alpha, 0.01, 0.99)
    alpha = dr.clamp(alpha*alpha_mul, 0.01, 0.99)
    to_uv = mi.Transform4f.scale([uv_scale, uv_scale, 1])
    uv = to_uv.transform_affine(mi.Point3f(si.uv.y, si.uv.x, 0))
    uv = mi.Point2f(uv.x, uv.y)

    duv = dr.abs(si.duv_dx) + dr.abs(si.duv_dy)
    duv = to_uv.transform_affine(mi.Point3f(duv.y, duv.x, 0))
    duv = mi.Point2f(duv.x, duv.y)
    duv = mi.Point2f(dr.maximum(dr.abs(duv.x), dr.abs(duv.y)))  # Take relaxed square footprints

    wi_world = si.to_world(si.wi)
    onb_s = si.sh_frame.s
    onb_t = si.sh_frame.t
    onb_n = si.sh_frame.n

    # Evaluate this beforehand to avoid running the
    # ray-tracing kernel for each theta
    dr.eval(uv, duv, wi_world, si.wi, onb_s, onb_t, onb_n, si.p, glint_pixels, alpha, clearcoat_alpha)
    width = dr.width(uv)

    alpha = alpha.torch().reshape(resy, resx)
    clearcoat_alpha = clearcoat_alpha.torch().reshape(resy, resx)
    glint_pixels = glint_pixels.torch().reshape(resy, resx)
    # If none are glint, treat all as glint
    if not torch.any(glint_pixels):
        glint_pixels[:,:] = 1

    wi_world = wi_world.torch().reshape(resy, resx, -1).cuda()
    onb_s = onb_s.torch().reshape(resy, resx, -1).cuda()
    onb_t = onb_t.torch().reshape(resy, resx, -1).cuda()
    onb_n = onb_n.torch().reshape(resy, resx, -1).cuda()
    uv = uv.torch().cuda()
    duv = duv.torch().cuda()

    times_hist = []
    times_prod = []

    theta_idx = 0
    theta_iter = range(ntheta)

    # Create query to SAT
    sat_query = torch.cat([uv, duv], -1)
    sat_query = torch.nan_to_num(sat_query, 0)
    glint_mask = glint_pixels.bool().ravel()
    sat_query = sat_query[glint_mask]
    ctx = mi.BSDFContext()

    ##### Base Layer
    total = torch.zeros(width, device="cuda:0", dtype=torch.float32)
    result = torch.zeros((resy, resx, nphi, 3), device="cuda:0")
    base_result = torch.zeros((width, nphi, 3), device="cuda:0", dtype=torch.float32)

    for theta_idx in tqdm(theta_iter, total=ntheta, leave=False):
        ###### PyTorch  ######
        # Get histogram (h*w,nphi)
        torch.cuda.synchronize()
        t0 = time.time()
        hist = hist_mod(sat_query, theta_idx*nphi, nphi)
        torch.cuda.synchronize()
        times_hist.append(time.time()-t0)

        total[glint_mask] += hist.sum(-1)
        if fast_rotation:
            kernel_time = fast_dot.render_glint_fast_rotation(
                onb_s, onb_t, onb_n, wi_world, alpha,
                bsdf_coeffs, emitter_coeffs, l_coeffs, glint_pixels,
                result, bin_centers, theta_idx, sh_order, nphi
            )
        else:
            kernel_time = fast_dot.render_glint_lookup(
                onb_s, onb_t, onb_n, wi_world,
                bsdf_coeffs, emitter_coeffs, glint_pixels,
                result, bin_centers, theta_idx, sh_order, nphi
            )

        base_result[glint_mask] += result.reshape(-1, nphi, 3)[glint_mask] * hist.reshape(-1, nphi, 1)
        times_prod.append(kernel_time / 1000)

    # FG decoupling
    wo = mi.reflect(si.wi)
    fg = bsdf.eval(ctx, si, wo).torch()

    base_result = base_result.reshape(-1, nphi, 3).sum(1)
    base_result = torch.maximum(base_result, torch.tensor(0, device="cuda"))
    base_result = torch.where(total.unsqueeze(-1) > 0, base_result / total.unsqueeze(-1), 0)
    final_result = (1-clearcoat_weight) * (base_result * envmap_scale * fg)
    #####

    ##### Clearcoat layer
    if render_clearcoat:
        # Reload all coefficients with order 100
        clearcoat_sh_order = 100
        bsdf_coeffs, emitter_coeffs, l_coeffs = load_coeffs(
            envmap_path, envmap_proj_meth,
            clearcoat_sh_order, fast_rotation,
            clearcoat_alpha_common, "gaussian"
        )

        clearcoat_result = torch.zeros((resy, resx, 3), device="cuda:0", dtype=torch.float32)
        if fast_rotation:
            fast_dot.render_half_fast_rotation(
                onb_n, wi_world, clearcoat_alpha,
                bsdf_coeffs, emitter_coeffs, l_coeffs,
                clearcoat_result, clearcoat_sh_order
            )
        else:
            fast_dot.render_half_lookup(
                onb_n, wi_world,
                bsdf_coeffs, emitter_coeffs,
                clearcoat_result, clearcoat_sh_order
            )
        clearcoat_result = torch.maximum(clearcoat_result, torch.tensor(0, device="cuda"))
        clearcoat_result = clearcoat_result.reshape(-1, 3)

        # FG decoupling
        wo = mi.reflect(si.wi)
        fg = bsdf.eval_pdf(ctx, si, wo)[0].torch()
        final_result += clearcoat_weight * fg * clearcoat_result * envmap_scale
        
    # Put it into the film
    final_result = mi.Color3f(final_result)
    final_result = [final_result.x, final_result.y, final_result.z, mi.Float(1)]

    # Develop the film
    film = scene.sensors()[0].film()
    # Image block
    block = film.create_block()
    # Offset is the currect location of the block
    # In case of GPU, the block covers the entire image, hence offset is 0
    block.set_offset(film.crop_offset())

    ################################
    # Save image
    ################################
    block.put(pos, final_result)
    film.put_block(block)
    final_result = film.develop().numpy()

    glint_pixels = glint_pixels.cpu().numpy().astype(bool)
    final_result[~glint_pixels] = 0

    return final_result, glint_pixels, times_hist, times_prod