import mitsuba as mi
import numpy as np
import drjit as dr
import os
import time
import torch
from tqdm import trange
from utils import sh
from utils.fs import create_dir
from utils.transforms import pol2cart, cart2pol
from glob import glob
from typing import Tuple

channels2dtype = [
    mi.Float,
    mi.Vector2f,
    mi.Vector3f,
    mi.Vector4f
]

class SHBase:
    """
    This is the base class which represents a spherical function
    using spherical harmonics as a basis. Any class that inherits from this
    class should override the `eval` function which will be used for projection.
    Optionally, the class can also override the `sample` function which
    can be used for importance sampling a function while projecting it.
    The class supports 4D as well as 2D functions. In the case 4D functions,
    the SH coefficients are calculated for a bunch of different directions
    and queried at runtime using bilinear interpolation.
    """
    def __init__(
            self,
            order: int,
            ntheta: int=64,
            nphi: int=128,
            nsamples: int=int(1e6),
            force_eval: bool=False,
            rotate_on_fly: bool=False,
            project_on_fly: bool=False,
            project_method: str="mc",
            **kwargs
        ) -> None:
        """
        Initialize the class by loading the SH cofficients from disk or
        computing them and storing them. Each class inheriting form this
        class should call the parent `__init__` function after setting any
        data terms that may be required by the `path_to_order` and `glob_str` functions

        Args:
            order: The number of bands to use
            nchannels: The number of channels of the function
            ntheta: The discretization of θ parameter (only used in case of 4D functions)
            nphi: The discretization of ϕ parameter (only used in case of 4D functions)
            nsamples: Number of samples to use for calculating SH coefficients
            force_eval: Whether the coefficients should be recalculated even in the case they exist
            rotate_on_fly: Whether to rotate ZH using fast rotation at runtime
            project_on_fly: Whether to project coefficients at runtime
            project_method: Which method to use for projecting to SH coefficient, valid choices are 'mc' and 'simpsons'
        """
        self.upper_hemisphere = getattr(self, "upper_hemisphere", None)
        self.cache_dir = getattr(self, "cache_dir", None)
        self.dim = getattr(self, "dim", None)
        self.nchannels = getattr(self, "nchannels", None)
        self.zonal_only = getattr(self, "zonal_only", None)

        # Coefficients
        self.coeffs = None

        assert self.dim == 2 or self.dim == 4, "Only 2D or 4D functions are supported"
        assert self.nchannels < 5, "Only functions with at max 4 channels are supported"
        assert not self.cache_dir is None, "`cache_dir` variable should be set by child class"
        assert not self.zonal_only is None, "`zonal_only` field should be set by child class"
        assert not self.upper_hemisphere is None, "The child should define the domain of function (hemisphere / full sphere)"
        assert project_method in ["mc", "simpsons"], "Valid options are 'mc' or 'simpsons'"

        self.order = order
        self.max_idx = (order+1)**2
        self.ntheta = ntheta
        self.nphi = nphi
        self.nsamples = nsamples
        self.rotate_on_fly = rotate_on_fly
        self.project_on_fly = project_on_fly
        self.project_method = project_method
        self.theta_res = int(np.sqrt(nsamples / 2))
        if self.theta_res % 2:
            self.theta_res += 1 # For Simpson's rule the resolution has to be even
        self.phi_res = self.theta_res*2

        self.dtype = channels2dtype[self.nchannels-1]

        # Load precomputed texture
        lcoeffs = np.load(os.path.join("data", "sh", "lcoeffs.npy"))
        lcoeffs = lcoeffs[:,:self.max_idx,None]  # Only keep values that are needed
        assert lcoeffs.shape[-2] == self.max_idx, "`lcoeffs.npy` doesn't have enough orders"
        self.lcoeffs = mi.Texture2f(mi.TensorXf(lcoeffs))

        # sin = np.load(os.path.join("data", "sh", "sin.npy"))[...,None]
        # self.sin = mi.Texture1f(mi.TensorXf(sin))
        self.sin = None

        # cos = np.load(os.path.join("data", "sh", "cos.npy"))[...,None]
        # self.cos = mi.Texture1f(mi.TensorXf(cos))
        self.cos = None

        if not project_on_fly:
            create_dir(self.cache_dir)
            cache_name = None
            if not force_eval:
                # Search for the cache file
                globstr = self.glob_str()
                candids = glob(globstr, root_dir=self.cache_dir)
                candid_orders = list(map(self.path_to_order, candids))
                for i in range(len(candids)):
                    if candid_orders[i] >= self.order:
                        cache_name = candids[i]
                        break

            if cache_name is None:
                mi.Log(mi.LogLevel.Info, "Couldn't find coefficients in cache, computing")
                # No cache found or 'force_eval' requested
                cache_name = self.glob_str()
                cache_name = cache_name.replace("*", str(order))

                # Project
                t0 = time.time()
                coeffs = self.project()
                np.save(os.path.join(self.cache_dir, cache_name), coeffs)
                mi.Log(mi.LogLevel.Info, f"Computation complete, time taken: {time.time()-t0:.2f}s")

            self.cache_name = cache_name
        self.loaded = False

    def load(self):
        if self.project_on_fly:
            return

        mi.Log(mi.LogLevel.Info, f"Using cached coefficients from: {os.path.join(self.cache_dir, self.cache_name)}")
        # Load the coefficients
        coeffs = np.load(os.path.join(self.cache_dir, self.cache_name))

        # Only keep the necessary number of coeffs (till max_idx)
        coeffs = coeffs[...,:self.max_idx,:]
        if self.dim == 2 or self.rotate_on_fly:
            # Store 2D function in an array of Floats / Vector3f
            if self.nchannels == 1:
                self.coeffs = mi.Float(coeffs[...,0])
            else:
                self.coeffs = self.dtype(coeffs)
        else:
            # Store 4D functions in a 3D texture
            self.coeffs = mi.Texture3f(mi.TensorXf(coeffs), wrap_mode=dr.WrapMode.Repeat)
        
        self.loaded = True

    def cache_path(self):
        return os.path.join(self.cache_dir, self.cache_name)

    def query(self, sh_idx: mi.UInt, w: mi.Vector3f=None, active: bool=True, **kwargs) -> mi.Float|mi.Vector3f:
        """
        Query the 'i'th SH coefficient. In case of 4D function, a direction
        vector will also be provided. 

        Args:
            sh_idx: The index to query
            w: The direction vector to reduce 4D function to 2D
        Returns:
            coeffs: The corresponding SH coefficients
        """
        if self.project_on_fly:
            idx = mi.UInt(0)
            coeff = self.dtype(0)
            rng = mi.PCG32()
            loop = mi.Loop("Project SH", lambda: (idx, coeff, rng))
            while loop(idx < self.nsamples):
                w_query, pdf = self.sample(mi.Point2f(rng.next_float32(), rng.next_float32()), w, **kwargs)
                l, m = sh.get_lm(sh_idx)
                res = self.eval(w_query, w, **kwargs)
                res = dr.select(pdf > 1e-6, res / pdf, 0)
                coeff += (res * sh.eval_sh(w_query, l, m, self.lcoeffs, self.max_idx)) / self.nsamples
                idx += 1

            return coeff

        if not self.loaded:
            raise RuntimeError("Coefficients not loaded yet, call 'load()' function")

        if self.dim == 2:
            return dr.gather(type(self.coeffs), self.coeffs, sh_idx, active)

        if self.rotate_on_fly:
            l, m = sh.get_lm(sh_idx)
            coeff = dr.gather(type(self.coeffs), self.coeffs, l*l+l, active)
            nl = dr.sqrt(dr.four_pi / (2*l+1))
            return nl * sh.eval_sh(w, l, m, self.lcoeffs, self.max_idx) * coeff

        # 4D function
        query_w = cart2pol(w)
        query_w.y = dr.select(query_w.y < 0, query_w.y + dr.two_pi, query_w.y)
        query_w /= mi.Point2f(dr.pi, dr.two_pi)
        query_z = (mi.Float(sh_idx) + 0.5) * dr.rcp(mi.Float(self.max_idx))

        coeff = self.coeffs.eval(mi.Point3f(query_z, query_w.y, query_w.x), active)
        if self.nchannels == 1:
            result = coeff[0]
        else:
            result = self.dtype(coeff)

        return dr.select(sh_idx < self.max_idx, result, 0)
        
    def _project_2d_mc(self, w2: mi.Vector3f, coeffs: torch.Tensor=None, d: mi.Vector3f=None, p: mi.Float=None) -> torch.Tensor:
        """
        Projects a 2D function to SH basis using Monte-Carlo.
        In case of a 4D function, the 'w2' argument reduces the 4D function to a 2D function.

        Args:
            w2: The vector to reduce 4D function to 2D
            coeffs: Buffer to store the coefficients (useful when the function has to be called many times)
        Returns:
            coeffs: A numpy array of projected coefficients
        """
        # In the batched mode, we project a 4D non spherically symmetric
        # functions parallelly
        batched = not w2 is None and dr.width(w2) > 1

        if not batched:
            rng = mi.PCG32(size=self.nsamples)

            # Sample directions
            eta = mi.Point2f(rng.next_float32(), rng.next_float32())
            d, p = self.sample(eta) # TODO: Support functions that need more random inputs

        # Expand w2 and d to same dimension
        if batched:
            nb = dr.width(w2)
            rng = mi.PCG32(size=self.nsamples)
            eta = mi.Point2f(rng.next_float32(), rng.next_float32())
            # TODO: Make this more elegant
            # The sample method does the tile and repeat
            # Currently this is only working for half-space envmaps
            d, p = self.sample(eta, w2)
            w2 = dr.repeat(w2, self.nsamples)   # 0,1,2 -> 0,0,1,1,2,2
            if coeffs is None:
                coeffs = torch.zeros((nb, self.max_idx, self.nchannels), device="cuda:0")
        else:
            if coeffs is None:
                coeffs = torch.zeros((self.max_idx, self.nchannels), device="cuda:0")

        # Evaluate function at given points
        vals = self.eval(d, w2)
        vals = dr.select(p > 1e-6, vals / p, 0)
        dr.eval(vals)

        # In the loop below, assigning the final coefficients to numpy causes
        # everything to be evaluated and no recording is done over the loop
        for i in trange(self.max_idx, leave=False):
            l, m = sh.get_lm(i)
            l = dr.opaque(mi.Int, l[0], shape=1)    # We don't want to recompile the kernel
            m = dr.opaque(mi.Int, m[0], shape=1)    # for different `l` and `m` values
            if self.zonal_only and m[0]:
                continue
            sh_vals = sh.eval_sh(d, l, m, self.lcoeffs, self.max_idx, self.sin, self.cos)
            if dr.any_nested(dr.or_(dr.isnan(sh_vals), dr.isinf(sh_vals))):
                mi.Log(mi.LogLevel.Warn, "Found NaN/Inf values")
            if batched:
                # Deal with batched case differently
                if self.nchannels > 1:
                    for c in range(self.nchannels):
                        coeff = (vals[c] * sh_vals).torch()
                        coeff = coeff.reshape(nb, self.nsamples)
                        coeff = torch.mean(coeff, dim=1)
                        coeffs[:,i,c] = coeff
                else:
                    coeff = (vals * sh_vals).torch()
                    coeff = coeff.reshape(nb, self.nsamples)
                    coeff = torch.mean(coeff, dim=1)
                    coeffs[:,i] = coeff
            else:
                if self.nchannels > 1:
                    for c in range(self.nchannels):
                        coeffs[i][c] = dr.mean(vals[c] * sh_vals)[0]
                else:
                    coeffs[i] = dr.mean(vals * sh_vals)[0]

        return coeffs

    def _project_2d_simpsons(self, w2: mi.Vector3f, coeffs: torch.Tensor=None) -> torch.Tensor:
        """
        Projects a 2D function to SH basis using the Simpsons rule for Quadrature.
        In case of a 4D function, the 'w2' argument reduces the 4D function to a 2D function.

        Args:
            w2: The vector to reduce 4D function to 2D
            coeffs: Buffer to store the coefficients (useful when the function has to be called many times)
        Returns:
            coeffs: A numpy array of projected coefficients
        """

        # In the batched mode, we project a 4D non spherically symmetric
        # functions parallelly
        batched = not w2 is None and dr.width(w2) > 1

        # Get directions on the sphere
        thetas = dr.linspace(mi.Float, 0, dr.pi, self.theta_res+1)
        phis = dr.linspace(mi.Float, 0, dr.two_pi, self.phi_res+1)
        w = mi.Point2f(dr.meshgrid(thetas, phis))
        d = pol2cart(w)

        # Get integration weights
        theta_weights = dr.zeros(mi.Float, self.theta_res+1)
        dr.scatter(theta_weights, 2, dr.arange(mi.Float, 0, self.theta_res, 2))
        dr.scatter(theta_weights, 4, dr.arange(mi.Float, 1, self.theta_res, 2))
        dr.scatter(theta_weights, 1, mi.UInt([0, self.theta_res]))

        phi_weights = dr.zeros(mi.Float, self.phi_res+1)
        dr.scatter(phi_weights, 2, dr.arange(mi.Float, 0, self.phi_res, 2))
        dr.scatter(phi_weights, 4, dr.arange(mi.Float, 1, self.phi_res, 2))
        dr.scatter(phi_weights, 1, mi.UInt([0, self.phi_res]))
        weights = mi.Point2f(dr.meshgrid(theta_weights, phi_weights))
        weights = weights.x * weights.y * dr.sin(w.x)

        nb = dr.width(w2)
        nsamples = dr.width(weights)
        if batched:
            weights = dr.tile(weights, nb)
            d = dr.tile(d, nb)
            w2 = dr.repeat(w2, nsamples)
            if coeffs is None:
                coeffs = torch.zeros((nb, self.max_idx, self.nchannels), device="cuda:0")
        else:
            if coeffs is None:
                coeffs = torch.zeros((self.max_idx, self.nchannels), device="cuda:0")

        # Evaluate the function at the given points multiplied by the weights
        vals = self.eval(d, w2) * weights
        dr.eval(vals)

        max_idx = dr.opaque(mi.Int, self.max_idx, 1)
        for i in trange(self.max_idx, leave=False):
            l, m = sh.get_lm(i)
            l = dr.opaque(mi.Int, l[0], shape=1)    # We don't want to recompile the kernel
            m = dr.opaque(mi.Int, m[0], shape=1)    # for different `l` and `m` values
            if self.zonal_only and m[0]:
                continue
            sh_vals = sh.eval_sh(d, l, m, self.lcoeffs, max_idx)
            if dr.any_nested(dr.or_(dr.isnan(sh_vals), dr.isinf(sh_vals))):
                mi.Log(mi.LogLevel.Warn, "Found NaN/Inf values")

            if batched:
                # Deal with batched case differently
                if self.nchannels > 1:
                    for c in range(self.nchannels):
                        coeff = (vals[c] * sh_vals).torch()
                        coeff = coeff.reshape((nb, nsamples))
                        coeff = torch.sum(coeff, dim=1)
                        coeffs[:,i,c] = coeff
                else:
                    coeff = (vals * sh_vals).torch()
                    coeff = coeff.reshape(nb, nsamples)
                    coeff = torch.sum(coeff, dim=1)
                    coeffs[:,i] = coeff
            else:
                if self.nchannels > 1:
                    for c in range(self.nchannels):
                        coeffs[i][c] = torch.sum((vals[c] * sh_vals).torch())
                else:
                    coeffs[i] = torch.sum((vals * sh_vals).torch())

        coeffs *= (dr.pi / self.theta_res) * (dr.two_pi / self.phi_res) * dr.rcp(9)

        return coeffs

    def project(self, theta_delta: mi.Float=0.0, phi_delta: mi.Float=0.0, batch_size:int=256) -> np.ndarray:
        """
        Project a given function to SH coefficients. This function calls the `eval` and
        `sample` function and uses Monte-Carlo integration to obtain the SH coefficients.

        Returns
            coeffs: A numpy array consisting of coefficients
            batch_size: Only used while projecting non-symmetric 4D functions
        """
        if self.dim == 2 or self.rotate_on_fly:
            if self.project_method == "mc":
                return self._project_2d_mc(None).cpu().numpy()
            else:
                return self._project_2d_simpsons(None).cpu().numpy()

        # Generate directions to rotate to
        thetas = (dr.arange(mi.Float, 0, self.ntheta) + 0.5) * dr.rcp(self.ntheta)
        thetas *= dr.pi
        thetas += theta_delta
        phis = (dr.arange(mi.Float, 0, self.nphi) + 0.5) * dr.rcp(self.nphi)
        phis *= dr.two_pi
        phis += phi_delta
        w = mi.Point2f(dr.meshgrid(thetas, phis, indexing='ij'))
        w = pol2cart(w)

        coeffs = torch.zeros((self.ntheta*self.nphi, self.max_idx, self.nchannels), device="cuda:0")
        if self.zonal_only:
            # Use fast rotations
            coeffs_can = self._project_2d_mc(None).cpu().numpy()

            # Rotate coefficients
            coeffs = sh.rotate_zh(coeffs_can, w, self.lcoeffs, self.max_idx)
        else:
            # Split rotation into multiple batches
            w = w.numpy()

            # Sample directions, use the same samples for all input directions
            for i in trange(int(np.ceil(w.shape[0] / batch_size)), leave=False):
                start = i*batch_size
                end = min(w.shape[0], start + batch_size)
                if self.project_method == "mc":
                    self._project_2d_mc(mi.Vector3f(w[start:end]), coeffs[start:end])
                else:
                    self._project_2d_simpsons(mi.Vector3f(w[start:end]), coeffs[start:end])

            coeffs = coeffs.cpu().numpy()

        coeffs = coeffs.reshape(self.ntheta, self.nphi, -1, self.nchannels)
        return coeffs.astype(np.float32)

    def unproject(self, ntheta: int=1024, nphi: int=2048, w: mi.Vector3f=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unproject a given function. This is a convinience function that can be used to
        debug and verify that SH projection is working correctly.

        Args:
            ntheta: The discretization for θ
            nphi: The discretization for ϕ
            w: Only used for 4D functions, the direction to be used to collapse

        Returns:
            func: The unprojected function as a numpy array
        """
        # Generate directions to query
        thetas = dr.linspace(mi.Float32, 0.5, ntheta-0.5, ntheta) / ntheta
        phis = dr.linspace(mi.Float32, 0.5, nphi-0.5, nphi) / nphi
        d = mi.Point2f(dr.meshgrid(thetas, phis, indexing='ij'))
        d *= mi.Point2f(dr.pi, dr.two_pi)
        d = self.pol2cart(d)

        result = torch.zeros((ntheta*nphi, self.nchannels)).to("cuda")
        for i in trange(self.max_idx, desc="Unprojecting function", leave=False):
            l, m = sh.get_lm(i)
            l = dr.opaque(mi.Int, l[0], shape=1)    # We don't want to recompile the kernel
            m = dr.opaque(mi.Int, m[0], shape=1)    # for different `l` and `m` values
            sh_vals = sh.eval_sh(d, l, m, self.lcoeffs, self.max_idx)
            coeff = self.query(i, w)
            if self.nchannels == 1:
                result += (coeff * sh_vals).torch()[...,None]
            else:
                result += (coeff * sh_vals).torch()

        # Evaluate ground truth
        result_gt = self.eval(d, w).numpy()

        return result.cpu().numpy().reshape(ntheta, nphi, -1), result_gt.reshape(ntheta, nphi, -1)

    def sample(self, eta: mi.Point2f, w: mi.Vector3f=None, **kwargs) -> Tuple[mi.Vector3f, mi.Float]:
        """
        Importance sampling procedure for the function, by default uniform [hemi]sphere
        sampling is used, but a child class may override this function to use a better
        importance sampling routine.

        Args:
            eta: The sample to transform
            w: The direction to use to collapse 4D to 2D
        Returns:
            d: The sampled direction
            pdf: The PDF of sampled direction
        """
        if self.upper_hemisphere:
            return mi.warp.square_to_uniform_hemisphere(eta), mi.Float(dr.inv_two_pi)
        else:
            return mi.warp.square_to_uniform_sphere(eta), mi.Float(dr.inv_four_pi)

    def eval(self, w1: mi.Vector3f, w2: mi.Vector3f=None, **kwargs) -> mi.Float|mi.Vector3f:
        """
        This function has to be overriden by each child class and will be
        called by the 'project' function. In case of 2D functions, only
        a single direction will be given as input, in case of 4D function,
        two direction vectors will be given as input

        Args:
            w1: The first direction vector (random samples)
            w2: The second direction vector (will be `None` when dim == 2)

        Returns:
            res: A Float or Vector3f based upon number of channels for the function
        """
        raise NotImplementedError

    def l2_error(self, nsamples: int, w: mi.Vector3f=None) -> mi.Float:
        """
        Calculate the mean L2 error of the reprojected function and
        the original. In case of 4D function, a direction vector
        is required to collapse the function to 2D.

        Args:
            nsamples: The number of samples to consider

        Returns:
            error: The mean L2 error
        """
        # Generate samples
        rng = mi.PCG32(nsamples)
        samples = mi.warp.square_to_uniform_sphere(mi.Point2f(rng.next_float32(), rng.next_float32()))

        # Evaluate ground truth
        gt = self.eval(samples, w).torch()

        reprojected = torch.zeros((nsamples, self.nchannels)).to("cuda")
        for i in trange(self.max_idx, desc="Unprojecting function", leave=False):
            l, m = sh.get_lm(i)
            l = dr.opaque(mi.Int, l[0], shape=1)    # We don't want to recompile the kernel
            m = dr.opaque(mi.Int, m[0], shape=1)    # for different `l` and `m` values
            sh_vals = sh.eval_sh(samples, l, m, self.lcoeffs, self.max_idx)
            coeff = self.query(i, w)
            if self.nchannels == 1:
                reprojected += (coeff * sh_vals).torch()[...,None]
            else:
                reprojected += (coeff * sh_vals).torch()
        
        # We are using MC to evaluate the integral of L2 error over sphere.
        # However, since we want to calculate the mean we don't need to multiply
        # by the volume of domain.
        reprojected = torch.squeeze(reprojected, -1)
        return ((gt - reprojected)**2).mean()

    def pol2cart(self, pol: mi.Point2f):
        return pol2cart(pol)


    def path_to_order(self, path: str) -> int:
        """
        Returns max order of SH coefficients that indicated by the filename.
        Child classes can override this method if they use a custom filename format
        which includes more information about the coefficients

        Args:
            path: The path of the SH coefficients
        Returns:
            order: The max order of coefficients stored in the file
        """
        filename = os.path.basename(path)
        filename = filename.replace(".npy", "")      # Remove npy
        filename = filename.split("_")[-1]

        return int(filename)

    def glob_str(self) -> str:
        """
        Returns the glob string that should be used to search for coefficients
        in the cache directory. Child classes can override this if they use
        a custom filename format which includes more information about the coefficients
        By default, the format used is 'coeffs_{order}.npy' for 2D functions and
        'coeffs_{ntheta}_{nphi}_{order}.npy' for 4D functions.
        
        Returns:
            glob: The glob search string
        """
        if self.dim == 2 or self.rotate_on_fly:
            return f"coeffs_{self.project_method}_*.npy"
        else:
            return f"coeffs_{self.project_method}_{self.ntheta}_{self.nphi}_*.npy"