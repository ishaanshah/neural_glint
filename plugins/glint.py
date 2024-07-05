import mitsuba as mi
import drjit as dr
from utils.ior import complex_ior_from_file

class GlintDummy(mi.BSDF):
    def __init__(self, props: mi.Properties):
        super().__init__(props)

        self.glint_idx = props.get("glint_idx", mi.Texture.D65(1))
        self.alpha: mi.Texture = props.get("alpha", mi.Texture.D65(0.01))
        self.material = props.get("material", "none")
        if self.material == "none" or props.has_property("eta"):
            self.eta: mi.Texture = props.get("eta", mi.Texture.D65(0.01))
            self.k: mi.Texture = props.get("k", mi.Texture.D65(1))
        else:
            self.eta, self.k = complex_ior_from_file(self.material)

        self.alpha_mul: mi.Texture = mi.Texture.D65(props.get("alpha_mul", 1.0))
        self.clearcoat_alpha: mi.Texture = mi.Texture.D65(props.get("clearcoat_alpha", 0.05))
        self.specular_reflectance: mi.Texture = props.get("specular_reflectance", mi.Texture.D65(1))

    def traverse(self, callback: mi.TraversalCallback):
        callback.put_object("alpha", self.alpha, mi.ParamFlags.Differentiable)
        callback.put_object("alpha_mul", self.alpha_mul, mi.ParamFlags.Differentiable)
        callback.put_object("glint_idx", self.glint_idx, mi.ParamFlags.Differentiable)
        callback.put_object("clearcoat_alpha", self.clearcoat_alpha, mi.ParamFlags.Differentiable)
    
    # This function returns the FG term for the base layer
    def eval(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True):
        eta = self.eta.eval(si, active)
        k = self.k.eval(si, active)
        alpha = self.alpha.eval_1(si, active)
        alpha_mul = self.alpha_mul.eval_1(si, active)
        alpha *= alpha_mul
        alpha = dr.clamp(alpha, 0.01, 0.99)

        # Find half vector
        m = dr.normalize(si.wi + wo)

        # Find fresnel term
        f = mi.Color3f(0)
        for i in range(3):
            f[i] = mi.fresnel_conductor(dr.dot(si.wi, m), mi.Complex2f(eta[i], k[i]))

        # Find shadowing
        ndf = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, alpha, sample_visible=False)
        g = ndf.G(si.wi, wo, m)

        # Specular reflectance
        spec = self.specular_reflectance.eval(si, active)
        return spec * f * g / (4 * si.wi.z)

    # This function returns the FG term for clearcoat layer
    def eval_pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: bool = True):
        clearcoat_alpha = self.clearcoat_alpha.eval_1(si, active)
        alpha = dr.clamp(clearcoat_alpha, 0.01, 0.99)

        # Find half vector
        m = dr.normalize(si.wi + wo)

        # Find fresnel term
        f = mi.Color3f(0)
        for i in range(3):
            f[i] = mi.fresnel_conductor(dr.dot(si.wi, m), mi.Complex2f(1.5, 1))

        # Find shadowing
        ndf = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, alpha, sample_visible=False)
        g = ndf.G(si.wi, wo, m)

        return f * g / (4 * si.wi.z), mi.Float(1)

    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, sample1: float, sample2: mi.Point2f, active: bool = True):
        bs = dr.zeros(mi.BSDFSample3f)
        return bs, mi.Color3f(1)