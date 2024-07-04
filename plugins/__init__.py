import mitsuba as mi
from plugins.glint import GlintDummy

# BSDFs
mi.register_bsdf("glint_dummy", lambda props: GlintDummy(props))