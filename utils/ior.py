import mitsuba as mi
from typing import Tuple

def ior_from_file(filename: str) -> mi.Texture:
    is_spectral = "spectral" in mi.variant()
    is_mono = "mono" in mi.variant()
    assert not is_spectral, "Spectral mode is not supported"
    assert not is_mono, "Mono mode is not supported"

    wavelengths, values = mi.spectrum_from_file(filename)
    unit_conversion = mi.MI_CIE_Y_NORMALIZATION
    for k in range(len(wavelengths)):
        values[k] *= unit_conversion

    color = mi.spectrum_list_to_srgb(wavelengths, values, False, False)

    return color

def complex_ior_from_file(material: str) -> Tuple[mi.Texture, mi.Texture]:
    eta = ior_from_file("data/ior/" + material + ".eta.spd")
    k = ior_from_file("data/ior/" + material + ".k.spd")

    eta = mi.load_dict({
        "type": "d65",
        "color": eta
    })
    k = mi.load_dict({
        "type": "d65",
        "color": k 
    })
    return eta, k