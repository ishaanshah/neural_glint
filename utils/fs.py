import os
import shutil
import numpy as np
import mitsuba as mi
import torch
import drjit as dr
from utils.agxc import applyAgX
from typing import Tuple

def linear_to_srgb(image: mi.TensorXf|np.ndarray, gamma: float=2.2) -> mi.TensorXf|np.ndarray:
    if isinstance(image, np.ndarray):
        image = np.maximum(image, 0)
        return np.clip(image ** (1 / gamma), 0, 1)
    else:
        image = dr.maximum(image, 0)
        return dr.clamp(image ** (1 / gamma), 0, 1)

def get_name(path: str) -> Tuple[str, str]:
    filename = os.path.basename(path)
    wo_ext = ".".join(filename.split(".")[:-1])
    return filename, wo_ext

def create_dir(path: os.PathLike, del_old: bool=False):
    if del_old and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=not del_old)

def split_path(path: os.PathLike) -> Tuple[str, str, str]:
    """
    Splits the given path into 3 components, the directory, file name and extension
    """
    ext = path.split(".")[-1]
    name = ".".join(os.path.basename(path).split(".")[:-1])
    dirname = os.path.dirname(path)

    return dirname, name, ext

def write_image(path: os.PathLike, img: np.ndarray|mi.TensorXf|torch.Tensor, tonemap: str="agx", ev: float=0.0):
    assert tonemap in ["agx", "srgb", "none"], "'tonemap' should be one of 'agx', 'srgb' or 'none'"
    if type(img) == mi.TensorXf:
        img = img.numpy()
    elif type(img) == torch.Tensor:
        img = img.cpu().numpy()

    if ".hdr" in str(path) or ".exr" in str(path):
        mi.util.write_bitmap(str(path), mi.Bitmap(img))
    else:
        # Convert NaNs to 0
        img = np.nan_to_num(img, 0)
        img *= np.power(2, ev)
        if tonemap == "agx":
            img = applyAgX(img)
        elif tonemap == "srgb":
            img = linear_to_srgb(img)
        else:
            img = np.clip(img, 0, 1)
        
        bmp = mi.Bitmap(img)
        bmp.set_srgb_gamma(True)

        # Special case because write_bitmap removes the alpha term
        pixel_format = mi.Bitmap.PixelFormat.RGBA if img.shape[-1] == 4 else mi.Bitmap.PixelFormat.RGB
        bmp = bmp.convert(pixel_format, mi.Struct.Type.UInt8, True)
        bmp.write(str(path))