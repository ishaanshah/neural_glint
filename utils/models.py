import tinycudann as tcnn
import torch
import numpy as np
import os
from argparse import ArgumentParser
from utils.sat import TorchSAT, open_sat
from natsort import natsorted
from torch import Tensor, nn
from typing import List
from glob import glob

def TriangleWave(period: float):
    def func(x: Tensor) -> Tensor:
        return 2 * torch.abs(x / period - torch.floor(x / period + 0.5))

    return func

class TorchWrapper(nn.Module):
    def __init__(self, sat_path: str, model_dir: str, model_class: nn.Module, full_hist: bool=True, use_sat: bool=False):
        super().__init__()
        self.full_hist = full_hist
        self.use_sat = use_sat
        if full_hist:
            model_data = torch.load(model_dir)
            self.model = model_class(**model_data["config"])
            self.model.load_state_dict(model_data["weights"])
            self.model.eval()
        else:
            sat, _, __ = open_sat(sat_path, "r")
            self.nphi = sat.shape[-1]
            self.ntheta = sat.shape[-2]
            if use_sat:
                sat = torch.tensor(sat.sum(-1).astype(np.int32), device="cuda")
                self.sat = TorchSAT(sat)
            else:
                model_path = os.path.join(model_dir, "model_theta.pt")
                self.sat = self.load_model(model_class, model_path)

            model_paths = natsorted(glob(f"{model_dir}/model_*.pt"))
            models = []
            for model_path in model_paths:
                if "theta" in model_path:
                    continue
                models.append(self.load_model(model_class, model_path))

            self.models = nn.ModuleList(models)

    def load_model(self, model_class: nn.Module, path: str) ->  nn.Module:
        model_data = torch.load(path)
        model = model_class(**model_data["config"])
        model.load_state_dict(model_data["weights"])
        model = model.half()
        model.eval()

        return model

    @torch.no_grad()
    def forward(self, query: Tensor, theta_idx: int=-1, _: int=-1) -> Tensor:
        duv = torch.abs(query[:,2:])

        # Wrap the UV coordinate
        cuv = query[:,:2]
        cuv -= torch.floor(cuv)

        query = torch.cat([cuv, duv], dim=-1)

        # TODO: Seperate the histogram combination from this module
        if self.full_hist:
            result = self.model(query)
        else:
            if theta_idx < 0:
                sat = self.sat(query)
                if self.use_sat:
                    sat_sum = sat.sum(-1, keepdim=True)
                    sat = torch.where(sat_sum > 0, sat / sat_sum, 0)
            else:
                if self.use_sat:
                    sat = self.sat(query)
                    sat = sat[:, theta_idx // self.nphi]  / sat.sum(-1)
                    sat = sat.unsqueeze(-1)
                else:
                    sat = self.sat(query)[:,theta_idx // self.nphi].unsqueeze(-1)
            if theta_idx < 0:
                result = torch.zeros((query.shape[0], self.nphi*self.ntheta), device=query.device)
                for theta_idx in range(self.ntheta):
                    tmp = self.models[theta_idx](query)
                    tmp = tmp * sat[:,theta_idx:theta_idx+1]
                    result[...,theta_idx*self.nphi:(theta_idx+1)*self.nphi] = tmp
            else:
                result = self.models[theta_idx // self.nphi](query)
                result *= sat

        return result

class MultiScaleEncoder(nn.Module):
    def __init__(self, res: int, nfeatures: int=32, start_stride: int=8):
        super().__init__()

        nlevels = int(np.ceil(np.log2(res // start_stride)+1))
        strides = []
        offsets = [0]
        encoders = []
        level_ress = []
        for level in range(nlevels):
            stride = start_stride * 2**level
            level_res = res // stride
            level_ress.append(level_res)
            encoder = torch.rand((level_res, level_res, nfeatures), dtype=torch.float32) * (2*1e-4) - 1e-4
            encoder = encoder.reshape(-1, nfeatures)
            encoders.append(encoder)
            strides.append(stride)
            offsets.append(encoder.shape[0] + offsets[-1])

        self.register_buffer('strides', torch.tensor(strides), persistent=True)
        self.register_buffer('offsets', torch.tensor(offsets), persistent=True)
        self.register_buffer('level_res', torch.tensor(level_ress), persistent=True)
        self.min_stride = start_stride
        self.max_stride = res
        self.res = res
        self.encoder = nn.Parameter(torch.cat(encoders, 0))

        self.n_output_dims = nfeatures

    def forward(self, x: Tensor) -> Tensor:
        cuv = x[:,:2]
        duv = x[:,2:]

        # Get the levels to interpolate from
        fp_size = torch.clamp(self.res * duv[:,0], self.min_stride, self.max_stride)
        level_idx_hi = torch.searchsorted(self.strides, fp_size, right=True)
        level_idx_hi = torch.clamp(level_idx_hi, 0, len(self.strides)-1)
        level_idx_lo = level_idx_hi - 1
        level_idx_lo = torch.clamp(level_idx_lo, 0, len(self.strides)-1)

        # Get offset of the levels
        offset_hi = self.offsets[level_idx_hi]
        offset_lo = self.offsets[level_idx_lo]

        # Get resolution of the levels
        level_res_hi = self.level_res[level_idx_hi]
        level_res_lo = self.level_res[level_idx_lo]

        # Get footrpint sizes of the levels
        fp_size_hi = self.strides[level_idx_hi]
        fp_size_lo = self.strides[level_idx_lo]

        # Get latent vectors from these levels
        lvec_hi = self.interp_level(cuv, level_res_hi, offset_hi)
        lvec_lo = self.interp_level(cuv, level_res_lo, offset_lo)

        # Interpolate latent vectors across levels
        w = ((fp_size - fp_size_lo) / (fp_size_hi - fp_size_lo)).unsqueeze(-1)
        lvec = lvec_lo * (1-w) + lvec_hi * w
        lvec = torch.where((fp_size_hi == fp_size_lo).unsqueeze(-1), lvec_lo, lvec)

        return lvec

    def interp_level(self, cuv: Tensor, level_res: Tensor, level_offset: Tensor) -> Tensor:
        pos_f = cuv * level_res.unsqueeze(-1) - 0.5
        pos_i = torch.floor(pos_f).int()

        w = pos_f - pos_i
        result = 0
        pos_i_w = torch.zeros((cuv.shape[0], 2), device=cuv.device)
        for i in range(4):
            pos_i_w[:,0] = (pos_i[:,0] + (i & 1)) % level_res
            pos_i_w[:,1] = (pos_i[:,1] + (i >> 1)) % level_res

            w_i = 1
            w_i *= w[:,0] if (i & 1) else (1 - w[:,0])
            w_i *= w[:,1] if (i >> 1) else (1 - w[:,1])

            lin_idx = (pos_i_w[:,0] * level_res + pos_i_w[:,1] + level_offset).int()
            result += self.encoder[lin_idx] * w_i.unsqueeze(-1)

        return result

    @staticmethod
    def parse_args(args: List[str]) -> dict:
        parser = ArgumentParser()
        parser.add_argument("--nfeatures", default=32, type=int)

        return vars(parser.parse_args(args))

class PredHist(nn.Module):
    """
    This class consists of the model which predicts the histogram. It takes as input
    the class of the encoder module and it's initialization parameters.
    """
    def __init__(self, depth: int, width: int, output_dim: int, encoder_config: dict):
        super().__init__()

        self.encoder = MultiScaleEncoder(**encoder_config)

        # Initialize the neural network
        assert depth > 0, "Depth should be atleast 0"
        if width in [16, 32, 64, 128]:
            mlp_type = "FullyFusedMLP"
        else:
            mlp_type = "CutlassMLP"

        # TCNN MLP
        network_config = {
            "otype": mlp_type,
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": width,
            "n_hidden_layers": depth+1
        }
        self.layers = tcnn.Network(self.encoder.n_output_dims, output_dim, network_config)

        # PyTorch MLP
        # layers = [
        #     nn.Linear(self.encoder.n_output_dims, width, bias=False),
        #     nn.ReLU()
        # ]
        # for _ in range(depth):
        #     layers.append(nn.Linear(width, width, bias=False))
        #     layers.append(nn.ReLU())
        # layers.append(nn.Linear(width, output_dim, bias=False))
        # self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Takes a tensor where the first two fields denote the center of pixel footprint
        and the second two denote the extents of the pixel footprint.
        """
        encoding = self.encoder(x)
        y = self.layers(encoding)

        y = TriangleWave(2)(y) + 1e-6
        y_sum = torch.sum(y, -1).unsqueeze(-1)
        y = y / y_sum

        return y