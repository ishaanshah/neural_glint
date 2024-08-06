import numpy as np
import os
import torch
from sklearn.cluster import KMeans
from typing import Tuple
from tqdm import trange

def open_sat(sat_dir: str, mmap_mode: str=None) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Helper function which opens the SAT, bin_center and bin_edges

    Returns: (sat, bin_centers, bin_edges)
    """
    sat = np.load(os.path.join(sat_dir, "sat.npy"), mmap_mode=mmap_mode)
    bin_centers = np.load(os.path.join(sat_dir, "bin_centers.npy"))
    bin_edges = np.load(os.path.join(sat_dir, "bin_edges.npy"))

    return sat, bin_centers, bin_edges

# TODO: Convert to actual mipmap packing
def open_mipmap(mipmap_path: str) -> np.ndarray:
    """
    Opens acceleration mipmap and composes it into a stack
    of highest resolution level.

    Returns: mipmap
    """
    mipmaps = np.load(mipmap_path, allow_pickle=True)
    result = np.zeros((len(mipmaps), *mipmaps[0].shape))
    for i, mipmap in enumerate(mipmaps):
        shape = mipmap.shape[0]
        result[i][:shape,:shape] = mipmap
    return result

def calc_optimal_bins(x: np.ndarray, nbins: int,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calulates the optimal discretization for the given sum of delta distribution
    """
    model = KMeans(n_clusters=nbins, n_init='auto')
    model.fit(x[...,None])
    bin_centers = np.sort(model.cluster_centers_[...,0])
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) / 2
    bin_edges = np.concatenate([np.array([0]), bin_edges, np.array([x.max()])])

    return bin_centers, bin_edges

def compute_sat(normals: np.ndarray, ntheta: int=9, nphi: int=32, bin_edges: np.ndarray=None) -> np.ndarray:
    res_x = normals.shape[0]
    res_y = normals.shape[1]

    # Create SAT
    SAT = np.zeros((res_x, res_y, ntheta, nphi), dtype=np.uint32)

    for i in trange(res_x*res_y, desc="Calculating SAT"):
        # Get current index
        x = i // res_x
        y = i % res_x

        # Get bin idx
        n = normals[x, y]
        theta = np.clip(np.arccos(n[2]), 0, np.pi / 2) # Since normals always point up we clip to np.pi/2
        if bin_edges is None:
            theta = ((theta / (np.pi/2)) * ntheta).astype(int)
        else:
            theta = np.searchsorted(bin_edges, theta).astype(int)
            theta = np.clip(theta - 1, 0, ntheta-1)

        phi = np.arctan2(n[1], n[0])
        phi = np.where(phi > 0, phi, phi + np.pi * 2)
        phi = ((phi / (2*np.pi)) * nphi).astype(int)

        # Create entry for current texel
        to_add = np.zeros((ntheta, nphi), dtype=int)
        to_add[theta, phi % nphi] = 1

        # Update SAT
        if x == 0 and y == 0:
            SAT[x,y] = to_add
        elif y == 0:
            SAT[x,y] = SAT[x-1,y] + to_add
        elif x == 0:
            SAT[x,y] = SAT[x,y-1] + to_add
        else:
            SAT[x,y] = SAT[x-1,y] + SAT[x,y-1] - SAT[x-1,y-1] + to_add

    return SAT

class TorchSAT(torch.nn.Module):
    def __init__(self, sat: torch.Tensor):
        super().__init__()

        # Store in the format we need for grid sample
        # self.sat = sat.permute(2, 0, 1).unsqueeze(0)
        self.ntheta = 9
        self.nphi = 32
        self.duv = torch.tensor([1 / sat.shape[0], 1 / sat.shape[1]], device=sat.device)
        self.res = torch.tensor([sat.shape[0], sat.shape[1]], device=sat.device)
        self.sat = sat.reshape(-1, sat.shape[-1])

    def sample(self, x: torch.Tensor, start_idx: int=-1, len_idx: int=1) -> torch.Tensor:
        if start_idx >= 0:
            sat = self.sat[:,start_idx:start_idx+len_idx]
        else:
            sat = self.sat
        idx = self.res * x - 0.5
        idx = torch.floor(idx).int()
        zero = torch.any(torch.logical_or(idx < 0, idx >= self.res), -1).unsqueeze(-1)
        return torch.where(zero, 0, sat[idx[:,0]*self.res[1]+idx[:,1]])

    def query_sat(self, suv: torch.Tensor, euv: torch.Tensor, start_idx: int=-1, len_idx: int=1) -> torch.Tensor:
        result = self.sample(euv, start_idx, len_idx)
        result += self.sample(suv-self.duv, start_idx, len_idx)

        query = torch.stack([suv[:,0]-self.duv[0], euv[:,1]], -1)
        result -= self.sample(query, start_idx, len_idx)

        query = torch.stack([euv[:,0], suv[:,1]-self.duv[1]], -1)
        result -= self.sample(query, start_idx, len_idx)

        return result

    @torch.no_grad()
    def forward(self, x: torch.Tensor, start_idx: int=-1, len_idx: int=1) -> torch.Tensor:
        duv = x[:,2:]
        cuv = x[:,:2]
        suv = cuv - duv / 2
        euv = cuv + duv / 2

        nsuv = suv.clone()
        neuv = euv.clone()

        # Make it so that suv <= euv
        swapx = duv[:,0] < 0
        nsuv[:,0] = torch.where(swapx, euv[:,0], suv[:,0])
        neuv[:,0] = torch.where(swapx, suv[:,0], euv[:,0])

        swapy = duv[:,1] < 0
        nsuv[:,1] = torch.where(swapy, euv[:,1], suv[:,1])
        neuv[:,1] = torch.where(swapy, suv[:,1], euv[:,1])

        # Get fractional part of texture
        nsuv = nsuv - torch.floor(nsuv)
        neuv = neuv - torch.floor(neuv)

        # nsuv.x <= neuv.x & nsuv.y <= neuv.y
        cond = torch.logical_and(nsuv[:,0] <= neuv[:,0], nsuv[:,1] <= neuv[:,1]).unsqueeze(-1)
        result = torch.where(cond, self.query_sat(nsuv, neuv, start_idx, len_idx), 0)

        # Bottom right uv coordinates
        bru = torch.full((x.shape[0],), 1-self.duv[0]/2, device=self.duv.device)
        brv = torch.full((x.shape[0],), 1-self.duv[1]/2, device=self.duv.device)
        # Top left UV coordinates
        tlu = 1 - bru
        tlv = 1 - brv 

        # nsuv.x > neuv.x & nsuv.y <= neuv.y
        cond = torch.logical_and(nsuv[:,0] > neuv[:,0], nsuv[:,1] <= neuv[:,1]).unsqueeze(-1)
        result += torch.where(cond, self.query_sat(nsuv, torch.stack([bru, neuv[:,1]], -1), start_idx, len_idx), 0)
        result += torch.where(cond, self.query_sat(torch.stack([tlu, nsuv[:,1]], -1), neuv, start_idx, len_idx), 0)

        # nsuv.x <= neuv.x & nsuv.y > neuv.y
        cond = torch.logical_and(nsuv[:,0] <= neuv[:,0], nsuv[:,1] > neuv[:,1]).unsqueeze(-1)
        result += torch.where(cond, self.query_sat(nsuv, torch.stack([neuv[:,0], brv], -1), start_idx, len_idx), 0)
        result += torch.where(cond, self.query_sat(torch.stack([nsuv[:,0], tlv], -1), neuv, start_idx, len_idx), 0)
        
        # nsuv.x > neuv.x & nsuv.y > neuv.y
        cond = torch.logical_and(nsuv[:,0] > neuv[:,0], nsuv[:,1] > neuv[:,1]).unsqueeze(-1)
        result += torch.where(cond, self.query_sat(torch.stack([tlu, tlv], -1), neuv, start_idx, len_idx), 0)
        result += torch.where(cond, self.query_sat(nsuv, torch.stack([bru, brv], -1), start_idx, len_idx), 0)
        result += torch.where(cond, self.query_sat(torch.stack([nsuv[:,0], tlv], -1), torch.stack([bru, neuv[:,1]], -1), start_idx, len_idx), 0)
        result += torch.where(cond, self.query_sat(torch.stack([tlu, nsuv[:,1]], -1), torch.stack([neuv[:,0], brv], -1), start_idx, len_idx), 0)

        return result