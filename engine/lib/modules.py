# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

def embed_sincos(x: Tensor, expand: int, min_freq: float, max_freq: float):
  # x: (batch, channel), -1 ~ 1
  assert x.ndim == 2
  assert expand % 2 == 0
  device, dtype = x.device, x.dtype
  freq = torch.arange(0, expand // 2, device=device, dtype=dtype) / (expand // 2 - 1)
  freq = freq * (math.log(max_freq) - math.log(min_freq)) + math.log(min_freq)  # min_freq ~ max_freq
  freq = math.pi * torch.exp(freq)
  enc = rearrange(x, 'b c -> b c 1') * rearrange(freq, 'c -> 1 1 c')
  enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)
  enc = rearrange(enc, 'b c1 c2 -> b (c1 c2)')
  return enc

def embed_coordinates(coords: Tensor, expand: int, max_freq: float):
  # return embed_sincos(coords, expand, 0.5, max_freq)
  # min_freq=0.5 で多分十分だけど、ノイズ付加で外に広がるかもしれないので余裕をもたせる
  # print(coords.min(), coords.max())
  return embed_sincos(coords.clamp(-4, 4), expand, 0.125, max_freq)

def embed_timestep(t: Tensor, dim: int, min_timestep: float):
  return embed_sincos(t, dim, 1, 1 / min_timestep)

def mix_embeddings(x: Tensor, y: Tensor, ratio: float):
  return (x * ratio + y * (1 - ratio)) / math.sqrt(ratio**2 + (1 - ratio)**2)

class GeoDiffBlock(nn.Module):
  def __init__(self, dim: int, hdim: int):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(dim, hdim),
        nn.GELU(),
        nn.Linear(hdim, dim),
    )
    self.cond = nn.Sequential(
        nn.SiLU(),
        nn.Linear(dim, dim * 2),
    )
    self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    nn.init.zeros_(self.model[-1].weight)
    nn.init.zeros_(self.model[-1].bias)

  def forward(self, x: Tensor, cond: Tensor):
    gamma, mu = self.cond(cond).chunk(2, dim=-1)
    residual = (1 + gamma) * self.norm(x) + mu
    x = mix_embeddings(x, residual, 0.5)  # TODO: optimize ratio
    return x

class TimeEmbedder(nn.Module):
  def __init__(self, dim: int, min_timestep: float):
    super().__init__()
    self.dim = dim
    self.min_timestep = min_timestep

  def forward(self, t: Tensor):
    return embed_timestep(t, self.dim, self.min_timestep)

class CoordsEmbedder(nn.Module):
  def __init__(self, dim: int, max_freq: float):
    super().__init__()
    assert dim % 6 == 0
    self.dim = dim
    self.max_freq = max_freq

  def forward(self, x: Tensor):
    return embed_coordinates(x, self.dim // 3, self.max_freq)

# similar to https://arxiv.org/pdf/2212.09748
class GeoDiffModel(nn.Module):
  def __init__(self, idim: int, odim: int, hdim: int, cdim: int, *, depth: int, min_timestep: float, expansion=4):
    super().__init__()
    self.min_timestep = min_timestep
    self.hdim = hdim
    self.time = nn.Sequential(
        nn.Linear(hdim, hdim),
        nn.SiLU(),
        nn.Linear(hdim, hdim),
    )
    self.cond = nn.Linear(cdim, hdim)
    self.initial = nn.Sequential(
        nn.Linear(hdim, hdim),
        nn.SiLU(),
        nn.Linear(hdim, hdim),
    )
    self.blocks = nn.ModuleList([GeoDiffBlock(hdim, hdim * expansion) for _ in range(depth)])
    self.final = nn.Linear(hdim, odim)

  def forward(self, x: Tensor, t: Tensor, cond: Tensor):
    time_embed = self.time(embed_timestep(t, self.hdim, self.min_timestep))
    cond_embed = self.cond(cond)
    c = mix_embeddings(time_embed, cond_embed, 0.5)
    x = self.initial(x)
    for block in self.blocks:
      x = block(x, c)
    x = self.final(x)
    return x
