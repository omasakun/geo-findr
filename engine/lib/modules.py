# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

def embed_timesteps(timesteps: Tensor, ndim: int, max_timestep: float):
  if timesteps.ndim == 2: timesteps = rearrange(timesteps, 'b 1 -> b')
  assert timesteps.ndim == 1
  device = timesteps.device
  dtype = timesteps.dtype

  half_dim = ndim // 2
  emb = torch.arange(half_dim, dtype=dtype, device=device) / (half_dim - 1)  # 0 ~ 1
  emb = torch.exp(emb * (-math.log(max_timestep)))  # 1/max_time ~ 1
  emb = timesteps[:, None] * emb[None, :] * math.pi  # half rotation every max_time steps ~ half rotation every step
  emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
  if ndim % 2 == 1: emb = F.pad(emb, (0, 1))

  assert emb.shape == (timesteps.shape[0], ndim)
  return emb

class TimeEmbedder(nn.Module):
  def __init__(self, dim: int, max_time: float):
    super().__init__()
    self.dim = dim
    self.max_time = max_time
    self.map_time = nn.Sequential(
        nn.Linear(dim, dim),
        nn.SiLU(),
        nn.Linear(dim, dim),
    )

  def forward(self, t):
    x = embed_timesteps(t, self.dim, self.max_time)
    x = self.map_time(x)
    return x

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
        nn.Linear(dim, dim * 3),
    )
    self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    nn.init.zeros_(self.model[-1].weight)
    nn.init.zeros_(self.model[-1].bias)

  def forward(self, x: Tensor, cond: Tensor):
    gamma, mu, sigma = self.cond(cond).chunk(3, dim=-1)
    residual = (1 + gamma) * self.norm(x) + mu
    x = x + self.model(residual) * sigma
    return x

# similar to https://arxiv.org/pdf/2212.09748
class GeoDiffModel(nn.Module):
  def __init__(self, idim: int, odim: int, hdim: int, cdim: int, depth: int, max_timestep: float, expansion=4):
    super().__init__()
    self.time = TimeEmbedder(hdim, max_timestep)
    self.cond = nn.Linear(cdim, hdim)
    self.initial = nn.Linear(idim, hdim)
    self.blocks = nn.ModuleList([GeoDiffBlock(hdim, hdim * expansion) for _ in range(depth)])
    self.final = nn.Linear(hdim, odim)

  def forward(self, x: Tensor, t: Tensor, cond: Tensor):
    c = self.time(t) + self.cond(cond)
    x = self.initial(x)
    for block in self.blocks:
      x = block(x, c)
    x = self.final(x)
    return x
