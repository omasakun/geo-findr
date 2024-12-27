# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

from typing import Callable, Protocol

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

Forward = Callable[[Tensor, Tensor], Tensor]  # (x_with_noise, var) -> noise

class NoiseSchedule(Protocol):
  def __call__(self, t: Tensor) -> Tensor:
    ...

class UniformSchedule(NoiseSchedule):
  def __call__(self, t):
    return t

class Diffusion:
  def __init__(self, schedule: NoiseSchedule, steps: int):
    self.schedule = schedule
    self.steps = steps

  def random_t(self, x: Tensor) -> Tensor:
    t = torch.randint(1, self.steps, (x.size(0), 1), device=x.device)
    return t.to(x.dtype) / self.steps

  def compute_loss(self, forward: Forward, x0: Tensor, noise: Tensor, t: Tensor):
    var = self.schedule(t)
    x_with_noise = (1 - var).sqrt() * x0 + var.sqrt() * noise
    noise_hat = forward(x_with_noise, var)
    return F.mse_loss(noise_hat, noise)

  def reverse(self, forward: Forward, x1: Tensor):
    x_with_noise = x1
    for i in reversed(range(1, self.steps)):
      t = torch.ones((x_with_noise.size(0), 1), device=x_with_noise.device, dtype=x_with_noise.dtype) * (i / self.steps)
      var = self.schedule(t)
      noise_hat = forward(x_with_noise, var)
      x0_hat = (x_with_noise - noise_hat * var.sqrt()) / (1 - var).sqrt()

      if i == 1: return x0_hat

      var_next = self.schedule(t - 1 / self.steps)
      x_with_noise = x0_hat * (1 - var_next).sqrt() + noise_hat * var_next.sqrt()
    raise RuntimeError("unreachable")

  def reverse_random(self, forward: Forward, noise: Callable[[], Tensor]):
    x1 = noise()
    assert x1.ndim == 3, "noise() must return a tensor of shape (noise, batch, channel)"
    n_noise = x1.size(0)
    x_with_noise = x1
    for i in reversed(range(1, self.steps)):
      x_with_noise = rearrange(x_with_noise, 'n b c -> (n b) c')
      t = torch.ones((x_with_noise.size(0), 1), device=x_with_noise.device, dtype=x_with_noise.dtype) * (i / self.steps)
      var = self.schedule(t)
      noise_hat = forward(x_with_noise, var)
      x0_hat = (x_with_noise - noise_hat * var.sqrt()) / (1 - var).sqrt()
      x0_hat = rearrange(x0_hat, '(n b) c -> n b c', n=n_noise).mean(dim=0, keepdim=True)

      if i == 1: return rearrange(x0_hat, '1 b c -> b c')

      var_next = self.schedule(t - 1 / self.steps)
      var_next = rearrange(var_next, '(n b) c -> n b c', n=n_noise)
      scaled_noise = noise() * var_next.sqrt()
      x_with_noise = x0_hat * (1 - var_next).sqrt() + scaled_noise
    raise RuntimeError("unreachable")
