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

from engine.lib.utils import sigmoid

Forward = Callable[[Tensor, Tensor], Tensor]  # (x_with_noise, var) -> noise

class NoiseSchedule(Protocol):
  def __call__(self, t: Tensor) -> Tensor:
    ...

class UniformSchedule(NoiseSchedule):
  def __call__(self, t):
    return t

# https://arxiv.org/abs/2301.10972
class SigmoidSchedule(NoiseSchedule):
  def __init__(self, start=-3.0, end=3.0, tau=1.0, clip_min=1e-6):
    self.start = start
    self.end = end
    self.tau = tau
    self.clip_min = clip_min

  def __call__(self, t):
    start, end, tau, clip_min = self.start, self.end, self.tau, self.clip_min
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    output = torch.sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return (1.0 - output).clamp(clip_min, 1.0)

class Diffusion:
  def __init__(self, schedule: NoiseSchedule, steps: int):
    self.schedule = schedule
    self.steps = steps

  def random_t(self, x: Tensor) -> Tensor:
    t = torch.randint(1, self.steps, (x.size(0), 1), device=x.device)
    return t.to(x.dtype) / self.steps

  def random_continuous_t(self, x: Tensor) -> Tensor:
    t = torch.rand(x.size(0), 1, device=x.device, dtype=x.dtype)
    return t

  def compute_loss(self, forward: Forward, x0: Tensor, noise: Tensor, t: Tensor):
    var = self.schedule(t)
    x_with_noise = (1 - var).sqrt() * x0 + var.sqrt() * noise
    noise_hat = forward(x_with_noise, var)
    return F.mse_loss(noise_hat, noise)

  def reverse(self, forward: Forward, x1: Tensor, return_traces=False):
    x_with_noise = x1
    x_trace = []
    x0_trace = []
    loglines = []
    for i in reversed(range(1, self.steps)):
      t = torch.ones((x_with_noise.size(0), 1), device=x_with_noise.device, dtype=x_with_noise.dtype) * (i / self.steps)
      var = self.schedule(t)
      noise_hat = forward(x_with_noise, var)
      x0_hat = (x_with_noise - noise_hat * var.sqrt()) / (1 - var).sqrt()

      if return_traces:
        loglines.append(f"{i:>4} var={var.item():.3f} std={var.sqrt().item():.3f} l2_noise={noise_hat.norm(dim=-1).mean().item():.3f}")

      if i == 1:
        if return_traces: return x0_hat, torch.stack(x_trace, dim=0), torch.stack(x0_trace, dim=0), "\n".join(loglines)
        return x0_hat

      var_next = self.schedule(t - 1 / self.steps)
      x_with_noise = x0_hat * (1 - var_next).sqrt() + noise_hat * var_next.sqrt()
      if return_traces: x_trace.append(x_with_noise)
      if return_traces: x0_trace.append(x0_hat)
    raise RuntimeError("unreachable")

  def reverse_random(self, forward: Forward, noise: Callable[[], Tensor], return_x0_trace=False):
    x1 = noise()
    assert x1.ndim == 3, f"noise() must return a tensor of shape (noise, batch, channel), but got {x1.shape}"
    n_noise = x1.size(0)
    x_with_noise = x1
    x0_trace = []
    for i in reversed(range(1, self.steps)):
      x_with_noise = rearrange(x_with_noise, 'n b c -> (n b) c')
      t = torch.ones((x_with_noise.size(0), 1), device=x_with_noise.device, dtype=x_with_noise.dtype) * (i / self.steps)
      var = self.schedule(t)
      noise_hat = forward(x_with_noise, var)
      x0_hat = (x_with_noise - noise_hat * var.sqrt()) / (1 - var).sqrt()
      x0_hat = rearrange(x0_hat, '(n b) c -> n b c', n=n_noise).mean(dim=0, keepdim=True)

      if i == 1:
        x0_hat = rearrange(x0_hat, '1 b c -> b c')
        if return_x0_trace: return x0_hat, torch.cat(x0_trace, dim=0)
        return x0_hat

      if return_x0_trace: x0_trace.append(x0_hat)

      var_next = self.schedule(t - 1 / self.steps)
      var_next = rearrange(var_next, '(n b) c -> n b c', n=n_noise)
      scaled_noise = noise() * var_next.sqrt()
      x_with_noise = x0_hat * (1 - var_next).sqrt() + scaled_noise
    raise RuntimeError("unreachable")

def _main():
  # visualize sigmoid schedule
  import matplotlib.pyplot as plt
  import numpy as np

  schedule = SigmoidSchedule()
  t = np.linspace(0, 1, 100)
  y = schedule(torch.tensor(t)).numpy()
  plt.plot(t, y, '.')
  plt.show()

  plt.plot(t, y, '.')
  plt.yscale('log')
  plt.show()

if __name__ == "__main__": _main()
