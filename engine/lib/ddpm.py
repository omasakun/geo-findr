# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import torch
from torch import Tensor

# https://arxiv.org/pdf/2006.11239
class Ddpm:
  def __init__(self, *, beta_min=0.001, beta_max=0.2, steps=100):
    self.steps = steps
    self.betas = torch.linspace(beta_min, beta_max, steps).double()
    self.alphas = 1. - self.betas
    self.alphas_cumprod = self.alphas.cumprod(0)

  def to(self, dtype, device):
    self.betas = self.betas.to(dtype=dtype, device=device)
    self.alphas = self.alphas.to(dtype=dtype, device=device)
    self.alphas_cumprod = self.alphas_cumprod.to(dtype=dtype, device=device)

  def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor):
    alpha_cumprod_t = self.alphas_cumprod[t]
    return x0 * alpha_cumprod_t.sqrt() + noise * (1 - alpha_cumprod_t).sqrt()

  def remove_noise(self, xt: Tensor, noise_hat: Tensor, t: int):
    """ x_t -> x_{t-1} """

    alpha_t = self.alphas[t]
    alphas_cumprod = self.alphas_cumprod

    # reverse add_noise
    x0 = xt - noise_hat * (1. - alphas_cumprod[t]).sqrt()
    x0 = x0 / alphas_cumprod[t].sqrt()

    mean = xt - noise_hat * (1 - alpha_t) / (1 - alphas_cumprod[t]).sqrt()
    mean = mean / alpha_t.sqrt()

    if t == 0: return mean, x0

    var = (1 - alphas_cumprod[t - 1]) / (1 - alphas_cumprod[t]) * self.betas[t]
    noise = torch.randn_like(x0)

    return mean + noise * var.sqrt(), x0

def _main():
  import matplotlib.pyplot as plt

  ddpm = Ddpm()
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 3, 1)
  plt.plot(ddpm.betas.numpy())
  plt.title('Betas')

  plt.subplot(1, 3, 2)
  plt.plot(ddpm.alphas_cumprod.numpy())
  plt.title('Alphas Cumprod')

  plt.subplot(1, 3, 3)
  plt.plot(ddpm.alphas_cumprod.numpy())
  plt.yscale('log')
  plt.title('Alphas Cumprod (log scale)')

  plt.tight_layout()
  plt.show()

if __name__ == '__main__': _main()
