# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import math

import torch
from einops import rearrange

def equirectangular_to_planar(panorama, width: int, height: int, fov: float, heading: float, pitch: float, roll: float):
  """ panorama: (channels, height, width) """

  device = panorama.device

  heading = heading * math.pi / 180
  pitch = pitch * math.pi / 180
  roll = roll * math.pi / 180
  fov = fov * math.pi / 180

  focal_length = (width / 2) / math.tan(fov / 2)

  R_heading = torch.tensor([[math.cos(heading), 0, -math.sin(heading)], [0, 1, 0], [math.sin(heading), 0, math.cos(heading)]], device=device)
  R_pitch = torch.tensor([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]], device=device)
  R_roll = torch.tensor([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]], device=device)
  R = torch.mm(torch.mm(R_heading, R_pitch), R_roll)

  x, y = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing='xy')
  x = x - width / 2
  y = y - height / 2
  z = focal_length * torch.ones_like(x)
  coords = torch.stack((x, y, z), dim=-1)
  coords = torch.matmul(coords, R.T)

  lat = torch.asin(coords[..., 1] / torch.norm(coords, dim=-1))
  lon = torch.atan2(coords[..., 0], coords[..., 2])

  lat = (lat + math.pi / 2) / math.pi * panorama.shape[1]
  lon = (lon + math.pi) / (2 * math.pi) * panorama.shape[2]

  lat = torch.clamp(lat.long(), 0, panorama.shape[1] - 1)
  lon = torch.clamp(lon.long(), 0, panorama.shape[2] - 1)
  output = panorama[:, lat, lon]

  # output = F.grid_sample(panorama.unsqueeze(0), torch.stack((lon, lat), dim=-1).unsqueeze(0), mode=mode, padding_mode="border", align_corners=False).squeeze(0)

  return output

def _main():
  import matplotlib.pyplot as plt

  from engine.attempts.lib.dataset import panorama_examples

  image, _ = next(iter(panorama_examples()))

  plt.imshow(rearrange(image, "c h w -> h w c"))
  plt.show()

  # Check the rotation
  projected = equirectangular_to_planar(image, 768, 512, 90, 90, 0, 0)
  plt.imshow(rearrange(projected, "c h w -> h w c"))
  plt.show()
  projected = equirectangular_to_planar(image, 768, 512, 90, 90, 30, 0)
  plt.imshow(rearrange(projected, "c h w -> h w c"))
  plt.show()
  projected = equirectangular_to_planar(image, 768, 512, 90, 90, 30, 30)
  plt.imshow(rearrange(projected, "c h w -> h w c"))
  plt.show()

  fig, axes = plt.subplots(3, 4, figsize=(20, 15))
  headings = range(0, 360, 30)
  for i, heading in enumerate(headings):
    row, col = divmod(i, 4)
    projected = equirectangular_to_planar(image, 512, 512, 90, heading, 0, 0)
    axes[row, col].imshow(rearrange(projected, "c h w -> h w c"))
    axes[row, col].set_title(f'Heading {heading}')
    axes[row, col].axis('off')

  plt.show()

  fig, axes = plt.subplots(2, 4, figsize=(20, 10))
  pitches = range(-45, 46, 15)
  for i, pitch in enumerate(pitches):
    row, col = divmod(i, 4)
    projected = equirectangular_to_planar(image, 512, 512, 90, 0, pitch, 0)
    axes[row, col].imshow(rearrange(projected, "c h w -> h w c"))
    axes[row, col].set_title(f'Pitch {pitch}')
    axes[row, col].axis('off')

  plt.show()

  fig, axes = plt.subplots(2, 4, figsize=(20, 10))
  rolls = range(-45, 46, 15)
  for i, roll in enumerate(rolls):
    row, col = divmod(i, 4)
    projected = equirectangular_to_planar(image, 512, 512, 90, 0, 0, roll)
    axes[row, col].imshow(rearrange(projected, "c h w -> h w c"))
    axes[row, col].set_title(f'Roll {roll}')
    axes[row, col].axis('off')

  plt.show()

  fig, axes = plt.subplots(3, 4, figsize=(20, 15))
  fovs = range(15, 181, 15)
  for i, fov in enumerate(fovs):
    row, col = divmod(i, 4)
    projected = equirectangular_to_planar(image, 512, 512, fov, 0, 0, 0)
    axes[row, col].imshow(rearrange(projected, "c h w -> h w c"))
    axes[row, col].set_title(f'FOV {fov}')
    axes[row, col].axis('off')

  plt.show()

if __name__ == "__main__": _main()
