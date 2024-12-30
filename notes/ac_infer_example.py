# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import torch
from einops import repeat
from torch import Tensor
from tqdm import tqdm

from engine.attempts.cb_finetune import GeoModule
from engine.attempts.lib.dataset import GeoVitDataModule
from engine.attempts.lib.utils import setup_environment
from engine.lib.geo import xyz_to_latlon_torch
from engine.lib.utils import DATA

Batch = tuple[Tensor, Tensor, str]

model = GeoModule.load_from_checkpoint(DATA / "models" / "siraka" / "last.ckpt")
config = model.config

datamodule = GeoVitDataModule(config, num_workers=1, cache_size=0)
setup_environment(config)

datamodule.setup()

# %%

batch = next(iter(datamodule.val_dataloader()))
batch = datamodule.on_after_batch_transfer(batch, 0)
# datamodule.preview(batch)

batch = (batch[0][14:15], batch[1][14:15], batch[2][14:15])

model.eval().to("cpu")

hats = []
scores = []

with torch.no_grad():
  images, targets, countries = batch
  vit_features = model.forward_vit(images)

  for j in tqdm(range(1000)):
    hat = model.diffusion.reverse_random(
        lambda x, var: model.forward_diffusion(repeat(vit_features, "b c -> (n b) c", n=1), var, x),
        lambda: torch.randn((1, *targets.shape), device=targets.device, dtype=targets.dtype) * model.config.init_noise_scale,
    )
    score = model.geoguess_score(hat, targets)
    hats.append(hat)
    scores.append(score)

hats = torch.stack(hats)
scores = torch.stack(scores)

# %%

scores.max(dim=0).values.mean()

# %%

scores.min(dim=0).values.mean()

# %%

for j, (image, target, country) in enumerate(zip(images, targets, countries)):
  i = torch.argmax(scores[:, j])
  lat, lon = xyz_to_latlon_torch(*targets[j].cpu())
  hat_lat, hat_lon = xyz_to_latlon_torch(*hats[i, j].cpu())
  hat_radius = torch.norm(hats[i, j])
  print(f"#{j}: {scores[i, j]:.2f} ({lat:.4f}, {lon:.4f}) -> ({hat_lat:.4f}, {hat_lon:.4f} / {hat_radius:.4f})")
  datamodule.preview_sample((image, target, country))
