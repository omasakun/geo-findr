# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import math
from typing import Optional

import click
import questionary
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from lightning import Trainer
from torch import Tensor
from transformers import ViTModel

from engine.attempts.lib.dataset import GeoVitXyzDataModule
from engine.attempts.lib.utils import (BaseLightningModule, LightningBar, LightningConfigSave, LightningModelCheckpoint, lightning_profiler, setup_environment,
                                       unique_run_name, wandb_logger)
from engine.lib.ddpm import Ddpm
from engine.lib.geo import (geoguesser_score, haversine_distance, xyz_to_latlon_torch)
from engine.lib.modules import GeoDiffModel
from engine.lib.utils import DATA, DotDict, num_workers_suggested
from engine.train import TrainContext

Batch = tuple[Tensor, Tensor, str]

class GeoModule(BaseLightningModule):
  def __init__(self, config: DotDict):
    super().__init__(config)

    self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
    self.ddpm = Ddpm()

    hdim = self.vit.config.hidden_size
    self.head = GeoDiffModel(768 + 3, 3, hdim, hdim, depth=6, max_timestep=self.ddpm.steps)

    for name, param in self.vit.named_parameters():
      if 'encoder.layer.11' not in name:
        param.requires_grad = False
    for param in self.head.parameters():
      param.requires_grad = True

  def encode_coordinates(self, coords: Tensor, expand=128, max_freq=1000.0):
    # coords: (n_batch, 3)
    device, dtype = coords.device, coords.dtype
    scale = torch.arange(expand, device=device, dtype=dtype)
    scale = max_freq**(scale / (expand - 1))  # 1, ..., max_freq
    scale = (math.pi / 2) * scale
    enc = rearrange(coords, 'b c -> b c 1') * rearrange(scale, 'c -> 1 1 c')
    enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)
    enc = rearrange(enc, 'b c1 c2 -> b (c1 c2)')
    return torch.cat([coords, enc], dim=-1)  # TODO: coords might not be needed

  def configure_optimizers(self):
    parameters = list(self.head.parameters()) + list(self.vit.encoder.layer[11].parameters())
    return torch.optim.Adam(parameters, lr=self.config.learning_rate)

  def forward_vit(self, x: Tensor):
    vit_outputs = self.vit(x, interpolate_pos_encoding=True)
    vit_features = vit_outputs.last_hidden_state[:, 0, :]
    return vit_features

  def forward_diffusion(self, vit_features: Tensor, noise_t: Tensor | int, xt: Tensor):
    if isinstance(noise_t, int): noise_t = torch.ones(xt.size(0), 1, device=xt.device) * noise_t
    noise_t = noise_t.to(self.dtype) / self.ddpm.steps
    xt_enc = self.encode_coordinates(xt)
    return self.head(xt_enc, noise_t, vit_features)

  def training_step(self, batch: Batch, batch_idx: int):
    self.log_amp_scale('amp_scale', prog_bar=True)
    self.ddpm.to(self.dtype, self.device)

    images, targets, _ = batch
    device = images.device

    vit_features = self.forward_vit(images)

    num_times = self.config.num_times
    targets = repeat(targets, 'b ... -> (n b) ...', n=num_times)
    vit_features = repeat(vit_features, 'b ... -> (n b) ...', n=num_times)
    n_batch = targets.size(0)

    noise = torch.randn_like(targets)
    noise_t = torch.randint(0, self.ddpm.steps, (n_batch, 1), device=device)
    xt = self.ddpm.add_noise(targets, noise, noise_t)
    noise_hat = self.forward_diffusion(vit_features, noise_t, xt)

    loss = F.mse_loss(noise_hat, noise)
    self.log('train/loss', loss)
    return loss

  def validation_step(self, batch: Batch, batch_idx: int):
    self.ddpm.to(self.dtype, self.device)

    images, targets, _ = batch
    device = images.device
    n_batch = images.size(0)

    vit_features = self.forward_vit(images)

    noise = torch.randn_like(targets)
    noise_t = torch.randint(0, self.ddpm.steps, (n_batch, 1), device=device)
    xt = self.ddpm.add_noise(targets, noise, noise_t)
    noise_hat = self.forward_diffusion(vit_features, noise_t, xt)

    loss = F.mse_loss(noise_hat, noise)
    self.log('valid/loss', loss)

    xt = torch.randn_like(targets)
    for t in reversed(range(self.ddpm.steps)):
      noise_hat = self.forward_diffusion(vit_features, t, xt)
      xt, _ = self.ddpm.remove_noise(xt, noise_hat, t)

    score = self.geoguess_score(xt, targets).mean()
    self.log('valid/score', score)
    self.log('score', score, prog_bar=True)
    return loss

  def geoguess_score(self, preds, targets):
    with torch.no_grad():
      preds = preds.to(torch.float64)
      targets = targets.to(torch.float64)
      pred_lat, pred_lon = xyz_to_latlon_torch(preds[:, 0], preds[:, 1], preds[:, 2])
      target_lat, target_lon = xyz_to_latlon_torch(targets[:, 0], targets[:, 1], targets[:, 2])
      score: Tensor = geoguesser_score(haversine_distance(pred_lat, pred_lon, target_lat, target_lon))  # type: ignore
      return score

@click.command(context_settings=dict(show_default=True))
@click.option("--project", default="geo-aa", help="Wandb project name.")
@click.option("--name", default=None, help="Name of the model. Used for loading and saving checkpoints.")
@click.option("--resume-from", default=None, help="Name of the model to resume from.")
@click.option("--weights-from", default=None, help="Name of the model to load weights from.")
@click.option("--num-workers", default=max(1, num_workers_suggested() - 1), help="Number of workers for data loading.")
@click.option("--cache-size", default=int(-1), help="Size of the webdataset cache. (0 = no cache, -1 = unlimited)")
@click.option("--log-frequency", default=50, help="Frequency of logging steps.")
@click.option("--val-frequency", default=1000, help="Frequency of validation steps.")
@click.option("--profile", is_flag=True, help="Enable profiler.")
@click.option("--skip-sanity-check", is_flag=True, help="Skip validation sanity check.")
@click.option("--detect-anomaly", is_flag=True, help="Detect anomaly.")
def train(ctx: TrainContext, project: str, name: Optional[str], resume_from: Optional[str], weights_from: Optional[str], num_workers: int, cache_size: int,
          log_frequency: int, val_frequency: int, profile: bool, skip_sanity_check: bool, detect_anomaly: bool):
  name = name or unique_run_name(project, 3)
  config = ctx.config

  print()
  print(f"Training '{name}' with {num_workers} workers and validation every {val_frequency} steps.")
  assert not (resume_from and weights_from), "Cannot resume from a checkpoint and load weights at the same time."

  model_dir = DATA / "models" / name
  ckpt_file = model_dir / "last.ckpt" if model_dir.exists() else None
  if resume_from: ckpt_file = DATA / "models" / resume_from / "last.ckpt"

  if ckpt_file and config != GeoModule.load_config_from_checkpoint(ckpt_file):
    if not questionary.confirm("Checkpoint config does not match current config. Continue?", default=False).ask(): return
    print("Continuing with current config.")

  model = GeoModule(config)

  if weights_from: model.load_weights_from_checkpoint(DATA / "models" / weights_from / "last.ckpt", strict=False)

  datamodule = GeoVitXyzDataModule(config, num_workers, cache_size)

  # datamodule.setup()
  # for batch in datamodule.train_dataloader():
  #   datamodule.preview(batch)
  #   break

  logger = wandb_logger(project, name)

  trainer = Trainer(
      max_steps=config.steps,
      precision=config.precision,
      log_every_n_steps=log_frequency,
      val_check_interval=val_frequency,
      check_val_every_n_epoch=None,
      callbacks=[
          LightningBar(name),
          LightningConfigSave(model_dir),
          LightningModelCheckpoint(
              dirpath=model_dir,
              filename="{step:08d}-{multimel:.3f}",
              monitor="score",
              mode="max",
              save_top_k=3,
              save_last=True,
          ),
      ],
      logger=logger,
      profiler=lightning_profiler() if profile else None,
      detect_anomaly=detect_anomaly,
      num_sanity_val_steps=0 if skip_sanity_check else 2,
      accumulate_grad_batches=config.accumulate_grad,
  )

  if resume_from: logger.log_hyperparams({"resume_from": resume_from})
  if weights_from: logger.log_hyperparams({"weights_from": weights_from})

  setup_environment(config)
  logger.watch(model, log=None, log_graph=True)
  trainer.fit(model, datamodule, ckpt_path=ckpt_file)
