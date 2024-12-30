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
from einops import rearrange, repeat
from lightning import Trainer
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from transformers import ViTModel

from engine.attempts.lib.dataset import GeoVitDataModule
from engine.attempts.lib.utils import (BaseLightningModule, LightningBar, LightningConfigSave, LightningModelCheckpoint, lightning_profiler, set_learning_rate,
                                       setup_environment, unique_run_name, wandb_logger)
from engine.lib.diffusion import Diffusion, UniformSchedule
from engine.lib.geo import (geoguesser_score, haversine_distance, xyz_to_latlon_torch)
from engine.lib.modules import CoordsEmbedder, TimeEmbedder
from engine.lib.utils import DATA, DotDict, num_workers_suggested
from engine.train import TrainContext

Batch = tuple[Tensor, Tensor, str]

class GeoModule(BaseLightningModule):
  def __init__(self, config: DotDict, score_frequency: int):
    super().__init__(config)
    self.score_frequency = score_frequency

    self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    self.diffusion = Diffusion(UniformSchedule(), self.config.diffusion_steps)

    hdim = self.vit.config.hidden_size
    assert self.diffusion.steps <= 1000
    self.embed_time = TimeEmbedder(hdim, 1 / 1000)
    self.embed_coords = CoordsEmbedder(hdim, 1000.0)  # 1000.0 may not be the best value
    self.post_time = nn.Sequential(
        nn.Linear(hdim, hdim),
        nn.SiLU(inplace=True),
        nn.Linear(hdim, hdim),
    )
    self.post_coords = nn.Sequential(
        nn.Linear(hdim, hdim),
        nn.SiLU(inplace=True),
        nn.Linear(hdim, hdim),
    )
    self.final = nn.Linear(hdim, 3)

    # TODO: shoud we zero-initialize the weights of the post_time and post_coords last layers?

  def forward(self, images: Tensor, coords: Tensor, noise_var: Tensor):
    vit = self.vit
    projections = images.shape[1] // 3

    if noise_var.size(0) == 1: noise_var = noise_var.expand(images.size(0), -1)

    time = self.post_time(self.embed_time(noise_var))
    coords = self.post_coords(self.embed_coords(coords))

    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/vit/modeling_vit.py#L592

    images = rearrange(images, 'b (p c) h w -> (b p) c h w', p=projections)
    time = repeat(time, 'b c -> (b p) 1 c', p=projections)
    coords = repeat(coords, 'b c -> (b p) 1 c', p=projections)

    patches = vit.embeddings(images, interpolate_pos_encoding=True)
    patches = torch.cat((patches, time, coords), dim=1)
    patches = vit.encoder(patches)
    patches = vit.layernorm(patches.last_hidden_state)  # (batch, patch, channels)

    output = patches[:, -1, :]  # extract the coords token

    output = rearrange(output, '(b p) c -> b p c', p=projections)
    output = output.mean(dim=1)
    output = self.final(output)
    return output

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

  def training_step(self, batch: Batch, batch_idx: int):
    self.log_amp_scale('amp_scale', prog_bar=True)
    step = self.current_step()

    optimizer: Optimizer = self.optimizers()  # type: ignore
    set_learning_rate(optimizer, self.config.learning_rate)

    images, targets, _ = batch

    num_times = self.config.num_times
    images = repeat(images, 'b ... -> (n b) ...', n=num_times)
    targets = repeat(targets, 'b ... -> (n b) ...', n=num_times)

    noise = torch.randn_like(targets) * self.config.init_noise_scale
    noise_t = self.diffusion.random_t(targets)
    loss = self.diffusion.compute_loss(lambda x, var: self.forward(images, x, var), targets, noise, noise_t)
    self.log('train/loss', loss)

    lr_decay = self.config.learning_rate_decay
    lr = self.config.learning_rate
    lr *= min(1.0, step / self.config.warmup_steps)
    if lr_decay == "none": pass
    elif lr_decay == "cosine": lr *= 0.5 * (1 + math.cos(math.pi * step / self.config.steps))
    else: raise ValueError(f"Unknown lr_decay: {lr_decay}")

    set_learning_rate(optimizer, lr)
    self.log_lr('lr', optimizer)
    return loss

  def validation_step(self, batch: Batch, batch_idx: int):
    step = self.current_step()
    images, targets, _ = batch

    noise = torch.randn_like(targets) * self.config.init_noise_scale
    noise_t = self.diffusion.random_t(targets)
    loss = self.diffusion.compute_loss(lambda x, var: self.forward(images, x, var), targets, noise, noise_t)
    self.log('valid/loss', loss)

    if (step + 1) % self.score_frequency == 0:
      # hat = self.diffusion.reverse(lambda x, var: self.forward(images, x, var), noise)
      # scores = self.geoguess_score(hat, targets)
      # score = scores.mean()
      # self.log('valid/score', score)
      # self.validation_scores.append(scores.detach())

      hat = self.diffusion.reverse_random(
          lambda x, var: self.forward(images, x, var),
          lambda: torch.randn((1, *targets.shape), device=targets.device, dtype=targets.dtype) * self.config.init_noise_scale,
      )
      scores = self.geoguess_score(hat, targets)
      score = scores.mean()
      self.log('valid/score_random', score)
      self.log('score', score, prog_bar=True)
      self.validation_scores_random.append(scores.detach())

      # hat = self.diffusion.reverse_random(
      #     lambda x, var: self.forward(repeat(images, "b h w c -> (n b) h w c", n=10), x, var),
      #     lambda: torch.randn((10, *targets.shape), device=targets.device, dtype=targets.dtype) * self.config.init_noise_scale,
      # )
      # scores = self.geoguess_score(hat, targets)
      # score = scores.mean()
      # self.log('valid/score_random10', score)
      # self.validation_scores_random10.append(scores.detach())

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
@click.option("--score-frequency", default=10000, help="Frequency of scoring steps.")
@click.option("--profile", is_flag=True, help="Enable profiler.")
@click.option("--skip-sanity-check", is_flag=True, help="Skip validation sanity check.")
@click.option("--detect-anomaly", is_flag=True, help="Detect anomaly.")
def train(ctx: TrainContext, project: str, name: Optional[str], resume_from: Optional[str], weights_from: Optional[str], num_workers: int, cache_size: int,
          log_frequency: int, val_frequency: int, score_frequency: int, profile: bool, skip_sanity_check: bool, detect_anomaly: bool):
  name = name or unique_run_name(project, 3)
  config = ctx.config

  print()
  print(f"Training '{name}' with {num_workers} workers and validation every {val_frequency} steps.")
  assert not (resume_from and weights_from), "Cannot resume from a checkpoint and load weights at the same time."
  assert score_frequency % val_frequency == 0, "score_frequency must be a multiple of val_frequency."

  model_dir = DATA / "models" / name
  ckpt_file = model_dir / "last.ckpt" if model_dir.exists() else None
  if resume_from: ckpt_file = DATA / "models" / resume_from / "last.ckpt"

  if ckpt_file and config != GeoModule.load_config_from_checkpoint(ckpt_file):
    if not questionary.confirm("Checkpoint config does not match current config. Continue?", default=False).ask(): return
    print("Continuing with current config.")

  model = GeoModule(config, score_frequency)

  if weights_from: model.load_weights_from_checkpoint(DATA / "models" / weights_from / "last.ckpt", strict=False)

  datamodule = GeoVitDataModule(config, num_workers, cache_size)

  # datamodule.setup()
  # for batch in datamodule.train_dataloader():
  #   batch = datamodule.on_after_batch_transfer(batch, 0)
  #   datamodule.preview(batch)

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
              filename="{step:08d}-{score:.3f}",
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
