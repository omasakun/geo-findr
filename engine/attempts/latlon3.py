# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import questionary
import torch
from einops import rearrange
from lightning import LightningDataModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel

from engine.attempts.lib.dataset import GeoDatasets
from engine.attempts.lib.utils import (BaseLightningModule, LightningBar, LightningConfigSave, LightningModelCheckpoint, lightning_profiler, setup_environment,
                                       unique_run_name, wandb_logger)
from engine.lib.geo import (geoguesser_score, haversine_distance, latlon_to_xyz, xyz_to_latlon_torch)
from engine.lib.projection import equirectangular_to_planar
from engine.lib.utils import DATA, DotDict, num_workers_suggested
from engine.train import TrainContext

Batch = tuple[Tensor, Tensor, str]

class GeoModule(BaseLightningModule):
  def __init__(self, config: DotDict):
    super().__init__(config)

    self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
    self.classifier = torch.nn.Linear(self.vit.config.hidden_size, 3)
    self.criterion = torch.nn.MSELoss()

    for name, param in self.vit.named_parameters():
      if 'encoder.layer.11' not in name:
        param.requires_grad = False
    for param in self.classifier.parameters():
      param.requires_grad = True

  def configure_optimizers(self):
    parameters = list(self.classifier.parameters()) + list(self.vit.encoder.layer[11].parameters())
    return torch.optim.Adam(parameters, lr=self.config.learning_rate)

  def forward(self, x: Tensor):
    outputs = self.vit(x)
    return self.classifier(outputs.last_hidden_state[:, 0, :])

  def training_step(self, batch: Batch, batch_idx: int):
    self.log_amp_scale('amp_scale', prog_bar=True)

    images, targets, _ = batch
    preds = self.forward(images)
    loss = self.criterion(preds, targets)
    score = self.geoguess_score(preds, targets).mean()
    self.log('train/loss', loss)
    self.log('train/score', score, prog_bar=True)
    return loss

  def validation_step(self, batch: Batch, batch_idx: int):
    images, targets, _ = batch
    preds = self.forward(images)
    loss = self.criterion(preds, targets)
    score = self.geoguess_score(preds, targets).mean()
    self.log('valid/loss', loss)
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

class GeoDataModule(LightningDataModule):
  def __init__(self, config: DotDict, num_workers: int):
    super().__init__()
    self.config = config
    self.num_workers = num_workers
    self.datasets = GeoDatasets(config.dataset)
    self.transform = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

  def setup(self, stage=None):
    def mapper(item: dict):
      with torch.no_grad():
        image = item["panorama"]
        image = torch.as_tensor(np.array(image))
        image = rearrange(image, "h w c -> c h w")
        image = equirectangular_to_planar(image, 224, 224, 90, 0, 0, 0)  # TODO: random crop, resize, etc.
        image = self.transform(image, return_tensors='pt')['pixel_values'][0]
        meta = item["metadata"]
        lat, lon = meta['lat'], meta['lon']
        target = torch.as_tensor(latlon_to_xyz(lat, lon))
        country = meta.get("countryCode", "??")
        return image, target, country

    self.train_dataset = self.datasets("train", resampled=True, shardshuffle=True).map(mapper)
    self.valid_dataset = self.datasets("valid", resampled=False, shardshuffle=False).map(mapper)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.valid_dataset, batch_size=self.config.batch_size, num_workers=self.num_workers)

  def preview(self, batch):
    images, targets, countries = batch
    for image, target, country in zip(images, targets, countries):
      plt.title(f"{country}")
      plt.imshow(rearrange(image / 2 + 0.5, "c h w -> h w c"))
      plt.show()

@click.command(context_settings=dict(show_default=True))
@click.option("--project", default="geo-aa", help="Wandb project name.")
@click.option("--name", default=None, help="Name of the model. Used for loading and saving checkpoints.")
@click.option("--resume-from", default=None, help="Name of the model to resume from.")
@click.option("--weights-from", default=None, help="Name of the model to load weights from.")
@click.option("--num-workers", default=max(1, num_workers_suggested() - 1), help="Number of workers for data loading.")
@click.option("--log-frequency", default=50, help="Frequency of logging steps.")
@click.option("--val-frequency", default=1000, help="Frequency of validation steps.")
@click.option("--profile", is_flag=True, help="Enable profiler.")
@click.option("--skip-sanity-check", is_flag=True, help="Skip validation sanity check.")
@click.option("--detect-anomaly", is_flag=True, help="Detect anomaly.")
def train(ctx: TrainContext, project: str, name: Optional[str], resume_from: Optional[str], weights_from: Optional[str], num_workers: int, log_frequency: int,
          val_frequency: int, profile: bool, skip_sanity_check: bool, detect_anomaly: bool):
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

  datamodule = GeoDataModule(config, num_workers)

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
  )

  if resume_from: logger.log_hyperparams({"resume_from": resume_from})
  if weights_from: logger.log_hyperparams({"weights_from": weights_from})

  setup_environment(config)
  logger.watch(model, log=None, log_graph=True)
  trainer.fit(model, datamodule, ckpt_path=ckpt_file)
