# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import itertools
from pathlib import Path
from typing import Any, override

import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from torch import Tensor
from torch.optim import Optimizer

from engine.lib.geo import geoguesser_score, haversine_distance
from engine.lib.utils import (CODE, DATA, DotDict, random_name, save_json, wandb_histogram)

# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html
class BaseLightningModule(LightningModule):
  def __init__(self, config: DotDict):
    super().__init__()
    self.config = config
    self.validation_scores: list[Tensor] = []

  def current_step(self):
    return self.trainer.fit_loop.epoch_loop._batches_that_stepped

  def step_optimizer(self, optimizer: Optimizer, loss: Tensor, zero_grad=True, grad_clip: float | None = None, check_nan=False, lr: float | None = None):
    self.toggle_optimizer(optimizer)
    if lr is not None: set_learning_rate(optimizer, lr)
    if zero_grad: optimizer.zero_grad()
    self.manual_backward(loss)
    if grad_clip: self.clip_gradients(optimizer, grad_clip, "norm")
    optimizer.step()
    self.untoggle_optimizer(optimizer)
    if check_nan and torch.isnan(loss).any(): raise ValueError("Loss contains NaN")

  def log_lr(self, name: str, optimizer: Optimizer | None, **kwargs):
    if optimizer:
      self.log(name, optimizer.param_groups[0]["lr"], **kwargs)

  def log_amp_scale(self, name: str, **kwargs):
    amp_scaler = getattr(self.trainer.precision_plugin, "scaler", None)
    if amp_scaler: self.log(name, amp_scaler.get_scale(), **kwargs)

  @override
  def on_validation_epoch_start(self):
    self.validation_scores = []
    return super().on_validation_epoch_start()

  @override
  def on_validation_epoch_end(self):
    if self.validation_scores:
      scores = torch.cat(self.validation_scores)
      self.logger.log_metrics({'valid/score_hist': wandb_histogram(scores, num_bins=100, range=(0, 5000))})  # type: ignore
    return super().on_validation_epoch_end()

  @override
  def on_save_checkpoint(self, checkpoint: dict[str, Any]):
    checkpoint["config"] = self.config.as_dict()

  # @override
  # def on_load_checkpoint(self, checkpoint: dict[str, Any]):
  #   self.config = DotDict(checkpoint["config"])

  # @classmethod
  # def load_from_checkpoint(cls, checkpoint_path: str | Path, **kwargs):  # type: ignore
  #   config = cls.load_config_from_checkpoint(checkpoint_path)
  #   return super().load_from_checkpoint(checkpoint_path, config=config, **kwargs)

  def load_weights_from_checkpoint(self, checkpoint_path: str | Path, strict=True):
    weights = torch.load(checkpoint_path)['state_dict']
    self.load_state_dict(weights, strict=strict)

  @classmethod
  def load_config_from_checkpoint(cls, checkpoint_path: str | Path):
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return DotDict(data["config"])

class LightningBar(TQDMProgressBar):
  def __init__(self, name: str, **kwargs):
    super().__init__(**kwargs)
    self.name = name

  @override
  def init_sanity_tqdm(self):
    tqdm = super().init_sanity_tqdm()
    tqdm.ncols = 0
    tqdm.dynamic_ncols = False
    return tqdm

  @override
  def init_train_tqdm(self):
    tqdm = super().init_train_tqdm()
    tqdm.ncols = 0
    tqdm.dynamic_ncols = False
    return tqdm

  @override
  def init_predict_tqdm(self):
    tqdm = super().init_predict_tqdm()
    tqdm.ncols = 0
    tqdm.dynamic_ncols = False
    return tqdm

  @override
  def init_validation_tqdm(self):
    tqdm = super().init_validation_tqdm()
    tqdm.ncols = 0
    tqdm.dynamic_ncols = False
    return tqdm

  @override
  def init_test_tqdm(self):
    tqdm = super().init_test_tqdm()
    tqdm.ncols = 0
    tqdm.dynamic_ncols = False
    return tqdm

  @override
  def on_train_epoch_start(self, trainer: Trainer, *args):
    super().on_train_epoch_start(trainer, *args)
    self.train_progress_bar.set_description(self.name)

  @override
  def get_metrics(self, trainer, pl_module) -> dict:
    items = []
    for k, v in super().get_metrics(trainer, pl_module).items():
      if k == "v_num": continue
      items.append((k, v))
    items.sort(key=lambda x: x[0])
    return dict(items)

class LightningModelCheckpoint(ModelCheckpoint):
  """Save checkpoint after every validation, not after training batch."""
  @override
  def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:
    assert not self._train_time_interval, "Not supported."
    pass  # do not save checkpoint now, delay until on_validation_batch_end

  @override
  def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    assert not self._should_save_on_train_epoch_end(trainer), "Not supported."

  @override
  def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    assert not self._should_save_on_train_epoch_end(trainer), "Not supported."
    if self._should_skip_saving_checkpoint(trainer): return
    # print(f"LightningModelCheckpoint {trainer.global_step=}")

    # save checkpoint
    monitor_candidates = self._monitor_candidates(trainer)
    self._save_topk_checkpoint(trainer, monitor_candidates)
    self._save_last_checkpoint(trainer, monitor_candidates)

class LightningConfigSave(Callback):
  def __init__(self, dirpath: str | Path):
    self.dirpath = Path(dirpath)
    self._saved = False

  def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx) -> None:
    if self._saved: return

    config = pl_module.config
    assert isinstance(config, DotDict), "Module must have a DotDict config attribute."

    if trainer.logger: trainer.logger.log_hyperparams(config.as_flat_dict())

    for i in itertools.count(0):
      filename = self.dirpath / f"config.{i}.json"
      if not Path(filename).exists():
        save_json(config, filename)
        break

    self._saved = True

def wandb_logger(project: str, name: str):
  return WandbLogger(project=project, name=name, id=name, save_dir=DATA, settings=wandb.Settings(code_dir=str(CODE)))

def lightning_profiler(**kwargs):
  # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.profilers.PyTorchProfiler.html
  return PyTorchProfiler(
      DATA / "prof-torch",
      export_to_chrome=True,
      schedule=torch.profiler.schedule(wait=0, warmup=30, active=2, repeat=1),
      record_module_names=True,
      with_stack=True,
      **kwargs,
  )

def setup_environment(config: DotDict):
  torch.set_float32_matmul_precision(config.precision_matmul)

def unique_run_name(wandb_project: str, syllable_count: int):
  api = wandb.Api()
  while True:
    name = random_name(syllable_count)
    if (DATA / "models" / name).exists(): continue
    try:
      api.run(f"{wandb_project}/{name}")
    except wandb.CommError:  # type: ignore
      return name

def set_learning_rate(optimizer: Optimizer, lr: float):
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr
