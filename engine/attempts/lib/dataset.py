# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

from random import Random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from einops import rearrange
from huggingface_hub import get_token
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, decode_image
from transformers import ViTImageProcessor
from webdataset import WebDataset

from engine.lib.geo import latlon_to_xyz
from engine.lib.projection import equirectangular_to_planar
from engine.lib.utils import DATA, DotDict

class GeoDatasets:
  def __init__(self, repo: str):
    headers = {"Authorization": f"Bearer {get_token()}"}
    metadata_url = f"https://huggingface.co/datasets/{repo}/resolve/main/metadata.json"

    self.repo = repo
    self.metadata = requests.get(metadata_url, headers=headers).json()

  def __call__(self, split: str, *, resampled=False, shardshuffle=True, cache_size=0) -> WebDataset:
    paths = self.metadata[split]["shards"]
    assert paths is not None, f"Dataset split '{split}' not found"

    shard_urls = [f"https://huggingface.co/datasets/{self.repo}/resolve/main/{path}" for path in paths]
    shard_urls = [f"pipe:curl -s -L -H 'Authorization:Bearer {get_token()}' {url}" for url in shard_urls]

    # remove extension from key names
    def mapper(item: dict):
      item = {k.split('.')[0]: v for k, v in item.items()}
      return item

    cache_dir = DATA / 'cache' / 'webdataset'
    if cache_size != 0: cache_dir.mkdir(parents=True, exist_ok=True)
    else: cache_dir = None

    dataset = WebDataset(shard_urls, shardshuffle=shardshuffle, resampled=resampled, empty_check=False, cache_size=cache_size, cache_dir=cache_dir)
    dataset = dataset.decode()
    dataset = dataset.map(mapper)
    return dataset

class GeoVitDataModule(LightningDataModule):
  def __init__(self, config: DotDict, num_workers: int, cache_size: int):
    super().__init__()
    self.config = config
    self.num_workers = num_workers
    self.cache_size = cache_size
    self.datasets = GeoDatasets(config.dataset)

    transform = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_resize=False)
    assert transform.do_normalize
    assert isinstance(transform.image_mean, list)
    assert isinstance(transform.image_std, list)
    self.image_mean = transform.image_mean
    self.image_std = transform.image_std

  def setup(self, stage=None):
    def mapper(item: dict, split: Literal["train", "valid"]):
      with torch.no_grad():
        image = self.get_image(item, split)
        target = self.get_target(item)
        country = item["metadata"].get("countryCode", "??")
        return image, target, country

    self.train_dataset = self.datasets("train", resampled=True, shardshuffle=True, cache_size=self.cache_size).map(lambda x: mapper(x, "train"))
    self.valid_dataset = self.datasets("valid", resampled=False, shardshuffle=False, cache_size=self.cache_size).map(lambda x: mapper(x, "valid"))

  def get_image(self, item: dict, split: Literal["train", "valid"]):
    fov, heading, pitch, roll = 90, 0, 0, 0
    if split == "train":
      random = Random()
      config: dict = self.config.panorama_crop
      if "fov" in config: fov = random.uniform(*config["fov"])
      if "heading" in config: heading = random.uniform(*config["heading"])
      if "pitch" in config: pitch = random.uniform(*config["pitch"])
      if "roll" in config: roll = random.uniform(*config["roll"])

    width, height = self.config.get("image_size", (224, 224))

    image = item["panorama"]
    image = decode_image(bytes_to_tensor(image), ImageReadMode.RGB).to(torch.float32) / 255
    image = equirectangular_to_planar(image, width, height, fov, heading, pitch, roll)
    image[0] = (image[0] - self.image_mean[0]) / self.image_std[0]
    image[1] = (image[1] - self.image_mean[1]) / self.image_std[1]
    image[2] = (image[2] - self.image_mean[2]) / self.image_std[2]
    return image

  def get_target(self, item: dict):
    meta = item["metadata"]
    return torch.as_tensor([meta['lat'], meta['lon']])

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.valid_dataset, batch_size=self.config.batch_size, num_workers=self.num_workers)

  def preview(self, batch):
    images, targets, countries = batch
    for image, target, country in zip(images, targets, countries):
      lat, lon = target
      plt.title(f"{country} ({lat:.6f}, {lon:.6f})")
      plt.imshow(rearrange(image / 2 + 0.5, "c h w -> h w c"))
      plt.show()

class GeoVitXyzDataModule(GeoVitDataModule):
  def get_target(self, item: dict):
    meta = item["metadata"]
    return torch.as_tensor(latlon_to_xyz(meta['lat'], meta['lon']))

  def preview(self, batch):
    images, targets, countries = batch
    for image, target, country in zip(images, targets, countries):
      plt.title(f"{country}")
      plt.imshow(rearrange(image / 2 + 0.5, "c h w -> h w c"))
      plt.show()

class GeoVitXyzCudaDataModule(GeoVitXyzDataModule):
  def get_image(self, item: dict, split: Literal["train", "valid"]):
    image = item["panorama"]
    image = decode_image(bytes_to_tensor(image), ImageReadMode.RGB).to(torch.float32) / 255
    if split == "train" and self.config.randomize_heading:
      random = Random()
      image = image.roll(random.randint(0, image.size(1)), 2)
    return image, split

  def on_after_batch_transfer(self, batch, dataloader_idx):
    with torch.no_grad():
      width, height = self.config.get("image_size", (224, 224))

      (images, split), targets, countries = batch

      # TODO: randomize for each image
      fov, heading, pitch, roll = 90, 0, 0, 0
      if split[0] == "train":
        random = Random()
        config: dict = self.config.panorama_crop
        if "fov" in config: fov = random.uniform(*config["fov"])
        if "heading" in config: heading = random.uniform(*config["heading"])
        if "pitch" in config: pitch = random.uniform(*config["pitch"])
        if "roll" in config: roll = random.uniform(*config["roll"])

      images = equirectangular_to_planar(images, width, height, fov, heading, pitch, roll)
      images[:, 0] = (images[:, 0] - self.image_mean[0]) / self.image_std[0]
      images[:, 1] = (images[:, 1] - self.image_mean[1]) / self.image_std[1]
      images[:, 2] = (images[:, 2] - self.image_mean[2]) / self.image_std[2]
      batch = images, targets, countries

      return super().on_after_batch_transfer(batch, dataloader_idx)

def panorama_examples(repo="geoguess-ai/panorama-div9", split="train"):
  geo_datasets = GeoDatasets(repo)
  dataset = geo_datasets(split)
  for item in dataset:
    pano = item["panorama"]
    pano = decode_image(bytes_to_tensor(pano), ImageReadMode.RGB)
    meta = item["metadata"]
    country = meta.get("countryCode", "??")
    text = f"{country} {meta['lat']:.6f} {meta['lon']:.6f}"
    yield pano, text

def bytes_to_tensor(data: bytes):
  x = np.frombuffer(data, dtype=np.uint8)
  if not x.flags.writeable: x = x.copy()
  return torch.from_numpy(x)

def _main():
  import matplotlib.pyplot as plt
  for pano, text in panorama_examples():
    plt.title(text)
    plt.imshow(rearrange(pano, "c h w -> h w c"))
    plt.show()

if __name__ == "__main__": _main()
