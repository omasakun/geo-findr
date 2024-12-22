# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import numpy as np
import requests
import torch
from einops import rearrange
from huggingface_hub import get_token
from webdataset import WebDataset

from engine.lib.utils import DATA

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
      for k, v in item.items():
        if isinstance(v, np.ndarray):
          item[k] = torch.tensor(v)
      return item

    cache_dir = DATA / 'cache' / 'webdataset'
    if cache_size != 0: cache_dir.mkdir(parents=True, exist_ok=True)
    else: cache_dir = None

    dataset = WebDataset(shard_urls, shardshuffle=shardshuffle, resampled=resampled, empty_check=False, cache_size=cache_size, cache_dir=cache_dir)
    dataset = dataset.decode("pil")
    dataset = dataset.map(mapper)
    return dataset

def panorama_examples(repo="geoguess-ai/panorama-div9", split="train"):
  geo_datasets = GeoDatasets(repo)
  dataset = geo_datasets(split)
  for item in dataset:
    pano = item["panorama"]
    pano = rearrange(np.array(pano), "h w c -> c h w")
    meta = item["metadata"]
    country = meta.get("countryCode", "??")
    text = f"{country} {meta['lat']:.6f} {meta['lon']:.6f}"
    yield pano, text

def _main():
  import matplotlib.pyplot as plt
  for pano, text in panorama_examples():
    plt.title(text)
    plt.imshow(rearrange(pano, "c h w -> h w c"))
    plt.show()

if __name__ == "__main__": _main()
