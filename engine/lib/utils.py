# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import ast
import json
import logging
import os
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from time import monotonic_ns
from typing import Any, Iterable, TypeVar, Union

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from torch import Tensor
from torch import device as TorchDevice

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"
TEMP_DIR = DATA / "temp"  # This can be None if you want to use the system default temp directory

T = TypeVar("T")

NPArray = NDArray[Any]
Device = Union[TorchDevice, str]

def pick(d: dict, *keys):
  return {k: v for k, v in d.items() if k in keys}

def omit(d: dict, *keys):
  return {k: v for k, v in d.items() if k not in keys}

def indent(text: str, spaces: int):
  return "\n".join(" " * spaces + line for line in text.splitlines())

def get_module_docstring(script: Path):
  if not script.exists(): return None
  with script.open('r', encoding='utf-8') as f:
    tree = ast.parse(f.read())
    return ast.get_docstring(tree)

def random_name(syllable_count: int) -> str:
  # https://www.reddit.com/r/tokipona/wiki/phonology_and_orthography/#wiki_phonotactics
  if not hasattr(random_name, 'syllables'):
    syllables = []
    for c in "jklmnptsw":
      for v in "aeiou":
        if (c, v) in [("w", "o"), ("w", "u"), ("j", "i"), ("t", "i")]: continue
        syllables.append(c + v)
    random_name.syllables = syllables  # type: ignore
  syllables = random_name.syllables  # type: ignore

  name = ""
  random = Random()
  for _ in range(syllable_count):
    name += random.choice(syllables)
  return name

@contextmanager
def timer(desc: str = "Duration"):
  start = monotonic_ns()
  yield
  end = monotonic_ns()
  print(f"{desc}: {(end - start) / 1e6:.3f} ms")

@contextmanager
def hide_warnings():
  with change_loglevel("transformers", logging.ERROR):
    with change_loglevel("speechbrain", logging.ERROR):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

@contextmanager
def change_loglevel(logger: str, level: int):
  prev_level = logging.getLogger(logger).level
  logging.getLogger(logger).setLevel(level)
  try:
    yield
  finally:
    logging.getLogger(logger).setLevel(prev_level)

class DotDict(dict):
  def __getattr__(self, key):
    val = self[key]
    return DotDict(val) if type(val) is dict else val

  def as_dict(self):
    return dict(self)

  def as_flat_dict(self, prefix="", sep="."):
    items = []
    for k, v in self.items():
      if type(v) is dict:
        items.extend(DotDict(v).as_flat_dict(prefix + k + sep, sep).items())
      else:
        items.append((prefix + k, v))
    return dict(items)

def mean(items: Iterable, start=0):
  items = list(items)
  return sum(items, start) / len(items)

def log_clamp(x: Tensor):
  eps = torch.finfo(x.dtype).eps
  return (x + eps).log()

def num_workers_suggested() -> int:
  # https://github.com/pytorch/pytorch/blob/0e6eee3c898c293efc3f172180e9f4d79cc0b13f/torch/utils/data/dataloader.py#L486
  if hasattr(os, 'sched_getaffinity'):
    try:
      return len(os.sched_getaffinity(0))
    except Exception:
      pass
  return os.cpu_count() or 1

def save_json(obj: dict | DotDict, path: str | Path, exist_ok=False):
  if isinstance(obj, DotDict): obj = obj.as_dict()
  with safe_save(path, "w", exist_ok) as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str | Path):
  with open(path, "r") as f:
    return json.load(f)

def save_yaml(obj: dict | DotDict, path: str | Path, exist_ok=False):
  if isinstance(obj, DotDict): obj = obj.as_dict()
  with safe_save(path, "w", exist_ok) as f:
    yaml.safe_dump(obj, f, allow_unicode=True)

def load_yaml(path: str | Path):
  with open(path, "r") as f:
    return yaml.safe_load(f)

def save_numpy(obj: NPArray, path: str | Path, exist_ok=False):
  with safe_save(path, "wb", exist_ok) as f:
    np.save(f, obj)

def save_bytes(obj: bytes, path: str | Path, exist_ok=False):
  with safe_save(path, "wb", exist_ok) as f:
    f.write(obj)

def save_torch(obj: Tensor, path: str | Path, exist_ok=False):
  with safe_save(path, "wb", exist_ok) as f:
    torch.save(obj, f)

@contextmanager
def safe_save(path: str | Path, mode: str, exist_ok=False):
  path = Path(path)

  if not exist_ok and path.exists(): raise FileExistsError(f"{path} already exists")

  name = None
  moved = False
  try:
    # 1. write to temp file
    if TEMP_DIR: TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode, suffix=path.suffix, dir=TEMP_DIR, delete=False) as f:
      name = f.name
      yield f
      f.flush()

    path.parent.mkdir(parents=True, exist_ok=True)

    # 2. move to destination
    if not exist_ok and path.exists(): raise FileExistsError(f"{path} already exists")
    try:
      Path(name).replace(path)
    except OSError as e:
      if e.errno == 18:
        warnings.warn(f"Atomic save failed, falling back to non-atomic save: {path}")
        shutil.copyfile(name, path)
        Path(name).unlink()
    moved = True
  finally:
    if not moved and name: Path(name).unlink()
