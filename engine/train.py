# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import importlib
import sys
from dataclasses import dataclass
from typing import Optional

import click
import questionary

from engine.lib.utils import ROOT, DotDict, load_yaml

@dataclass
class TrainContext:
  config: DotDict

@click.command(context_settings=dict(allow_interspersed_args=False, show_default=True))
@click.argument('config_name', default=None, required=False)
@click.argument('train_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def main(ctx: click.Context, config_name: Optional[str], train_args: list[str]):
  if config_name is None:
    config_name = questionary.select("Select a config to train with", choices=sorted([p.stem for p in (ROOT / 'configs').glob('*.yaml')])).ask()

  if config_name is None:
    print("No config selected")
    sys.exit(1)

  print(f"Training with '{config_name}' config")

  config_path = ROOT / 'configs' / f"{config_name}.yaml"
  config = DotDict(load_yaml(config_path))

  attempt_name = config.attempt
  if attempt_name is None:
    print("Attempt name not found in config. Aborting.")
    sys.exit(1)

  module = importlib.import_module(f"engine.attempts.{attempt_name}")
  context: click.Context = module.train.make_context(f"{ctx.info_name} {attempt_name}", list(train_args))
  context.params['ctx'] = TrainContext(config)
  module.train.invoke(context)

if __name__ == "__main__":
  main()
