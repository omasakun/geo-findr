# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%

from pathlib import Path
from subprocess import run
from sys import argv

ROOT = Path(__file__).parent.parent

def main():
  if len(argv) > 1:
    files = argv[1:]
  else:
    files = run(["fdfind", "--type", "file", '--hidden', '--extension', 'py'], capture_output=True, cwd=ROOT, check=True).stdout.decode().splitlines()

  # autoflake: removes unused imports
  print("Running autoflake...")
  run([
      "autoflake", "--in-place", "--remove-all-unused-imports", "--remove-unused-variables", "--remove-duplicate-keys", "--expand-star-imports", "--recursive",
      *files
  ],
      cwd=ROOT,
      check=True)

  # isort: sorts imports
  print("Running isort...")
  run(["isort", *files], cwd=ROOT, check=True)

  # yapf: formats code
  print("Running yapf...")
  run(["yapf", "--recursive", "--in-place", *files], cwd=ROOT, check=True)

if __name__ == "__main__": main()
