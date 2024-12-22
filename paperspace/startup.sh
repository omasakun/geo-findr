#!/bin/bash

# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Set the git user

git config --global user.name o137
git config --global user.email git-user@o137.net

# Update the git repository and install the package

git pull --all || echo 'Git pull failed'
pip install --no-dependencies -e /notebooks || echo 'Pip install failed'
npm install || echo 'Npm install failed'

# Persist the secrets and history

function persist() {
  src="$1"
  dest="/notebooks/paperspace/secrets/$2"
  echo "Persisting $src to $dest"
  rm -f "$src"
  mkdir -p "$(dirname "$src")"
  mkdir -p "$(dirname "$dest")"
  touch "$dest"
  ln -s "$dest" "$src"
}

persist ~/.netrc netrc
persist ~/.bash_history bash_history
persist ~/.cache/huggingface/stored_tokens huggingface/stored_tokens
persist ~/.cache/huggingface/token huggingface/token
persist ~/.vscode/cli/code_tunnel.json vscode/cli/code_tunnel.json
persist ~/.vscode/cli/token.json vscode/cli/token.json

mkdir -p /tmp/data/cache
