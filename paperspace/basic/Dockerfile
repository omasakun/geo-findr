# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

# Load ~/.profile
SHELL ["/bin/bash", "-l", "-c"]

# 1st line: CLI apps and project dependencies
# 2nd line: Python dependencies -- https://devguide.python.org/getting-started/setup-building/#linux
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y --no-install-recommends \
  atop curl fd-find file git htop ncdu neovim nnn progress pv ripgrep rsync sudo unzip wget zip \
  build-essential gdb lcov pkg-config libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma lzma-dev tk-dev uuid-dev zlib1g-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install asdf
RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.14.1 && \
  echo "source ~/.asdf/asdf.sh" >>~/.profile

# Install asdf plugins
ADD .tool-versions /root/
RUN asdf plugin add python && asdf install python
RUN asdf plugin add nodejs && asdf install nodejs

# Install vscode
RUN curl -L "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" | tar -xz -C /usr/local/bin

# Install vscode extensions
RUN curl -L "https://github.com/coder/code-server/releases/download/v4.95.3/code-server_4.95.3_amd64.deb" -o ~/code-server.deb && \
  apt-get install -y --no-install-recommends ~/code-server.deb && \
  rm ~/code-server.deb && \
  mkdir -p ~/.vscode-server/extensions && \
  code-server --extensions-dir ~/.vscode-server/extensions \
  --install-extension editorconfig.editorconfig \
  --install-extension esbenp.prettier-vscode \
  --install-extension foxundermoon.shell-format \
  --install-extension gruntfuggly.todo-tree \
  --install-extension mhutchie.git-graph \
  --install-extension ms-python.isort \
  --install-extension ms-python.python \
  --install-extension ms-toolsai.jupyter \
  --install-extension tamasfe.even-better-toml && \
  apt-get remove -y --purge code-server && \
  apt-get autoremove -y && \
  rm -rf ~/.local/share/code-server ~/.config/code-server

# Install python dependencies
ADD requirements.txt /root/
RUN pip --no-cache-dir install --upgrade pip jupyterlab && \
  pip --no-cache-dir install --requirement ~/requirements.txt && \
  rm ~/requirements.txt

ARG REVISION
ENV IMAGE_REVISION=$REVISION

RUN echo "if [ -f /notebooks/paperspace/bashrc.sh ]; then source /notebooks/paperspace/bashrc.sh; fi" >>~/.profile

EXPOSE 8888 6006
ADD boot.sh /root/
CMD bash -l /root/boot.sh
