#!/bin/bash

# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Load the latest startup script
if [ -f /notebooks/paperspace/startup.sh ]; then
  source /notebooks/paperspace/startup.sh
fi

# Based on https://github.com/gradient-ai/base-container/blob/b1cffa23de83edece0d64762569abe55131cdafd/pt211-tf215-cudatk120-py311/Dockerfile
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True
