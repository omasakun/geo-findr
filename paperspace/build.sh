#!/bin/bash

# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -eu

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(dirname "$DIR")"

REVISION="geoguess/trainer:$(date +%Y%m%d-%H%M%S-%Z)"

NAME="basic"
DOCKERFILE="$DIR/$NAME/Dockerfile"

if [ ! -f "$DOCKERFILE" ]; then
  echo "Dockerfile not found: $DOCKERFILE"
  exit 1
fi

echo "Building docker image '$REVISION'"

pdm run export
cp "$ROOT/.tool-versions" "$DIR"
cp "$ROOT/requirements.txt" "$DIR"

# host.docker.internal : https://github.com/docker/for-win/issues/6736#issuecomment-630044405
BUILD_EXIT_CODE=0
docker image build -t "$REVISION" "$DIR" -f "$DOCKERFILE" --build-arg REVISION="$REVISION" || BUILD_EXIT_CODE=$?

echo "Cleaning up"
kill %1 || true
rm "$DIR/.tool-versions"
rm "$DIR/requirements.txt"

if [ "$BUILD_EXIT_CODE" -ne 0 ]; then
  echo "Build failed"
  exit 1
fi

echo "Pushing docker image '$REVISION'"
docker image push "$REVISION"

echo "Tagging as latest"
docker image tag "$REVISION" "geoguess/trainer:latest"
docker image push "geoguess/trainer:latest"

echo "Done: $REVISION (https://hub.docker.com/r/geoguess/trainer)"
