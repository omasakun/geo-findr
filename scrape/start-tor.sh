#!/bin/bash

# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DATA="$DIR/../data/tor"

mkdir -p "$DATA"
cd "$DATA" || exit

JOBS=()

for i in {0..7}; do
  mkdir "$i"

  {
    echo "SocksPort $(($i + 19000))"
    echo "DataDirectory $i"
    echo "ClientOnly 1"
    echo "NumEntryGuards 10"
    echo "MaxCircuitDirtiness 60"
    echo "Log notice file log-$i"
  } >"$DATA/$i/torrc"

  tor -f "$i/torrc" &
  JOBS+=($!)
done

sleep 5

echo ""
echo "PIDs:" "${JOBS[@]}"
echo "Press enter to kill all tor instances"
read

kill "${JOBS[@]}"
