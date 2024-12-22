#!/bin/bash

# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

shopt -s histappend
shopt -s checkwinsize

PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"

HISTCONTROL=ignoredups
HISTSIZE=1000000
HISTFILESIZE=2000000

alias fd="fdfind"

echo "Revision: ${IMAGE_REVISION:-unknown}"
TZ="Asia/Tokyo" git log -1 --pretty=format:"Commit: %ad (%ar)%n        %s" --date=iso-local
echo
