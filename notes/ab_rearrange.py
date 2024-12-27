# Copyright 2024 omasakun <omasakun@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# %%

import torch
from einops import rearrange

x = torch.tensor([1, 2, 3, 11, 12, 13])
x = rearrange(x, '(p c) -> p c', p=2)
print(x)
