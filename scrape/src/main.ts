// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { searchPanorama } from './streetview.js'

for (const [lat, lon] of Array.from({ length: 100 }, (_, i) => [
  ((Math.random() - 0.5) * 180) << 0,
  ((Math.random() - 0.5) * 360) << 0,
])) {
  const res = (await searchPanorama(lat, lon, 10000000, { searchThirdParty: false })).parse()
  const pos = `${lat.toString().padStart(4)},${lon.toString().padStart(4)}`
  if (res) {
    const error = `${(res.lat - lat).toFixed(1).padStart(6)},${(res.lon - lon).toFixed(1).padStart(6)}`
    console.log(`${pos} : ${error} : ${res.id}`)
  } else {
    console.log(`${pos} not found`)
  }
}
