// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { fetchPanoramaImage, searchPanorama } from './lib/streetview.js'

const pano = (
  await searchPanorama(32.9664691, -87.1676722, 1000, { searchThirdParty: false })
).parse()!
console.log(`lat: ${pano.lat}, lon: ${pano.lon}, id: ${pano.id}`)

let zoom = pano.sizes.findIndex((size) => size.width >= 4096)
if (zoom === -1) zoom = pano.sizes.length - 1
const image = await fetchPanoramaImage(pano, { zoom })

await image.toFile('pano.jpg')
await image.toFile('pano.webp')
await image.toFile('pano.avif')
