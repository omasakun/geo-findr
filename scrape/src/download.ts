// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

// while true; do npx tsx --max-old-space-size=16000 src/coverage.ts || true; done

import { SingleBar } from 'cli-progress'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createFetchClient } from './lib/client.js'
import { PanoramaSearchDatabase } from './lib/database.js'
import {
  generateIcosphereVertices,
  getIcosphereHaversineDistance,
  vec3ToLatLon,
} from './lib/icosphere.js'
import { fetchPanoramaImage, PanoramaMetadata } from './lib/streetview.js'
import { DATA, haversineDistance, makeDirectoryForFile, shuffle } from './lib/utils.js'
import { WebDatasetWriter } from './lib/webdataset.js'

const SUBDIVISIONS = 8
const RADIUS_MARGIN = 1.2
const SHARD_SIZE = 1e9

const searchDB = PanoramaSearchDatabase.open(join(DATA, 'streetview.sqlite3'))
const client = createFetchClient({ concurrencyLimit: 50, retryLimit: 4 })

const panos = new Map<string, PanoramaMetadata>()
const locations = generateIcosphereVertices(SUBDIVISIONS)
const searchRadius = getIcosphereHaversineDistance(SUBDIVISIONS) * RADIUS_MARGIN
const trueRadius = getIcosphereHaversineDistance(SUBDIVISIONS) / 2

const bar = new SingleBar({})
bar.start(locations.length, 0)
for (const location of locations) {
  bar.increment()
  const { lat, lon } = vec3ToLatLon(location)
  const cache = searchDB.select({
    lat,
    lon,
    radius: searchRadius,
    options: { searchThirdParty: false },
  })
  const pano_id = cache?.pano_id
  const pano_lat = cache?.pano_lat
  const pano_lon = cache?.pano_lon
  if (pano_id && pano_lat && pano_lon) {
    const distance = haversineDistance(lat, lon, pano_lat, pano_lon)

    // prevent oversampling of edges
    if (distance > trueRadius) continue

    const response = cache.response.parse()
    if (!response) continue
    panos.set(pano_id, response)
  }
}
bar.stop()

let panoIds = shuffle(Array.from(panos.keys()))
const panoIdsFile = join(DATA, 'panos/panoids.txt')

if (existsSync(panoIdsFile)) {
  console.log('Reading existing panoids...')
  panoIds = readFileSync(panoIdsFile, 'utf-8').split('\n')
  panoIds = panoIds.filter((line) => line)
} else {
  makeDirectoryForFile(panoIdsFile)
  writeFileSync(panoIdsFile, panoIds.join('\n'))
}

bar.start(panos.size, 0)
const webdataset = new WebDatasetWriter(join(DATA, 'panos/%06d.tar'), SHARD_SIZE)
await Promise.all(
  panoIds.map(async (pano_id) => {
    const pano = panos.get(pano_id)!
    let zoom = pano.sizes.findIndex((size) => size.width >= 2048)
    if (zoom === -1) zoom = pano.sizes.length - 1
    const image = await fetchPanoramaImage(pano, { client, zoom })
    await webdataset.addEntry(pano_id, {
      'metadata.json': JSON.stringify(pano),
      'panorama.webp': await image.webp().toBuffer(),
    })
    bar.increment()
  }),
)
await webdataset.close()
bar.stop()
