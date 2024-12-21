// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { SingleBar } from 'cli-progress'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createFetchClient, createTorClient } from './lib/client.js'
import { PanoramaSearchDatabase } from './lib/database.js'
import {
  generateIcosphereVertices,
  getIcosphereHaversineDistance,
  vec3ToLatLon,
} from './lib/icosphere.js'
import { fetchPanoramaImage, PanoramaMetadata, searchPanorama } from './lib/streetview.js'
import {
  assert,
  DATA,
  haversineDistance,
  makeDirectoryForFile,
  randomGeoCoordinate,
  shuffle,
} from './lib/utils.js'
import { FileDataset } from './lib/webdataset.js'

const RADIUS_MARGIN = 1.2

const SUBDIVISIONS = 9
const VALID_SIZE = 10000

console.log("Run this script after 'coverage.ts'")
console.log()

const searchDB = PanoramaSearchDatabase.open(join(DATA, 'streetview.sqlite3'))

const TORS = 8
const searchClient = createFetchClient({ concurrencyLimit: 100, retryLimit: 4 })
const downloadClient = createTorClient(
  (i) => `socks5h://${i}:pass@localhost:${19000 + (i % TORS)}`,
  {
    clients: TORS * 20,
    concurrencyLimit: TORS * 200,
    retryLimit: 8,
  },
)

const baseRadius = getIcosphereHaversineDistance(SUBDIVISIONS)
const searchRadius = baseRadius * RADIUS_MARGIN
const filterRadius = baseRadius / 2
const valSearchRadius = getIcosphereHaversineDistance(SUBDIVISIONS + 2) * RADIUS_MARGIN
const minimumTrainValidDistance = 100

console.log(`Search radius: ${searchRadius}`)
console.log(`Filter radius: ${filterRadius}`)
console.log(`Validation search radius: ${valSearchRadius}`)
console.log(`Minimum train-validation distance: ${minimumTrainValidDistance}`)
console.log()

console.log('Checking locations with panorama')

const trainPanos = new Map<string, PanoramaMetadata>()
const locations = generateIcosphereVertices(SUBDIVISIONS)

const bar = new SingleBar({ etaBuffer: 10000 })
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
    if (distance > filterRadius) continue

    const response = cache.response.parse()
    if (!response) continue
    trainPanos.set(pano_id, response)
  }
}
bar.stop()

console.log('Downloading train set')

function saveMetadata(filePath: string, data: object) {
  makeDirectoryForFile(filePath)
  writeFileSync(filePath, JSON.stringify(data, null, 2))
}

async function downloadPanoramas(
  panoIds: string[],
  panos: Map<string, PanoramaMetadata>,
  dataset: FileDataset,
  client: any,
  bar: SingleBar,
) {
  bar.start(panoIds.length, 0)
  await Promise.all(
    panoIds.map(async (pano_id) => {
      if (await dataset.hasEntry(pano_id, ['metadata.json', 'panorama.webp'])) {
        bar.increment()
        return
      }

      const pano = panos.get(pano_id)!
      let zoom = pano.sizes.findIndex((size) => size.width >= 2048)
      if (zoom === -1) zoom = pano.sizes.length - 1
      const image = await fetchPanoramaImage(pano, { client, zoom })
      await dataset.addEntry(pano_id, {
        'metadata.json': JSON.stringify(pano),
        'panorama.webp': await image.webp().toBuffer(),
      })
      bar.increment()
    }),
  )
  bar.stop()
}

const trainPanoIds = shuffle(Array.from(trainPanos.keys()))
const trainMetadataFile = join(DATA, `panos/train.json`)

saveMetadata(trainMetadataFile, {
  subdivision: SUBDIVISIONS,
  search_radius: searchRadius,
  filter_radius: filterRadius,
  count: trainPanoIds.length,
  panoids: trainPanoIds,
})

const dataset = new FileDataset(join(DATA, 'panos'))
await downloadPanoramas(trainPanoIds, trainPanos, dataset, downloadClient, bar)

console.log('Downloading valid set')

async function searchWithCache(lat: number, lon: number, radius: number, searchThirdParty = false) {
  const cache = searchDB.select({ lat, lon, radius, options: { searchThirdParty } })
  if (cache) return cache
  const response = await searchPanorama(lat, lon, radius, {
    client: searchClient,
    searchThirdParty,
  })
  searchDB.insert({ lat, lon, radius, options: { searchThirdParty } }, response)
  const parsed = response.parse()
  return { response, pano_id: parsed?.id, pano_lat: parsed?.lat, pano_lon: parsed?.lon }
}

const trainPanoIdSet = new Set(trainPanoIds)
const trainPanoArray = Array.from(trainPanos.values())
const validPanos = new Map<string, PanoramaMetadata>()
bar.start(VALID_SIZE, 0)
while (validPanos.size < VALID_SIZE) {
  await Promise.all(
    Array.from({ length: Math.max(100, VALID_SIZE - validPanos.size) }, async () => {
      const [lat, lon] = randomGeoCoordinate()
      const { response, pano_id, pano_lat, pano_lon } = await searchWithCache(
        lat,
        lon,
        valSearchRadius,
      )
      if (!pano_id) return
      if (validPanos.size >= VALID_SIZE) return
      if (trainPanoIdSet.has(pano_id) || validPanos.has(pano_id)) return

      assert(pano_lat && pano_lon, 'pano_lat and pano_lon must be defined')

      const trainDistances = trainPanoArray.reduce(
        (d, trainPano) =>
          Math.min(d, haversineDistance(pano_lat, pano_lon, trainPano.lat, trainPano.lon)),
        Infinity,
      )

      if (trainDistances < minimumTrainValidDistance) return

      validPanos.set(pano_id, response.parse()!)
      bar.increment()
    }),
  )
}

const validMetadataFile = join(DATA, `panos/valid.json`)
saveMetadata(validMetadataFile, {
  count: validPanos.size,
  search_radius: valSearchRadius,
  minimum_train_valid_distance: minimumTrainValidDistance,
  panoids: shuffle(Array.from(validPanos.keys())),
})

await downloadPanoramas(Array.from(validPanos.keys()), validPanos, dataset, downloadClient, bar)
