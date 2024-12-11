// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { SingleBar } from 'cli-progress'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PanoramaSearchDatabase } from './lib/database.js'
import {
  generateIcosphereVertices,
  getIcosphereHaversineDistance,
  latLonToVec3,
  vec3ToLatLon,
} from './lib/icosphere.js'
import { searchPanorama } from './lib/streetview.js'
import { DATA, exportToPCD, shuffle } from './lib/utils.js'

const subdivisions = 5

const searchDB = PanoramaSearchDatabase.open(join(DATA, 'streetview.sqlite3'))

async function searchWithCache(lat: number, lon: number, radius: number) {
  const cache = searchDB.select({ lat, lon, radius, options: {} })
  if (cache) return cache
  const response = await searchPanorama(lat, lon, radius)
  searchDB.insert({ lat, lon, radius, options: {} }, response)
  return response
}

const baseRadius = getIcosphereHaversineDistance(subdivisions)
let locations = generateIcosphereVertices(subdivisions).map(vec3ToLatLon)
let results: Result[] = []

interface Result {
  lat: number
  lon: number
  covered: boolean
}

shuffle(locations)

const bar = new SingleBar({})
bar.start(locations.length, 0)
while (locations.length > 0) {
  const { lat, lon } = locations.pop()!
  const radius = baseRadius * 1.1
  const response = await searchWithCache(lat, lon, radius)
  const parsed = response.parse()
  // results.push({ lat, lon, covered: !!parsed })
  results.push({ lat: parsed?.lat ?? lat, lon: parsed?.lon ?? lon, covered: !!parsed })
  bar.increment()
}

bar.stop()

console.log(`Covered: ${results.filter((r) => r.covered).length} / ${results.length}`)

writeFileSync(
  'output.pcd',
  exportToPCD(
    results.map(({ lat, lon, covered }) => ({
      ...latLonToVec3(lat, lon),
      color: covered ? 0x63d676 : 0x3c7fde, // Green for covered, Blue for uncovered
    })),
  ),
)
