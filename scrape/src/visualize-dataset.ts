// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

// while true; do npx tsx --max-old-space-size=16000 src/coverage.ts || true; done

import { SingleBar } from 'cli-progress'
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PanoramaSearchDatabase } from './lib/database.js'
import { latLonToVec3 } from './lib/icosphere.js'
import { DATA, exportToPCD, haversineDistance } from './lib/utils.js'

const searchDB = PanoramaSearchDatabase.open(join(DATA, 'streetview.sqlite3'))
const bar = new SingleBar({ etaBuffer: 10000 })

const trainPanos = new Map<
  string,
  { lat: number; lon: number; x: number; y: number; z: number; color: number }
>()
const validPanos = new Map<
  string,
  { lat: number; lon: number; x: number; y: number; z: number; color: number }
>()

const trainPanoids: string[] = JSON.parse(
  readFileSync(join(DATA, 'panos/train.json'), 'utf-8'),
).panoids

bar.start(trainPanoids.length, 0)
await Promise.all(
  trainPanoids.map(async (panoid) => {
    const meta = searchDB.selectByPanoId(panoid)?.response.parse()
    if (!meta) throw new Error(`Panorama ${panoid} not found in database`)
    trainPanos.set(panoid, {
      lat: meta.lat,
      lon: meta.lon,
      ...latLonToVec3(meta.lat, meta.lon),
      color: 0x63d676,
    })
    bar.increment()
  }),
)
bar.stop()

const validPanoids: string[] = JSON.parse(
  readFileSync(join(DATA, 'panos/valid.json'), 'utf-8'),
).panoids

bar.start(validPanoids.length, 0)
await Promise.all(
  validPanoids.map(async (panoid) => {
    const meta = searchDB.selectByPanoId(panoid)?.response.parse()
    if (!meta) throw new Error(`Panorama ${panoid} not found in database`)
    validPanos.set(panoid, {
      lat: meta.lat,
      lon: meta.lon,
      ...latLonToVec3(meta.lat, meta.lon),
      color: 0xffcf4d,
    })
    bar.increment()
  }),
)
bar.stop()

writeFileSync(
  join(DATA, 'panos/locations.pcd'),
  exportToPCD([...trainPanos.values(), ...validPanos.values()]),
)

// Calculate distances from validPanos to the nearest trainPanos
bar.start(validPanos.size, 0)
const validPanoDistances = Array.from(validPanos.entries()).map(([panoid, validPano]) => {
  let minDistance = Infinity
  for (const trainPano of trainPanos.values()) {
    const dist = haversineDistance(trainPano.lat, trainPano.lon, validPano.lat, validPano.lon)
    if (dist < minDistance) {
      minDistance = dist
    }
  }
  bar.increment()
  return { panoid, minDistance }
})
bar.stop()

// Sort validPanos by the longest distance
validPanoDistances.sort((a, b) => b.minDistance - a.minDistance)

// Output the shortest distance and panoid to the console
validPanoDistances.forEach(({ panoid, minDistance }) => {
  console.log(`Panoid: ${panoid}, Shortest Distance: ${minDistance}`)
})
