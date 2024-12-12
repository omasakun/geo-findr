// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

// while true; do npx tsx --max-old-space-size=16000 src/coverage.ts || true; done

import { SingleBar } from 'cli-progress'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createFetchClient } from './lib/client.js'
import { PanoramaSearchDatabase } from './lib/database.js'
import {
  generateIcosahedron,
  getIcosphereHaversineDistance,
  subdivideIcosphere,
  Vec3,
  vec3ToLatLon,
} from './lib/icosphere.js'
import { searchPanorama } from './lib/streetview.js'
import { DATA, exportToPCD } from './lib/utils.js'

const RADIUS_MARGIN = 1.2
const VERIFY = false // Verify that the child is indeed uncovered. Adjust the radius margin if errors occur
const SUBDIVISIONS = 20
const client = createFetchClient({ concurrencyLimit: 100, retryLimit: 4 })
// const TORS = 8
// const client = createTorClient((i) => `socks5h://${i}:pass@localhost:${19000 + (i % TORS)}`, {
//   clients: TORS * 10,
//   concurrencyLimit: TORS * 10 * 100,
//   retryLimit: 4,
// })

const searchDB = PanoramaSearchDatabase.open(join(DATA, 'streetview.sqlite3'))

async function searchWithCache(lat: number, lon: number, radius: number, searchThirdParty = false) {
  const cache = searchDB.select({ lat, lon, radius, options: { searchThirdParty } })
  if (cache) return cache
  const response = await searchPanorama(lat, lon, radius, { client, searchThirdParty })
  searchDB.insert({ lat, lon, radius, options: { searchThirdParty } }, response)
  const parsed = response.parse()
  return { response, pano_id: parsed?.id, pano_lat: parsed?.lat, pano_lon: parsed?.lon }
}

async function isCovered(lat: number, lon: number, radius: number) {
  const response = await searchWithCache(lat, lon, radius)
  return !!response.pano_id
}

interface Result extends Vec3 {
  lat: number
  lon: number
  covered: boolean
  error: boolean
}

let [baseVertices, faces] = generateIcosahedron()
let radius = getIcosphereHaversineDistance(0) * RADIUS_MARGIN
let vertices: Result[] = baseVertices.map((v) => ({
  ...v,
  ...vec3ToLatLon(v),
  covered: false,
  error: false,
}))
await Promise.all(vertices.map(async (v) => (v.covered = await isCovered(v.lat, v.lon, radius))))

console.log(
  `Subdivision #0 (radius: ${(radius / 1000).toFixed(2)} km, ${vertices.length} vertices)`,
)

const bar = new SingleBar({})
for (let subdivisions = 1; subdivisions <= SUBDIVISIONS; subdivisions++) {
  let radius = getIcosphereHaversineDistance(subdivisions) * RADIUS_MARGIN

  console.log(
    `Subdivision #${subdivisions} (radius: ${(radius / 1000).toFixed(2)} km, previously: ${vertices.length} vertices)`,
  )

  // 0. Remove the faces that are not covered
  // This is an optimization to reduce the memory usage
  if (!VERIFY) faces = faces.filter((face) => face.some((v) => vertices[v].covered))

  // 1. Remember the parent vertices to update them later
  const parentsTodo = vertices.filter((v) => v.covered)

  // 2. Subdivide
  bar.start(Infinity, 0)
  let tasks: Promise<unknown>[] = []
  faces = subdivideIcosphere(vertices, faces, (v, parent1, parent2) => {
    const child: Result = { ...v, ...vec3ToLatLon(v), covered: false, error: false }
    if (!parent1.covered || !parent2.covered) {
      // If any of the parents are uncovered, the child is also uncovered
      // This is an optimization to reduce the number of API calls

      // Verify that the child is indeed uncovered
      if (VERIFY) {
        tasks.push(
          isCovered(child.lat, child.lon, radius).then((covered) => {
            child.covered = covered
            if (covered) {
              child.error = true
              console.error('Child is covered but parents are not')
            }
            bar.increment()
          }),
        )
      }
    } else {
      tasks.push(
        isCovered(child.lat, child.lon, radius).then((covered) => {
          child.covered = covered
          bar.increment()
        }),
      )
    }
    return child
  })

  bar.setTotal(tasks.length)
  await Promise.all(tasks)
  bar.stop()

  // 3. Update the parent vertices
  bar.start(parentsTodo.length, 0)
  await Promise.all(
    parentsTodo.map(async (v) =>
      isCovered(v.lat, v.lon, radius).then((covered) => {
        v.covered = covered
        bar.increment()
      }),
    ),
  )
  bar.stop()

  writeFileSync(
    'coverage.pcd',
    exportToPCD(
      vertices.map((v) => ({
        ...v,
        color: v.error ? 0xff0000 : v.covered ? 0x63d676 : 0x3c7fde, // Green for covered, Blue for uncovered
      })),
    ),
  )
}
