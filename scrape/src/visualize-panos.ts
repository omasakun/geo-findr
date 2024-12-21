// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

// while true; do npx tsx --max-old-space-size=16000 src/coverage.ts || true; done

import { SingleBar } from 'cli-progress'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PanoramaSearchDatabase } from './lib/database.js'
import { latLonToVec3 } from './lib/icosphere.js'
import { DATA, exportToPCD } from './lib/utils.js'

const searchDB = PanoramaSearchDatabase.open(join(DATA, 'streetview.sqlite3'))

const panos = new Map<string, { x: number; y: number; z: number; color: number }>()

const bar = new SingleBar({ etaBuffer: 10000 })
bar.start(searchDB.count(), 0)
for (const { response } of searchDB.iterateAll()) {
  bar.increment()
  if (response.isFound()) {
    const metadata = response.parse()
    if (metadata) {
      const { id, lat, lon } = metadata
      panos.set(id, { ...latLonToVec3(lat, lon), color: 0x63d676 })
    }
  }
}
bar.stop()

console.log(`Found ${panos.size} panoramas`)
writeFileSync(join(DATA, 'panoloc.pcd'), exportToPCD(Array.from(panos.values())))
