// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { SingleBar } from 'cli-progress'
import { readdirSync } from 'node:fs'
import { join } from 'node:path'
import sharp from 'sharp'
import { DATA } from './lib/utils.js'

const PANOS_DIR = join(DATA, 'panos')

function getAllFiles(dir: string, bar: SingleBar) {
  let results: string[] = []
  const list = readdirSync(dir, { withFileTypes: true })
  bar.increment()
  list.forEach((file) => {
    const filePath = join(dir, file.name)
    if (file.isDirectory()) {
      results.push(...getAllFiles(filePath, bar))
    } else if (filePath.endsWith('.webp')) {
      results.push(filePath)
    }
  })
  return results
}

const bar = new SingleBar({ etaBuffer: 10000 })

bar.start(Infinity, 0)
const files = getAllFiles(PANOS_DIR, bar)
bar.stop()

bar.start(files.length, 0)

for (const filePath of files) {
  bar.increment()
  try {
    await sharp(filePath).metadata()
  } catch (error) {
    console.log()
    console.log(`Corrupted file: ${filePath}`)
    console.log()
  }
}

bar.stop()
