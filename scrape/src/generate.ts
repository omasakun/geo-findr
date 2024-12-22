// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { SingleBar } from 'cli-progress'
import { readFileSync, writeFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import { DATA } from './lib/utils.js'
import { FileDataset, WebDatasetWriter } from './lib/webdataset.js'

interface Metadata {
  subdivision: number
  panoids: string[]
}

async function generateDataset(metadataFile: string, datasetName: string) {
  const metadata: Metadata = JSON.parse(readFileSync(metadataFile, 'utf-8'))
  console.log(`Generating ${datasetName} dataset with ${metadata.panoids.length} panos`)

  const bar = new SingleBar({ etaBuffer: 10000 })
  bar.start(metadata.panoids.length, 0)
  const src = new FileDataset(join(DATA, 'panos'))
  const dest = new WebDatasetWriter(join(DATA, `datasets/${datasetName}/%06d.tar`), 1e9)

  for (const panoid of metadata.panoids) {
    const data = await src.getEntry(panoid, ['metadata.json', 'panorama.webp'])
    await dest.addEntry(panoid, data)
    bar.increment()
  }

  await src.close()
  await dest.close()
  bar.stop()

  return {
    ...metadata,
    shards: dest.allShards().map((shard) => datasetName + '/' + basename(shard)),
  }
}

const trainMetadata = await generateDataset(join(DATA, `panos/train.json`), 'train')
const validMetadata = await generateDataset(join(DATA, `panos/valid.json`), 'valid')

writeFileSync(
  join(DATA, 'datasets/metadata.json'),
  JSON.stringify(
    {
      train: trainMetadata,
      valid: validMetadata,
    },
    null,
    2,
  ),
)
