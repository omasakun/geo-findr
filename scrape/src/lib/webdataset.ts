// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import archiver, { Archiver } from 'archiver'
import { createWriteStream } from 'node:fs'
import { Lock, makeDirectoryForFile } from './utils.js'

interface Entry {
  [extension: string]: Buffer | string
}

export class WebDatasetWriter {
  private pattern: string
  private maxShardSize: number
  private currentShardIndex = 0
  private lock = new Lock()
  private pack: Archiver | null = null

  constructor(pattern: string, maxShardSize: number) {
    this.pattern = pattern
    this.maxShardSize = maxShardSize
  }

  private _getShardFilename(): string {
    const match = this.pattern.match(/%(0\d+)d/)
    if (!match) throw new Error('Invalid pattern: missing %0Xd format.')

    const padLength = parseInt(match[1], 10)
    return this.pattern.replace(match[0], String(this.currentShardIndex).padStart(padLength, '0'))
  }

  async addEntry(key: string, entries: Entry): Promise<void> {
    await this.lock.acquire()

    if (!this.pack || this.pack.pointer() > this.maxShardSize) {
      if (this.pack) {
        await this.pack.finalize()
      }

      const filename = this._getShardFilename()
      makeDirectoryForFile(filename)

      this.pack = archiver.create('tar')

      const output = createWriteStream(filename)
      // output.on('close', () => console.log('output file has closed.'))
      // output.on('end', () => console.log('data has been drained'))
      this.pack.pipe(output)

      this.currentShardIndex++

      // console.log(`Opened new shard: ${filename}`)
    }

    for (let [extension, content] of Object.entries(entries)) {
      const filename = `${key}.${extension}`

      if (typeof content === 'string') {
        content = Buffer.from(content)
      }

      this.pack!.append(content, { name: filename })
    }

    this.lock.release()
  }

  async close(): Promise<void> {
    if (this.pack) {
      await this.pack.finalize()
    }
  }
}
