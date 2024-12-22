// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import archiver, { Archiver } from 'archiver'
import { createHash } from 'node:crypto'
import { createWriteStream } from 'node:fs'
import { access, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { Lock, makeDirectoryForFile } from './utils.js'

interface Entry {
  [extension: string]: Buffer | string
}

export class WebDatasetWriter {
  private pattern: string
  private maxShardSize: number
  private currentShardIndex = 0
  private currentShardSize = 0
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

    if (!this.pack || this.currentShardSize > this.maxShardSize) {
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
      this.currentShardSize = 0

      // console.log(`Opened new shard: ${filename}`)
    }

    for (let [extension, content] of Object.entries(entries)) {
      const filename = `${key}.${extension}`

      if (typeof content === 'string') {
        content = Buffer.from(content)
      }

      this.pack!.append(content, { name: filename })
      this.currentShardSize += content.length
    }

    this.lock.release()
  }

  async close(): Promise<void> {
    if (this.pack) {
      await this.pack.finalize()
    }
  }

  allShards(): string[] {
    const shards: string[] = []
    for (let i = 0; i < this.currentShardIndex; i++) {
      shards.push(this.pattern.replace(/%(0\d+)d/, String(i).padStart(6, '0')))
    }
    return shards
  }
}

export class FileDataset {
  private directory: string

  constructor(directory: string) {
    this.directory = directory
  }

  private _getPrefix(key: string): string {
    const keyHash = createHash('sha256').update(key).digest('hex')
    return join(this.directory, keyHash.substring(0, 2), keyHash.substring(2, 4))
  }

  async addEntry(key: string, entries: Entry): Promise<void> {
    const prefix = this._getPrefix(key)

    for (let [extension, content] of Object.entries(entries)) {
      const filePath = join(prefix, `${key}.${extension}`)
      makeDirectoryForFile(filePath)

      await writeFile(filePath, content)
    }
  }

  async hasEntry(key: string, entryKeys: string[]): Promise<boolean> {
    const prefix = this._getPrefix(key)

    for (let extension of entryKeys) {
      const filePath = join(prefix, `${key}.${extension}`)
      try {
        await access(filePath)
      } catch {
        return false
      }
    }
    return true
  }

  async getEntry(key: string, entryKeys: string[]): Promise<Record<string, Buffer>> {
    const prefix = this._getPrefix(key)

    const entries: Record<string, Buffer> = {}
    for (let extension of entryKeys) {
      const filePath = join(prefix, `${key}.${extension}`)
      entries[extension] = await readFile(filePath)
    }
    return entries
  }

  async close(): Promise<void> {
    // nothing to do
  }
}
