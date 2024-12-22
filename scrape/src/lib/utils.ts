// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import dedent from 'dedent'
import * as msgpackr from 'msgpackr'
import { existsSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { gunzipSync, gzipSync } from 'node:zlib'

export const ROOT = new URL('../../..', import.meta.url).pathname
export const DATA = join(ROOT, 'data')

export function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message)
  }
}

export function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>
  for (const key of keys) {
    result[key] = obj[key]
  }
  return result
}

/** Shuffle an array in place */
export function shuffle<T>(array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    const temp = array[i]
    array[i] = array[j]
    array[j] = temp
  }
  return array
}

export function encodeMsgpackGzip(data: unknown): Buffer {
  return gzipSync(msgpackr.encode(data))
}

export function decodeMsgpackGzip<T = unknown>(buffer: Buffer): T {
  return msgpackr.decode(gunzipSync(buffer))
}

export function deterministicJsonStringify(data: unknown): string {
  return JSON.stringify(data, (key, value) => {
    if (typeof value === 'object' && value !== null) {
      return Object.keys(value)
        .sort()
        .reduce(
          (acc, key) => {
            acc[key] = value[key]
            return acc
          },
          {} as Record<string, unknown>,
        )
    }
    return value
  })
}

/** Uniformly distributed over the surface of the sphere */
export function randomGeoCoordinate(): [lat: number, lon: number] {
  const lat = Math.asin(Math.random() * 2 - 1) * (180 / Math.PI)
  const lon = Math.random() * 360 - 180
  return [lat, lon]
}

/** https://en.wikipedia.org/wiki/Haversine_formula */
export function haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  // TODO: Is this the same formula used by Geoguessr?
  const radius = 6378137 // Equatorial radius (https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84)
  lat1 *= Math.PI / 180
  lon1 *= Math.PI / 180
  lat2 *= Math.PI / 180
  lon2 *= Math.PI / 180
  const dLat = lat2 - lat1
  const dLon = lon2 - lon1
  const inner = Math.sin(dLat / 2) ** 2 + Math.sin(dLon / 2) ** 2 * Math.cos(lat1) * Math.cos(lat2)
  return 2 * radius * Math.asin(Math.sqrt(inner))
}

export function exportToPCD(items: { x: number; y: number; z: number; color: number }[]) {
  let output =
    dedent`
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z rgb
    SIZE 4 4 4 4
    TYPE F F F U
    COUNT 1 1 1 1
    WIDTH ${items.length}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS ${items.length}
    DATA ascii
  ` + '\n'
  for (const { x, y, z, color } of items) {
    output += `${x} ${y} ${-z} ${color}\n`
  }
  return output
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export class TaskQueue {
  private concurrencyLimit: number
  private running = 0
  private taskQueue = new Queue<() => Promise<void>>()

  constructor(concurencyLimit: number) {
    this.concurrencyLimit = concurencyLimit
  }

  private next(): void {
    if (this.running < this.concurrencyLimit) {
      const task = this.taskQueue.shift()
      if (task) {
        this.running++
        task().finally(() => {
          this.running--
          this.next()
        })
      }
    }
  }

  add(task: () => Promise<void>): void {
    this.taskQueue.push(task)
    this.next()
  }

  async waitAll(): Promise<void> {
    while (this.taskQueue.length > 0 || this.running > 0) {
      await sleep(100)
    }
  }

  async waitQueue(): Promise<void> {
    while (this.taskQueue.length > 0) {
      await sleep(100)
    }
  }

  get length() {
    return this.taskQueue.length + this.running
  }
}

class Queue<T> {
  private pushStack: T[] = []
  private shiftStack: T[] = []

  push(value: T) {
    this.pushStack.push(value)
  }

  shift() {
    if (this.shiftStack.length === 0) {
      this.shiftStack = this.pushStack.reverse()
      this.pushStack = []
    }
    return this.shiftStack.pop()
  }

  get length() {
    return this.pushStack.length + this.shiftStack.length
  }
}

export class Lock {
  private locked = false
  private waiting: (() => void)[] = []

  async acquire(): Promise<void> {
    if (this.locked) {
      await new Promise<void>((resolve) => {
        this.waiting.push(resolve)
      })
    }
    this.locked = true
  }

  release(): void {
    if (this.waiting.length > 0) {
      const resolve = this.waiting.shift()
      if (resolve) {
        resolve()
      }
    } else {
      this.locked = false
    }
  }
}

export function makeDirectoryForFile(filename: string): void {
  const directory = dirname(filename)

  if (!existsSync(directory)) {
    mkdirSync(directory, { recursive: true })
  }
}
