// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import dedent from 'dedent'
import * as msgpackr from 'msgpackr'
import { join } from 'node:path'
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

export function geoguessrScore(distance: number, mapSize = 14916862): number {
  /*
    https://www.reddit.com/r/geoguessr/comments/zqwgnr/comment/j12rjkq/ by MiraMattie on 2022-12-21 (edited on 2023-06-10)

    There have been lots of discussions. One of the older ones is:
    https://www.reddit.com/r/geoguessr/comments/7ekj80/for_all_my_geoguessing_math_nerds/

    You can look at the code to Chatguessr on github for one of the most impactful estimates, which uses score 5000 * 0.99866017 ^ ( distance in meters / scale), where the scale is determined by the distance between the top left and bottom right corners of the map, divided by 7.458421.

    I hate odd magic numbers, so I did some math. With:
    - s = score
    - d = distance, meters
    - z = map size, meters (provided as maxErrorDistance by the map API that indicates the map's size; chatguessr recalculates it from the bounds; but gets the same number - for the world map, 14916862 if you're using distance in meters, 14916.862 if in KM)
    - m = max score (5000)
    - k1 = exponent base - (0.99866017 in chatguessr)
    - k2 = power factor (7458.421 in chatguessr - well actually they divide the error by this, and then multiply the distance by 1000)

    ... jugging around the code, the general equation they use is:
    s = m * k1 ^ ( k2 * d / z )
    For simplicity, let's divide both sides by m:
    s / m = k1 ^ ( k2 * d / z )
    ... then we can say:
    - Ps = s / m (Percent score)
    - Pd = d / z (Percent distance)
    ... and the equation gets real simple:
    Ps = k1 ^ ( k2 * Pd )
    Take the log:
    ln(Ps) = ln (k1 ^ ( k2 * Pd ) )
    ... so we can pull out the expontant:
    ln(Ps) = k2 * Pd * ln (k1)
    ... Rearrange ever so slightly:
    ln(Ps) = Pd * k2 * ln (k1)
    Now we can observe that since k1 and k2 are constants, k2 * ln(k1) is itself a different constant. So let's call it k, and define it as:
    k = k2 * ln (k1)
    ... making the calculation:
    ln(Ps) = Pd * k
    un-log both sides, and that's:
    Ps = e ^ ( Pd * k )
    Using the values of k1 and k2 in chatguessr, k = -10.0040256448936.

    The value of that constant is just too close to -10 to be a coincidence. So I believe Geoguessr's score calculation uses k = -10; plugging that in and backing out the substitutions, then:
    s = 5000 * e ^ ( -10 * d / z )
    ... and now we have a nice, simple formula with no weird high-precision constants needed to calculate the score.

    There are lots of smart geoguessers and it wouldn't surprise me if someone has derived it before, but it's the first time I've seen a precise calculation with a single whole number as a constant.
  */
  return 5000 * Math.exp((-10000 * distance) / mapSize)
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
