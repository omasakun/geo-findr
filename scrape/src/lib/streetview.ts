// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sharp from 'sharp'
import { createFetchClient } from './client.js'
import { pDouble, pEnum, pInt, ProtobufValue, serialize } from './protobuf.js'
import { decodeMsgpackGzip, encodeMsgpackGzip } from './utils.js'

const defaultLocale = 'en-US'
const defaultFetch = createFetchClient()

export interface LocalizedText {
  text: string
  language: string
}

export interface PanoramaDate {
  year: number
  month: number
  day?: number
}

export interface CorePanoramaMetadata {
  id: string
  lat: number
  lon: number
  elevation?: number
  /** (0, 90, 180, 270) = (north, east, south, west) */
  heading?: number
  pitch?: number
  roll?: number
  date?: PanoramaDate
}

export interface PanoramaMetadata extends CorePanoramaMetadata {
  sizes: { width: number; height: number }[]
  tileSize: { width: number; height: number }
  countryCode?: string
  address?: LocalizedText
  historical: CorePanoramaMetadata[]
  neighbors: CorePanoramaMetadata[]
}

export class SearchPanoramaResponse {
  constructor(public data: any) {}
  parse(): PanoramaMetadata | null {
    const status = this.data[0][0][0]
    if (status !== 0) return null
    return parseResponse(this.data[0][1])
  }
  encode() {
    return encodeMsgpackGzip(this.data)
  }
  static decode(buffer: Buffer) {
    return new SearchPanoramaResponse(decodeMsgpackGzip(buffer))
  }
}

export class PanoramaMetadataResponse {
  constructor(public data: any) {}
  parse(): PanoramaMetadata {
    return parseResponse(this.data[0][1][0])
  }
  encode() {
    return encodeMsgpackGzip(this.data)
  }
  static decode(buffer: Buffer) {
    return new PanoramaMetadataResponse(decodeMsgpackGzip(buffer))
  }
}

/** Note: This function does not always return the closest panorama. */
export async function searchPanorama(
  lat: number,
  lon: number,
  radius: number,
  { locale = defaultLocale, searchThirdParty = false, client = defaultFetch } = {},
) {
  const url = panoramaSearchUrl(lat, lon, radius, locale, searchThirdParty)
  const text = await client.getText(url)
  const healed = text.match(/callback\((.*)\)/)?.[1] ?? ''
  const parsed = JSON.parse(`[${healed}]`)
  return new SearchPanoramaResponse(parsed)
}

export async function fetchPanoramaMetadata(
  id: string,
  { locale = defaultLocale, client = defaultFetch } = {},
) {
  const url = panoramaMetadataUrl(id, locale)
  const text = await client.getText(url)
  const healed = text.match(/\n(.*)/)?.[1] ?? ''
  const parsed = JSON.parse(`[${healed}]`)
  return new PanoramaMetadataResponse(parsed)
}

export async function fetchPanoramaImage(
  pano: PanoramaMetadata,
  { zoom = 'max' as number | 'max', client = defaultFetch } = {},
) {
  const zoomLevel = zoom === 'max' ? pano.sizes.length - 1 : zoom

  const isThirdParty = isThirdPartyId(pano.id)
  const getUrl = (x: number, y: number) =>
    isThirdParty
      ? `https://lh5.googleusercontent.com/p/${pano.id}=x${x}-y${y}-z${zoomLevel}`
      : `https://streetviewpixels-pa.googleapis.com/v1/tile?panoid=${pano.id}&x=${x}&y=${y}&zoom=${zoomLevel}`

  const width = pano.sizes[zoomLevel].width
  const height = pano.sizes[zoomLevel].height
  const cols = Math.ceil(width / pano.tileSize.width)
  const rows = Math.ceil(height / pano.tileSize.height)
  const tiles = await Promise.all(
    Array.from({ length: cols * rows }, (_, i) =>
      client.getBuffer(getUrl(i % cols, (i / cols) << 0)),
    ),
  )

  return sharp({
    create: {
      width: cols * pano.tileSize.width,
      height: rows * pano.tileSize.height,
      channels: 3,
      background: '#000000',
    },
  })
    .composite(
      tiles.map((tile, i) => ({
        input: tile,
        left: (i % cols) * pano.tileSize.width,
        top: ((i / cols) << 0) * pano.tileSize.height,
      })),
    )
    .resize(width, height)
}

function panoramaSearchUrl(
  lat: number,
  lon: number,
  radius: number,
  locale: string,
  searchThirdParty: boolean,
): string {
  const panoType = searchThirdParty ? 10 : 2 // 2 or 3 or 10 ?
  const [lang, country] = locale.split('-')

  // Based on many packages on GitHub (search for "GeoPhotoService.SingleImageSearch")
  const pb = {
    1: { 1: 'apiv3', 5: 'US', 11: { 1: { 1: false } } },
    2: { 1: { 3: pDouble(lat), 4: pDouble(lon) }, 2: pDouble(radius) },
    3: {
      2: { 1: lang, 2: country },
      9: { 1: pEnum(2) },
      11: {
        1: [{ 1: pEnum(panoType), 2: true, 3: pEnum(2) }],
      },
    },
    4: {
      1: [pEnum(1), pEnum(2), pEnum(3), pEnum(4), pEnum(6), pEnum(8)],
    },
  } satisfies ProtobufValue

  return `https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=${serialize(pb)}&callback=callback`
}

function panoramaMetadataUrl(id: string, locale: string): string {
  const panoType = isThirdPartyId(id) ? 10 : 2
  const [lang, country] = locale.split('-')

  // Based on https://maps.google.com/ requests
  const pb = {
    1: { 1: 'maps_sv.tactile', 11: { 2: { 1: true } } },
    2: { 1: lang, 2: country },
    3: { 1: { 1: pEnum(panoType), 2: id } },
    4: {
      1: [
        pEnum(1),
        pEnum(2),
        pEnum(3),
        pEnum(4),
        pEnum(5),
        pEnum(6),
        pEnum(8),
        pEnum(12),
        pEnum(17),
      ],
      2: { 1: pEnum(1) },
      4: { 1: pInt(48) },
      5: [{ 1: pEnum(1) }, { 1: pEnum(2) }],
      6: [{ 1: pEnum(1) }, { 1: pEnum(2) }],
      9: {
        1: [
          { 1: pEnum(2), 2: true, 3: pEnum(2) },
          { 1: pEnum(2), 2: false, 3: pEnum(3) },
          { 1: pEnum(3), 2: true, 3: pEnum(2) },
          { 1: pEnum(3), 2: false, 3: pEnum(3) },
          { 1: pEnum(8), 2: false, 3: pEnum(3) },
          { 1: pEnum(1), 2: false, 3: pEnum(3) },
          { 1: pEnum(4), 2: false, 3: pEnum(3) },
          { 1: pEnum(10), 2: true, 3: pEnum(2) },
          { 1: pEnum(10), 2: false, 3: pEnum(3) },
        ],
      },
    },
    11: { 3: { 4: true } },
  } satisfies ProtobufValue

  return `https://www.google.com/maps/photometa/v1?authuser=0&hl=${lang}&gl=${country}&pb=${serialize(pb)}`
}

function parseResponse(response: any): PanoramaMetadata {
  const get = (...path: number[]) => path.reduce((x, i) => x?.[i], response) ?? undefined
  const zeroUndef = (x: any) => (x === 0 ? undefined : x)
  const id = get(1, 1)
  const otherDates = new Map<number, PanoramaDate>(
    (get(5, 0, 8) ?? []).map((x: any) => {
      const get = (...path: number[]) => path.reduce((x, i) => x?.[i], x) ?? undefined
      return [x[0], { year: get(1, 0), month: get(1, 1), day: get(1, 2) }]
    }),
  )
  const others = ((get(5, 0, 3, 0) as any[]) ?? []).map((item: any, i: number) => {
    const get = (...path: number[]) => path.reduce((x, i) => x?.[i], item) ?? undefined
    return {
      id: get(0, 1),
      lat: get(2, 0, 2),
      lon: get(2, 0, 3),
      elevation: zeroUndef(get(2, 1, 0)),
      heading: zeroUndef(get(2, 2, 0)),
      pitch: zeroUndef(90 - get(2, 2, 1)),
      roll: get(2, 2, 2),
      date: otherDates.get(i), // TODO: neightbors seems to have the same date as the main panorama
    } satisfies CorePanoramaMetadata
  })

  return {
    id,
    sizes: (get(2, 3, 0) ?? []).map((x: any) => ({ width: x[0][1], height: x[0][0] })),
    tileSize: { width: get(2, 3, 1, 0), height: get(2, 3, 1, 1) },
    address: parseLocalizedText(get(3, 2, 0)),
    lat: get(5, 0, 1, 0, 2),
    lon: get(5, 0, 1, 0, 3),
    elevation: get(5, 0, 1, 1, 0),
    heading: zeroUndef(get(5, 0, 1, 2, 0)),
    pitch: zeroUndef(90 - get(5, 0, 1, 2, 1)),
    roll: zeroUndef(get(5, 0, 1, 2, 2)),
    countryCode: get(5, 0, 1, 4),
    date: { year: get(6, 7, 0), month: get(6, 7, 1), day: get(6, 7, 2) },
    historical: others
      .filter((x, i) => otherDates.has(i) && x.id !== id)
      .sort((a, b) => comparePanoramaDates(a.date!, b.date!)),
    neighbors: others.filter((x, i) => !otherDates.has(i) && x.id !== id),
  }
}

function parseLocalizedText(item: any): LocalizedText | undefined {
  if (!item) return undefined
  return {
    text: item[0],
    language: item[1],
  }
}

export function comparePanoramaDates(a: PanoramaDate, b: PanoramaDate) {
  return a.year - b.year || a.month - b.month || (a.day ?? 0) - (b.day ?? 0)
}

export function isThirdPartyId(id: string): boolean {
  return id.length > 22
}
