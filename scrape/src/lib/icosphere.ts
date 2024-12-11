// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { encode } from 'msgpackr'
import { haversineDistance } from './utils.js'

export interface Vec3 {
  x: number
  y: number
  z: number
}

function normalize(v: Vec3): Vec3 {
  const length = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
  return { x: v.x / length, y: v.y / length, z: v.z / length }
}

/** v1, v2 and v3 must be normalized */
function subdivide(
  vertices: Vec3[],
  verticesSet: Set<string>,
  v1: Vec3,
  v2: Vec3,
  v3: Vec3,
  depth: number,
) {
  if (depth === 0) {
    for (const v of [v1, v2, v3]) {
      const key = encode(v).toString('base64')
      if (!verticesSet.has(key)) {
        vertices.push(v)
        verticesSet.add(key)
      }
    }
  } else {
    const v12 = normalize({ x: (v1.x + v2.x) / 2, y: (v1.y + v2.y) / 2, z: (v1.z + v2.z) / 2 })
    const v23 = normalize({ x: (v2.x + v3.x) / 2, y: (v2.y + v3.y) / 2, z: (v2.z + v3.z) / 2 })
    const v31 = normalize({ x: (v3.x + v1.x) / 2, y: (v3.y + v1.y) / 2, z: (v3.z + v1.z) / 2 })

    subdivide(vertices, verticesSet, v1, v12, v31, depth - 1)
    subdivide(vertices, verticesSet, v2, v23, v12, depth - 1)
    subdivide(vertices, verticesSet, v3, v31, v23, depth - 1)
    subdivide(vertices, verticesSet, v12, v23, v31, depth - 1)
  }
}

/** Generate vertices of an icosphere */
export function generateIcosphereVertices(subdivisions: number): Vec3[] {
  const phi = (1 + Math.sqrt(5)) / 2
  const baseVertices: Vec3[] = []
  for (const a of [1, -1]) {
    for (const b of [phi, -phi]) {
      baseVertices.push(normalize({ x: a, y: b, z: 0 }))
      baseVertices.push(normalize({ x: 0, y: a, z: b }))
      baseVertices.push(normalize({ x: b, y: 0, z: a }))
    }
  }

  const faces = [
    [0, 1, 2],
    [0, 2, 8],
    [0, 4, 6],
    [0, 6, 1],
    [0, 8, 4],
    [1, 5, 7],
    [1, 6, 5],
    [1, 7, 2],
    [2, 3, 8],
    [2, 7, 3],
    [3, 7, 9],
    [3, 9, 10],
    [3, 10, 8],
    [4, 8, 10],
    [4, 10, 11],
    [4, 11, 6],
    [5, 6, 11],
    [5, 9, 7],
    [5, 11, 9],
    [9, 11, 10],
  ]

  const vertices: Vec3[] = []
  const verticesSet = new Set<string>()

  for (const index of faces) {
    const [v1, v2, v3] = index.map((i) => baseVertices[i])
    subdivide(vertices, verticesSet, v1, v2, v3, subdivisions)
  }

  return vertices
}

/** Get the Haversine distance between two vertices of an icosphere */
export function getIcosphereHaversineDistance(subdivisions: number) {
  const phi = (1 + Math.sqrt(5)) / 2
  const pos1 = normalize({ x: 1, y: phi, z: 0 })
  let pos2 = normalize({ x: 0, y: 1, z: phi })

  for (let i = 0; i < subdivisions; i++) {
    pos2 = normalize({
      x: (pos1.x + pos2.x) / 2,
      y: (pos1.y + pos2.y) / 2,
      z: (pos1.z + pos2.z) / 2,
    })
  }

  const { lat: lat1, lon: lon1 } = vec3ToLatLon(pos1)
  const { lat: lat2, lon: lon2 } = vec3ToLatLon(pos2)
  return haversineDistance(lat1, lon1, lat2, lon2)
}

export function vec3ToLatLon(v: Vec3): { lat: number; lon: number } {
  const lat = Math.asin(v.y) * (180 / Math.PI)
  const lon = Math.atan2(v.z, v.x) * (180 / Math.PI)
  const vv = latLonToVec3(lat, lon)
  return { lat, lon }
}

export function latLonToVec3(lat: number, lon: number): Vec3 {
  const latRad = lat * (Math.PI / 180)
  const lonRad = lon * (Math.PI / 180)
  return normalize({
    x: Math.cos(latRad) * Math.cos(lonRad),
    y: Math.sin(latRad),
    z: Math.cos(latRad) * Math.sin(lonRad),
  })
}

export function exportToOBJ(vertices: Vec3[]): string {
  let objData = ''
  vertices.forEach((v) => {
    objData += `v ${v.x} ${v.y} ${v.z}\n`
  })
  return objData
}

// Check the vertices in Blender
// const icosphereVertices = generateIcosphereVertices(2)
// const latLon = icosphereVertices.map(vec3ToLatLon)
// const objData = exportToOBJ(icosphereVertices)
// console.log(objData)
