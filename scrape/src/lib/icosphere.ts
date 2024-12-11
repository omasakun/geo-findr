// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { haversineDistance } from './utils.js'

export interface Vec3 {
  x: number
  y: number
  z: number
}

export type Face = [number, number, number]

function normalize(v: Vec3): Vec3 {
  const length = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
  return { x: v.x / length, y: v.y / length, z: v.z / length }
}

export function generateIcosahedron(): [vertices: Vec3[], faces: Face[]] {
  const phi = (1 + Math.sqrt(5)) / 2

  const vertices: Vec3[] = []
  for (const a of [1, -1]) {
    for (const b of [phi, -phi]) {
      vertices.push(normalize({ x: a, y: b, z: 0 }))
      vertices.push(normalize({ x: 0, y: a, z: b }))
      vertices.push(normalize({ x: b, y: 0, z: a }))
    }
  }

  const faces: Face[] = [
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

  return [vertices, faces]
}

/** Subdivide an icosphere. new vertices will be pushed to the vertices array */
export function subdivideIcosphere<T extends Vec3>(
  vertices: T[],
  faces: Face[],
  mapper: (v: Vec3, parent1: T, parent2: T) => T,
): Face[] {
  const newVerticesMap = new Map<string, number>()
  const newFaces: Face[] = []

  // Get or create a child vertex
  const getChild = (p1: number, p2: number) => {
    const key = [p1, p2].sort().join(',')
    if (!newVerticesMap.has(key)) {
      const v1 = vertices[p1]
      const v2 = vertices[p2]
      const pos = normalize({ x: (v1.x + v2.x) / 2, y: (v1.y + v2.y) / 2, z: (v1.z + v2.z) / 2 })
      const v12 = mapper(pos, v1, v2)
      newVerticesMap.set(key, vertices.length)
      vertices.push(v12)
    }
    return newVerticesMap.get(key)!
  }

  for (const face of faces) {
    const v01 = getChild(face[0], face[1])
    const v12 = getChild(face[1], face[2])
    const v20 = getChild(face[2], face[0])
    newFaces.push([face[0], v01, v20])
    newFaces.push([face[1], v12, v01])
    newFaces.push([face[2], v20, v12])
    newFaces.push([v01, v12, v20])
  }

  return newFaces
}

export function generateIcosphereVertices(subdivisions: number): Vec3[] {
  let [vertices, faces] = generateIcosahedron()
  for (let i = 0; i < subdivisions; i++) {
    faces = subdivideIcosphere(vertices, faces, (v) => v)
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
