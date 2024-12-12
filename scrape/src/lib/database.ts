// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import Database from 'better-sqlite3'
import { PanoramaMetadataResponse, SearchPanoramaResponse } from './streetview.js'
import { deterministicJsonStringify } from './utils.js'

export interface SearchPanoramaRequest {
  lat: number
  lon: number
  radius: number
  options: any
}

export class PanoramaSearchDatabase {
  constructor(private db: Database.Database) {}

  static open(filename: string, initialize = true) {
    const db = new Database(filename)
    const database = new PanoramaSearchDatabase(db)
    if (initialize) database.initialize()
    return database
  }

  close() {
    this.db.close()
  }

  initialize() {
    this.db.exec(
      `
      CREATE TABLE IF NOT EXISTS panorama_search (
        id INTEGER PRIMARY KEY,
        request TEXT NOT NULL,
        response BLOB NOT NULL,
        pano_id TEXT,
        pano_lat REAL,
        pano_lon REAL
      );
      CREATE INDEX IF NOT EXISTS idx_panorama_search_request ON panorama_search (request);
      `,
    )
  }

  insert(request: SearchPanoramaRequest, response: SearchPanoramaResponse) {
    const pano = response.parse()
    this.db
      .prepare(
        'INSERT INTO panorama_search (request, response, pano_id, pano_lat, pano_lon) VALUES (?, ?, ?, ?, ?)',
      )
      .run(deterministicJsonStringify(request), response.encode(), pano?.id, pano?.lat, pano?.lon)
  }

  select(request: SearchPanoramaRequest): {
    response: SearchPanoramaResponse
    pano_id: string | null
    pano_lat: number | null
    pano_lon: number | null
  } | null {
    const row: any = this.db
      .prepare('SELECT * FROM panorama_search WHERE request = ?')
      .get(deterministicJsonStringify(request))
    if (!row) return null
    return {
      response: SearchPanoramaResponse.decode(row.response),
      pano_id: row.pano_id,
      pano_lat: row.pano_lat,
      pano_lon: row.pano_lon,
    }
  }

  count() {
    return this.db.prepare('SELECT COUNT(*) FROM panorama_search').pluck().get() as number
  }

  *iterateAll(): IterableIterator<{
    request: SearchPanoramaRequest
    response: SearchPanoramaResponse
  }> {
    const rows: Iterable<any> = this.db.prepare('SELECT * FROM panorama_search').iterate()
    for (const row of rows) {
      yield {
        request: JSON.parse(row.request) as SearchPanoramaRequest,
        response: SearchPanoramaResponse.decode(row.response),
      }
    }
  }
}

export class PanoramaMetadataDatabase {
  constructor(private db: Database.Database) {}

  static open(filename: string, initialize = true) {
    const db = new Database(filename)
    const database = new PanoramaMetadataDatabase(db)
    if (initialize) database.initialize()
    return database
  }

  close() {
    this.db.close()
  }

  initialize() {
    this.db.exec(
      `
      CREATE TABLE IF NOT EXISTS panorama_metadata (
        id TEXT PRIMARY KEY,
        locale TEXT NOT NULL,
        response BLOB NOT NULL
      );
      `,
    )
  }

  insert(id: string, locale: string, response: PanoramaMetadataResponse) {
    this.db
      .prepare('INSERT INTO panorama_metadata (id, locale, response) VALUES (?, ?, ?)')
      .run(id, locale, response.encode())
  }

  select(id: string, locale: string): PanoramaMetadataResponse | null {
    const row: any = this.db
      .prepare('SELECT * FROM panorama_metadata WHERE id = ? AND locale = ?')
      .get(id, locale)
    if (!row) return null
    return PanoramaMetadataResponse.decode(row.response)
  }

  count() {
    return this.db.prepare('SELECT COUNT(*) FROM panorama_metadata').pluck().get() as number
  }

  *iterateAll(): IterableIterator<{
    id: string
    locale: string
    response: PanoramaMetadataResponse
  }> {
    const rows: Iterable<any> = this.db.prepare('SELECT * FROM panorama_metadata').iterate()
    for (const row of rows) {
      yield {
        id: row.id,
        locale: row.locale,
        response: PanoramaMetadataResponse.decode(row.response),
      }
    }
  }
}
