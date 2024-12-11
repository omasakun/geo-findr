// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pLimit from 'p-limit'
import { sleep } from './utils.js'

export interface FetchClient {
  getText(url: string): Promise<string>
  getBuffer(url: string): Promise<Buffer>
}

export function createFetchClient({
  concurrencyLimit = 4,
  retryLimit = 3,
  retryDelay = 1000,
} = {}): FetchClient {
  const fetchLimit = pLimit(concurrencyLimit)

  const retry = async <T>(fn: () => Promise<T>) => {
    let i = 0
    while (true) {
      try {
        return await fn()
      } catch (e) {
        if (i === retryLimit - 1) throw e
      }
      await sleep(retryDelay * 2 ** i)
      i++
    }
  }

  return {
    async getText(url: string) {
      return fetchLimit(() =>
        retry(async () => {
          const response = await fetch(url)
          return await response.text()
        }),
      )
    },
    async getBuffer(url: string) {
      return fetchLimit(() =>
        retry(async () => {
          const response = await fetch(url)
          const arrayBuffer = await response.arrayBuffer()
          return Buffer.from(arrayBuffer)
        }),
      )
    },
  }
}
