// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pLimit from 'p-limit'

export interface FetchClient {
  getText(url: string): Promise<string>
  getBuffer(url: string): Promise<Buffer>
}

export function createFetchClient(concurrencyLimit = 4): FetchClient {
  const fetchLimit = pLimit(concurrencyLimit)
  return {
    async getText(url: string) {
      return fetchLimit(async () => {
        const response = await fetch(url)
        return await response.text()
      })
    },
    async getBuffer(url: string) {
      return fetchLimit(async () => {
        const response = await fetch(url)
        const arrayBuffer = await response.arrayBuffer()
        return Buffer.from(arrayBuffer)
      })
    },
  }
}

export function torFetchClient(torProxies: string[], concurrencyLimit = 4): FetchClient {
  const fetchLimit = pLimit(concurrencyLimit)
  let proxyIndex = 0
  return {
    async getText(url: string) {
      return fetchLimit(async () => {
        const proxy = torProxies[proxyIndex]
        proxyIndex = (proxyIndex + 1) % torProxies.length
        const response = await fetch(url, { agent: new URL(proxy) })
        return await response.text()
      })
    },
    async getBuffer(url: string) {
      return fetchLimit(async () => {
        const proxy = torProxies[proxyIndex]
        proxyIndex = (proxyIndex + 1) % torProxies.length
        const response = await fetch(url, { agent: new URL(proxy) })
        const arrayBuffer = await response.arrayBuffer()
        return Buffer.from(arrayBuffer)
      })
    },
  }
}
