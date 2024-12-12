// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { IncomingMessage } from 'node:http'
import https, { RequestOptions } from 'node:https'
import pLimit from 'p-limit'
import { SocksProxyAgent } from 'socks-proxy-agent'
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

  return {
    async getText(url: string) {
      return fetchLimit(() =>
        retry(
          async () => {
            const response = await fetch(url)
            return await response.text()
          },
          { retryLimit, retryDelay },
        ),
      )
    },
    async getBuffer(url: string) {
      return fetchLimit(() =>
        retry(
          async () => {
            const response = await fetch(url)
            const arrayBuffer = await response.arrayBuffer()
            return Buffer.from(arrayBuffer)
          },
          { retryLimit, retryDelay },
        ),
      )
    },
  }
}

export function createTorClient(
  proxyUrl: (client: number) => string,
  { concurrencyLimit = 4, retryLimit = 3, retryDelay = 1000, clients = 10 } = {},
): FetchClient {
  const fetchLimit = pLimit(concurrencyLimit)
  const agents = Array.from({ length: clients }, (_, i) => new SocksProxyAgent(proxyUrl(i)))
  let client = 0

  return {
    async getText(url: string) {
      return this.getBuffer(url).then((buffer) => buffer.toString('utf-8'))
    },
    async getBuffer(url: string) {
      return fetchLimit(() =>
        retry(
          async () => {
            client = (client + 1) % clients
            const agent = agents[client]
            const response = await getHttps(url, { agent })
            const chunks: Buffer[] = []
            for await (const chunk of response) {
              chunks.push(chunk)
            }
            return Buffer.concat(chunks)
          },
          { retryLimit, retryDelay },
        ),
      )
    },
  }
}

function getHttps(url: string, options: RequestOptions = {}): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    https.get(url, options, (res) => resolve(res)).on('error', reject)
  })
}

async function retry<T>(
  fn: () => Promise<T>,
  options: { retryLimit: number; retryDelay: number },
): Promise<T> {
  const { retryLimit, retryDelay } = options
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
