// Copyright 2024 omasakun <omasakun@o137.net>.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import { assert } from './utils.js'

export type ProtobufValue =
  | boolean
  | string
  | ProtoEnum
  | ProtoInt
  | ProtoDouble
  | { [key: string]: ProtobufValue }
  | ProtobufValue[]

/** Represents a value that will be serialized as an enum. */
export class ProtoEnum {
  constructor(public value: number) {}
}

/** Represents a value that will be serialized as an integer. */
export class ProtoInt {
  constructor(public value: number) {}
}

/** Represents a value that will be serialized as a float. */
export class ProtoFloat {
  constructor(public value: number) {}
}

/** Represents a value that will be serialized as a double. */
export class ProtoDouble {
  constructor(public value: number) {}
}

export function pEnum(value: number): ProtoEnum {
  return new ProtoEnum(value)
}

export function pInt(value: number): ProtoInt {
  return new ProtoInt(value)
}

export function pDouble(value: number): ProtoDouble {
  return new ProtoDouble(value)
}

export function serialize(value: ProtobufValue): string {
  return Object.entries(value)
    .flatMap(([k, v]) => serializeField(k, v))
    .join('')
}

function serializeField(key: string, value: ProtobufValue): string[] {
  assert(key.match(/^[0-9]+$/), `Invalid key: ${key}`)
  switch (true) {
    case typeof value === 'boolean':
      return [`!${key}b${value ? 1 : 0}`]
    case typeof value === 'string':
      return [`!${key}s${value}`]
    case value instanceof ProtoEnum:
      return [`!${key}e${value.value}`]
    case value instanceof ProtoInt:
      return [`!${key}i${value.value}`]
    case value instanceof ProtoFloat:
      return [`!${key}f${value.value}`]
    case value instanceof ProtoDouble:
      return [`!${key}d${value.value}`]
    case Array.isArray(value):
      return value.flatMap((v) => serializeField(key, v))
    case typeof value === 'object' && value !== null:
      const children = Object.entries(value).flatMap(([k, v]) => serializeField(k, v))
      return [`!${key}m${children.length}`, ...children]
    default:
      console.error('Cannot serialize', key, value)
      throw new Error(`Cannot serialize ${value}`)
  }
}

export function deserialize(value: string): Record<string, ProtobufValue> {
  const items = value.split('!').slice(1)
  return deserializeItems(items)
}

function deserializeItems(items: string[]): Record<string, ProtobufValue> {
  const result: Record<string, ProtobufValue> = {}

  const put = (key: string, value: ProtobufValue) => {
    if (key in result) {
      if (!Array.isArray(result[key])) {
        result[key] = [result[key], value]
      } else {
        result[key].push(value)
      }
    } else {
      result[key] = value
    }
  }

  while (items.length > 0) {
    const item = items.shift()!
    const [, key, type, value] = item.match(/^([0-9]+)([bifdsme])(.*)$/)!

    switch (type) {
      case 'b':
        put(key, value === '1')
        break
      case 'i':
        put(key, new ProtoInt(parseInt(value, 10)))
        break
      case 'f':
        put(key, new ProtoFloat(parseFloat(value)))
        break
      case 'd':
        put(key, new ProtoDouble(parseFloat(value)))
        break
      case 's':
        put(key, value)
        break
      case 'e':
        put(key, new ProtoEnum(parseInt(value, 10)))
        break
      case 'm':
        const length = parseInt(value, 10)
        put(key, deserializeItems(items.splice(0, length)))
        break
      default:
        console.error('Cannot deserialize', item, key, type, value)
        throw new Error(`Cannot deserialize ${item}`)
    }
  }

  return result
}
