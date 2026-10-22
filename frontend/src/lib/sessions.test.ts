// frontend/src/lib/sessions.test.ts
// 会话排序工具的测试：最近的放前面。

import { describe, it, expect } from 'vitest'
import { byRecent } from './sessions'
import type { SessionSummary } from '../types'

function sess(id: string, last_active: string): SessionSummary {
  return { id, title: id, created_at: last_active, last_active, turns: 0 }
}

describe('byRecent', () => {
  it('按 last_active 降序（新的在前）', () => {
    const a = sess('a', '2026-10-22T10:00:00')
    const b = sess('b', '2026-10-22T11:00:00')
    const c = sess('c', '2026-10-22T09:00:00')
    expect([a, b, c].sort(byRecent).map((s) => s.id)).toEqual(['b', 'a', 'c'])
  })

  it('中文 ISO 字符串按字典序即位时间顺序', () => {
    const old = sess('o', '2026-10-01T00:00:00')
    const new_ = sess('n', '2026-10-22T00:00:00')
    expect(byRecent(new_, old)).toBeLessThan(0) // 新的排在前面 → 返回负数
  })
})
