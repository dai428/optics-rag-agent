// frontend/src/lib/sessions.test.ts
// 会话排序工具的测试：最近的放前面。

import { describe, it, expect } from 'vitest'
import { byRecent, filterSessions, mapHistory } from './sessions'
import type { SessionSummary } from '../types'

function sess(id: string, last_active: string): SessionSummary {
  return { id, title: id, created_at: last_active, last_active, turns: 0 }
}

describe('会话搜索与历史映射', () => {
  const sessions = [
    { ...sess('ABC-1', ''), title: '光学实验' },
    { ...sess('DEF-2', ''), title: 'React Notes' },
  ]
  it('空白查询保留原顺序且不修改输入', () => {
    expect(filterSessions(sessions, '  ')).toEqual(sessions)
    expect(sessions[0].id).toBe('ABC-1')
  })
  it('中文标题与不区分大小写的 ID / 标题可搜索', () => {
    expect(filterSessions(sessions, '光学')).toEqual([sessions[0]])
    expect(filterSessions(sessions, ' abc ')).toEqual([sessions[0]])
    expect(filterSessions(sessions, 'REACT')).toEqual([sessions[1]])
  })
  it('无匹配和特殊字符按普通子串处理', () => {
    expect(filterSessions(sessions, '[.*')).toEqual([])
    expect(filterSessions([], 'abc')).toEqual([])
  })
  it('历史映射保留服务端 ID、答案元信息和引用', () => {
    const citations = [{ index: 1, source: 'fixture.txt' }]
    const result = mapHistory([
      { id: 42, role: 'assistant', content: '测试', meta: { retrieved_count: 1, citations } },
    ])
    expect(result[0]).toMatchObject({
      id: 'srv-42',
      serverId: 42,
      role: 'assistant',
      citations,
      meta: { retrieved_count: 1 },
    })
    expect(mapHistory([{ id: 1, role: 'user', content: '?', meta: null }])[0].meta).toBeUndefined()
  })

  it('旧数据里 citations 是数字时不当作引用数组（回归：历史引用曾全部丢失）', () => {
    const [msg] = mapHistory([
      { id: 7, role: 'assistant', content: '旧回答', meta: { citations: 5, retrieved_count: 5 } },
    ])
    expect(msg.citations).toBeUndefined()
    expect(msg.meta).toEqual({ retrieved_count: 5 })
  })

  it('新数据里 citations 是数组时还原到顶层并从 meta 移除', () => {
    const citations = [
      {
        index: 1,
        source: 'a.md',
        rel_path: 'a.md',
        section: 's',
        chunk_id: 'c1',
        file_type: 'markdown',
        quote: 'q',
      },
      {
        index: 2,
        source: 'b.md',
        rel_path: 'b.md',
        section: 's',
        chunk_id: 'c2',
        file_type: 'markdown',
        quote: 'q',
      },
    ]
    const [msg] = mapHistory([
      { id: 8, role: 'assistant', content: '新回答', meta: { citations, citation_count: 2 } },
    ])
    expect(msg.citations).toHaveLength(2)
    expect(msg.meta).not.toHaveProperty('citations')
    expect(msg.meta?.citation_count).toBe(2)
  })
})

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
