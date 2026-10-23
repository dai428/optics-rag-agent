// frontend/src/lib/sessions.ts
// 会话列表相关的纯工具（可单测）。

import type { ChatMessage, SessionSummary } from '../types'

/** 只搜索已获取的会话标题和 ID；不伪装成服务端全文检索。 */
export function filterSessions(sessions: SessionSummary[], query: string): SessionSummary[] {
  const term = query.trim().toLocaleLowerCase()
  return sessions.filter((s) => `${s.title}\n${s.id}`.toLocaleLowerCase().includes(term))
}

/** 历史与实时消息采用同一引用结构，导出和界面共享真实数据。 */
export function mapHistory(
  history: {
    id: number
    role: string
    content: string
    meta: Record<string, unknown> | null
  }[],
): ChatMessage[] {
  return history.map((m) => {
    // 旧数据里 meta.citations 存的是条数（number），新数据才是完整数组。
    // 两种都兼容：不是数组就当作「没有可还原的引用」，避免拿数字去 .length 白屏。
    const raw = m.meta?.citations
    const citations = Array.isArray(raw) ? (raw as ChatMessage['citations']) : undefined
    // citations 已提到顶层，meta 里不再重复保留，免得两处口径打架
    const { citations: _drop, ...rest } = (m.meta ?? {}) as Record<string, unknown>
    void _drop
    return {
      id: `srv-${m.id}`,
      serverId: m.id,
      role: m.role === 'user' ? 'user' : 'assistant',
      content: m.content,
      meta: m.meta ? (rest as ChatMessage['meta']) : undefined,
      citations,
    }
  })
}

/** ISO 字符串可直接按字典序比较大小，省掉转 Date 的开销 */
export function byRecent(a: SessionSummary, b: SessionSummary): number {
  return (b.last_active || '').localeCompare(a.last_active || '')
}
