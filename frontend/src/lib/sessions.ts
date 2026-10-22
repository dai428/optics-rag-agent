// frontend/src/lib/sessions.ts
// 会话列表相关的纯工具（可单测）。

import type { SessionSummary } from '../types'

/** ISO 字符串可直接按字典序比较大小，省掉转 Date 的开销 */
export function byRecent(a: SessionSummary, b: SessionSummary): number {
  return (b.last_active || '').localeCompare(a.last_active || '')
}
