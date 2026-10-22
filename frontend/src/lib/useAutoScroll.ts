// frontend/src/lib/useAutoScroll.ts
// 消息区自动滚动 Hook。
// 流式逐字追加时自动贴底；但用户手动往上滚（回看前文）时不再抢滚动位置。
// 判断标准：是否在底部 80px 以内。

import { useEffect, useRef } from 'react'

const THRESHOLD = 80

/**
 * @param trigger 只要这个值变化就尝试贴底（通常传 messages 数组引用）。
 *   用单个值而非 deps 数组，既让 ESLint 能静态校验，也避免调用方误传不稳定数组。
 */
export function useAutoScroll(trigger: unknown) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const pinnedRef = useRef(true)

  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < THRESHOLD
  }

  useEffect(() => {
    const el = scrollRef.current
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight
  }, [trigger])

  return { scrollRef, handleScroll, pinnedRef }
}
