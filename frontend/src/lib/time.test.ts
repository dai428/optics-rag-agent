// frontend/src/lib/time.test.ts
// 纯时间格式化函数的单测 —— 用固定"现在"时间保证确定性。

import { describe, it, expect } from 'vitest'
import { fmtTime, timeAgo } from './time'

// 固定"现在"：本地时间 2026-10-22 12:00:00
const NOW = new Date(2026, 9, 22, 12, 0, 0)

describe('fmtTime', () => {
  it('今天只显示时分', () => {
    expect(fmtTime('2026-10-22T09:05:00', NOW)).toBe('09:05')
  })

  it('更早显示月/日', () => {
    expect(fmtTime('2026-10-20T23:59:00', NOW)).toBe('10/20')
  })

  it('空串返回空串', () => {
    expect(fmtTime('', NOW)).toBe('')
  })

  it('非法日期返回空串', () => {
    expect(fmtTime('not-a-date', NOW)).toBe('')
  })
})

describe('timeAgo', () => {
  it('小于 1 分钟为「刚刚」', () => {
    expect(timeAgo('2026-10-22T11:59:30', NOW)).toBe('刚刚')
  })

  it('分钟级', () => {
    expect(timeAgo('2026-10-22T11:40:00', NOW)).toBe('20 分钟前')
  })

  it('小时级', () => {
    expect(timeAgo('2026-10-22T09:00:00', NOW)).toBe('3 小时前')
  })

  it('跨天显示月/日', () => {
    expect(timeAgo('2026-10-20T09:00:00', NOW)).toBe('10月20日')
  })
})
