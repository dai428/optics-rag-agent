// frontend/src/lib/time.ts
// 纯时间格式化工具（无 DOM 依赖，便于单元测试）。
// 集中在这里而非散落在组件里：Sidebar/消息时间戳共用同一套规则。

/**
 * 会话列表时间：今天只显示时分，更早显示「月/日」。
 * @param iso 后端给的 ISO 字符串（无时区，JS 按本地时间解析）
 * @param nowParam 可选"当前时间"，测试时可注入固定值
 */
export function fmtTime(iso: string, nowParam: Date = new Date()): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const sameDay = d.toDateString() === nowParam.toDateString()
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  return sameDay ? `${hh}:${mm}` : `${d.getMonth() + 1}/${d.getDate()}`
}

/** 相对时间：刚刚 / N 分钟前 / N 小时前 / 更早走 fmtTime */
export function timeAgo(iso: string, nowParam: Date = new Date()): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const diffMs = Math.max(0, nowParam.getTime() - d.getTime())
  const minutes = Math.floor(diffMs / 60000)
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} 小时前`
  return `${d.getMonth() + 1}月${d.getDate()}日`
}
