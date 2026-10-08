// frontend/src/api.ts
// 后端接口封装。生产环境前后端同源（FastAPI 托管 dist），所以 BASE 留空；
// 开发环境由 vite.config.ts 的 proxy 把 /api 转发到 8000 端口。

import type { DocumentsInfo, DoneMeta, SessionSummary } from './types'

const BASE = ''

/** 统一把非 2xx 响应变成带上下文的异常 */
async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status} ${body.slice(0, 180)}`)
  }
  return (await res.json()) as T
}

export async function fetchHealth() {
  return asJson<{ status: string; has_api_key: boolean; sessions: number }>(
    await fetch(`${BASE}/api/health`),
  )
}

export async function fetchDocuments() {
  return asJson<DocumentsInfo>(await fetch(`${BASE}/api/documents`))
}

export async function fetchSessions() {
  const data = await asJson<{ sessions: SessionSummary[] }>(await fetch(`${BASE}/api/sessions`))
  return data.sessions
}

export async function deleteSession(sessionId: string) {
  return asJson<{ status: string }>(
    await fetch(`${BASE}/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),
  )
}

/**
 * 生成会话 ID。
 * ⚠️ 必须是 ASCII —— 实测中文 thread_id 会让 LangGraph 的 checkpointer 卡死。
 * 这里用 时间戳(base36) + 随机串，短且唯一。
 */
export function newSessionId(): string {
  return `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

/** SSE 各事件的回调 */
export interface StreamHandlers {
  onStage?: (s: { stage: string; label: string; detail?: string; elapsed?: number }) => void
  onPending?: (label: string) => void
  onToken?: (text: string) => void
  /** 后端在重试前通知「已流出的内容作废」，前端应清空当前气泡 */
  onReset?: (reason: string) => void
  onDone?: (data: { answer: string; title?: string } & DoneMeta) => void
  onError?: (message: string) => void
}

/**
 * 流式问答。
 *
 * 为什么不用浏览器原生的 EventSource？
 * ─────────────────────────────────────
 * EventSource 只能发 GET，参数只能塞进 URL，而且没法带请求体。
 * 我们的问题文本要走 POST body（长文本 + 中文，放 URL 不合适），
 * 所以只能用 fetch 拿到 ReadableStream，自己按 SSE 协议切分。
 */
export async function streamChat(
  message: string,
  sessionId: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  })

  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status} ${body.slice(0, 180)}`)
  }
  if (!res.body) throw new Error('响应没有可读流（浏览器不支持 ReadableStream？）')

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE 用「空行」分隔报文。切分时要注意最后一段可能被网络截断，
      // 必须留在 buffer 里等下一块数据 —— 否则 JSON 解析会炸。
      const parts = buffer.split(/\r?\n\r?\n/)
      buffer = parts.pop() ?? ''
      for (const part of parts) {
        dispatchEvent(part, handlers)
      }
    }
    // 收尾：万一服务端最后一条报文没跟空行
    if (buffer.trim()) dispatchEvent(buffer, handlers)
  } finally {
    reader.releaseLock()
  }
}

/** 解析一条 SSE 报文（形如 `event: token\ndata: {...}`）并分发 */
function dispatchEvent(raw: string, h: StreamHandlers) {
  let event = 'message'
  const dataLines: string[] = []

  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return

  let payload: Record<string, unknown>
  try {
    payload = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
  } catch {
    return // 半截报文，忽略
  }

  switch (event) {
    case 'stage':
      h.onStage?.(payload as never)
      break
    case 'pending':
      h.onPending?.(String(payload.label ?? ''))
      break
    case 'token':
      h.onToken?.(String(payload.text ?? ''))
      break
    case 'reset':
      h.onReset?.(String(payload.reason ?? ''))
      break
    case 'done':
      h.onDone?.(payload as never)
      break
    case 'error':
      h.onError?.(String(payload.message ?? '未知错误'))
      break
  }
}
