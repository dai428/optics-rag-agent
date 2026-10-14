// frontend/src/api.ts
// 后端接口封装。生产环境前后端同源（FastAPI 托管 dist），所以 BASE 留空；
// 开发环境由 vite.config.ts 的 proxy 把 /api 转发到 8000 端口。

import type {
  Citation,
  DocumentsInfo,
  DoneMeta,
  IndexReport,
  MetricsInfo,
  ReadyInfo,
  SessionSummary,
} from './types'

const BASE = ''

/** 可选访问令牌。设置后写接口会带上 Authorization 头。 */
let accessToken = ''
export function setAccessToken(token: string) {
  accessToken = token
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return accessToken ? { ...extra, Authorization: `Bearer ${accessToken}` } : extra
}

/** 统一把非 2xx 响应变成带上下文的异常 */
async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    // 尽量把后端给的 detail 提出来，便于界面直接展示
    let detail = body.slice(0, 200)
    try {
      const parsed = JSON.parse(body)
      if (parsed?.detail) detail = String(parsed.detail)
    } catch {
      /* 非 JSON 响应，保留原文 */
    }
    throw new Error(`HTTP ${res.status} ${detail}`)
  }
  return (await res.json()) as T
}

export async function fetchHealth() {
  return asJson<{ status: string; has_api_key: boolean; sessions: number }>(
    await fetch(`${BASE}/api/health`),
  )
}

export async function fetchReady() {
  return asJson<ReadyInfo>(await fetch(`${BASE}/api/ready`))
}

export async function fetchDocuments() {
  return asJson<DocumentsInfo>(await fetch(`${BASE}/api/documents`))
}

export async function fetchMetrics() {
  return asJson<MetricsInfo>(await fetch(`${BASE}/api/metrics`))
}

export async function fetchSessions() {
  const data = await asJson<{ sessions: SessionSummary[] }>(await fetch(`${BASE}/api/sessions`))
  return data.sessions
}

export async function createSession() {
  return asJson<SessionSummary>(
    await fetch(`${BASE}/api/sessions`, { method: 'POST', headers: authHeaders() }),
  )
}

export async function renameSession(sessionId: string, title: string) {
  return asJson<{ status: string; title: string }>(
    await fetch(`${BASE}/api/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'PATCH',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ title }),
    }),
  )
}

export async function deleteSession(sessionId: string) {
  return asJson<{ status: string }>(
    await fetch(`${BASE}/api/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
      headers: authHeaders(),
    }),
  )
}

/** 服务端持久化的会话历史 */
export async function fetchMessages(sessionId: string) {
  const data = await asJson<{
    messages: {
      id: number
      role: string
      content: string
      created_at: string
      meta: Record<string, unknown> | null
    }[]
  }>(await fetch(`${BASE}/api/sessions/${encodeURIComponent(sessionId)}/messages`))
  return data.messages
}

/** 上传文档并（默认）立即重建索引 */
export async function uploadDocument(file: File, subdir = ''): Promise<{
  status: string
  rel_path: string
  message: string
  index?: IndexReport
  hint?: string
}> {
  const form = new FormData()
  form.append('file', file)
  form.append('subdir', subdir)
  form.append('reindex', 'true')
  return asJson(
    await fetch(`${BASE}/api/documents/upload`, {
      method: 'POST',
      headers: authHeaders(),
      body: form,
    }),
  )
}

export async function deleteDocument(relPath: string) {
  return asJson<{ status: string; index: IndexReport }>(
    await fetch(`${BASE}/api/documents/${encodeURIComponent(relPath)}`, {
      method: 'DELETE',
      headers: authHeaders(),
    }),
  )
}

export async function reindex(force = false) {
  return asJson<IndexReport>(
    await fetch(`${BASE}/api/documents/reindex`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ force }),
    }),
  )
}

export async function sendFeedback(payload: {
  session_id: string
  rating: 'up' | 'down'
  comment?: string
  question?: string
  answer?: string
}) {
  return asJson<{ status: string; feedback_id: number }>(
    await fetch(`${BASE}/api/feedback`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    }),
  )
}

/**
 * 生成会话 ID。
 * ⚠️ 必须是 ASCII —— 实测中文 thread_id 会让 LangGraph 的 checkpointer 卡死。
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
  onDone?: (
    data: { answer: string; title?: string; citations?: Citation[] } & DoneMeta,
  ) => void
  onError?: (message: string) => void
}

/**
 * 流式问答。
 *
 * 为什么不用浏览器原生的 EventSource？
 *   EventSource 只能发 GET，参数只能塞进 URL，而且没法带请求体。
 *   我们的问题文本要走 POST body（长文本 + 中文，放 URL 不合适），
 *   所以只能用 fetch 拿到 ReadableStream，自己按 SSE 协议切分。
 */
export async function streamChat(
  message: string,
  sessionId: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/api/chat/stream`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
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
  let sawDone = false

  const wrapped: StreamHandlers = {
    ...handlers,
    onDone: (data) => {
      sawDone = true
      handlers.onDone?.(data)
    },
  }

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
        dispatchEvent(part, wrapped)
      }
    }
    // 收尾：万一服务端最后一条报文没跟空行
    if (buffer.trim()) dispatchEvent(buffer, wrapped)
  } finally {
    reader.releaseLock()
  }

  // 连接正常结束但没有收到 done：说明后端中途断了（反代超时等）。
  // 不报错的话前端会永远停在「生成中」，所以这里显式抛出让上层收尾。
  if (!sawDone) {
    throw new Error('连接已结束但未收到完成事件（可能是服务端中断或代理超时）')
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
