// frontend/src/App.tsx
// 应用主组件：管会话、管消息、编排流式状态。
// 视图层被拆到 components/ 下，这里只负责「状态 + 逻辑」。

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  deleteSession,
  fetchDocuments,
  fetchHealth,
  fetchSessions,
  newSessionId,
  streamChat,
} from './api'
import ChatPanel from './components/ChatPanel'
import Sidebar from './components/Sidebar'
import type { ChatMessage, DocumentsInfo, SessionSummary } from './types'

// ── 本地持久化 key ────────────────────────────────────────────────
// 为什么消息要存 localStorage？
// 后端 LangGraph 的 checkpointer 保存的是「给模型看的多轮记忆」，
// 并不提供「按会话拉历史消息」的接口；而用户刷新页面不该丢对话，
// 所以前端自己留一份。生产级做法是把消息落库，这里第一版先用浏览器存储。
const LS_SESSIONS = 'optics-rag-agent:sessions:v1'
const LS_MESSAGES = 'optics-rag-agent:messages:v1'
const LS_CURRENT = 'optics-rag-agent:current:v1'

function loadLocal<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

const nowSec = () => Date.now() / 1000
const rid = (prefix: string) =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`

export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>(() => loadLocal(LS_SESSIONS, []))
  const [messagesMap, setMessagesMap] = useState<Record<string, ChatMessage[]>>(() =>
    loadLocal(LS_MESSAGES, {}),
  )
  const [currentSid, setCurrentSid] = useState<string>(() => loadLocal(LS_CURRENT, ''))
  const [streaming, setStreaming] = useState(false)
  const [kb, setKb] = useState<DocumentsInfo | null>(null)
  const [health, setHealth] = useState<{ status: string; has_api_key: boolean } | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  // 用 ref 做并发闸门：setStreaming 是异步的，连点两次仍可能重复发
  const busyRef = useRef(false)

  // ── 持久化 ──
  useEffect(() => void localStorage.setItem(LS_SESSIONS, JSON.stringify(sessions)), [sessions])
  useEffect(() => void localStorage.setItem(LS_MESSAGES, JSON.stringify(messagesMap)), [messagesMap])
  useEffect(() => void localStorage.setItem(LS_CURRENT, JSON.stringify(currentSid)), [currentSid])

  // ── 首次加载：探活 + 知识库概况 + 后台已有会话 ──
  useEffect(() => {
    fetchHealth()
      .then((h) => setHealth({ status: h.status, has_api_key: h.has_api_key }))
      .catch(() => setHealth({ status: 'down', has_api_key: false }))

    fetchDocuments().then(setKb).catch(() => undefined)

    fetchSessions()
      .then((remote) => {
        if (remote.length === 0) return
        // 后端重启后本地仍留着旧会话，这里做并集，避免历史凭空消失
        setSessions((prev) => {
          const seen = new Set(prev.map((s) => s.session_id))
          const merged = [...prev, ...remote.filter((r) => !seen.has(r.session_id))]
          return merged.sort((a, b) => b.last_active - a.last_active)
        })
      })
      .catch(() => undefined)
  }, [])

  const messages = currentSid ? (messagesMap[currentSid] ?? []) : []

  /** 精确替换某会话里的某条消息 */
  const patchMessage = useCallback(
    (sid: string, id: string, fn: (m: ChatMessage) => ChatMessage) => {
      setMessagesMap((prev) => {
        const list = prev[sid]
        if (!list?.length) return prev
        const idx = list.findIndex((m) => m.id === id)
        if (idx === -1) return prev
        const next = list.slice()
        next[idx] = fn(next[idx])
        return { ...prev, [sid]: next }
      })
    },
    [],
  )

  /** 拿到当前会话 id；如果没有就现造一个 */
  const ensureSession = useCallback(() => {
    if (currentSid) return currentSid
    const sid = newSessionId()
    const ts = nowSec()
    setSessions((prev) => [
      { session_id: sid, title: '新会话', created_at: ts, last_active: ts, turns: 0 },
      ...prev,
    ])
    setCurrentSid(sid)
    return sid
  }, [currentSid])

  // ── 发送一条提问 ──
  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || busyRef.current) return
      busyRef.current = true

      const sid = ensureSession()
      const asstId = rid('a')

      setMessagesMap((prev) => ({
        ...prev,
        [sid]: [
          ...(prev[sid] ?? []),
          { id: rid('u'), role: 'user', content: trimmed },
          { id: asstId, role: 'assistant', content: '', streaming: true, stages: [] },
        ],
      }))
      setSessions((prev) =>
        prev.map((s) =>
          s.session_id === sid
            ? {
                ...s,
                // 第一轮用提问前缀当标题（与后端 _touch_session 规则一致）
                title: s.turns === 0 ? trimmed.slice(0, 24) : s.title,
                last_active: nowSec(),
                turns: s.turns + 1,
              }
            : s,
        ),
      )
      setStreaming(true)

      const ac = new AbortController()
      abortRef.current = ac

      // 轨迹数组里「保留最新一个 pending 占位」，其余按到达顺序排列
      const upsertStage = (item: ChatMessage['stages'] extends (infer T)[] | undefined ? T : never) =>
        patchMessage(sid, asstId, (m) => ({
          ...m,
          stages: [...(m.stages ?? []).filter((x) => !x.pending), item],
        }))

      try {
        await streamChat(
          trimmed,
          sid,
          {
            onStage: (s) => upsertStage({ ...s }),
            onPending: (label) =>
              upsertStage({ stage: '__pending__', label, pending: true }),
            onToken: (t) =>
              patchMessage(sid, asstId, (m) => ({ ...m, content: m.content + t })),
            // 模型长回答被上游掐断 → 后端重试前会发 reset。
            // 不清空的话，用户会看到「半截 + 完整」两段重复答案。
            onReset: () => patchMessage(sid, asstId, (m) => ({ ...m, content: '' })),
            onDone: (d) =>
              patchMessage(sid, asstId, (m) => ({
                ...m,
                // 用 done 里的权威答案覆盖流式拼接结果 —— 若中途重试过，
                // 流式片段可能有重复，必须以它为准
                content: d.answer || m.content,
                streaming: false,
                stages: (m.stages ?? []).filter((x) => !x.pending),
                meta: {
                  retrieved_count: d.retrieved_count,
                  sub_questions: d.sub_questions,
                  rewritten_query: d.rewritten_query,
                  used_retrieval: d.used_retrieval,
                  elapsed: d.elapsed,
                },
              })),
            onError: (msg) =>
              patchMessage(sid, asstId, (m) => ({ ...m, streaming: false, error: msg })),
          },
          ac.signal,
        )
      } catch (e) {
        const err = e as Error
        patchMessage(sid, asstId, (m) => ({
          ...m,
          streaming: false,
          // 主动停止：已吐出的内容保留，不标红
          error: err.name === 'AbortError' ? undefined : err.message,
        }))
      } finally {
        setStreaming(false)
        busyRef.current = false
        abortRef.current = null
      }
    },
    [ensureSession, patchMessage],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])

  // ── 会话操作 ──
  const createSession = useCallback(() => {
    if (busyRef.current) return
    const sid = newSessionId()
    const ts = nowSec()
    setSessions((prev) => [
      { session_id: sid, title: '新会话', created_at: ts, last_active: ts, turns: 0 },
      ...prev,
    ])
    setCurrentSid(sid)
  }, [])

  const removeSession = useCallback(
    async (sid: string) => {
      setSessions((prev) => {
        const rest = prev.filter((s) => s.session_id !== sid)
        if (sid === currentSid) setCurrentSid(rest[0]?.session_id ?? '')
        return rest
      })
      setMessagesMap((prev) => {
        const copy = { ...prev }
        delete copy[sid]
        return copy
      })
      try {
        // 同时清掉后端 LangGraph 里的对话记忆，否则下次同 id 提问模型还记得上文
        await deleteSession(sid)
      } catch {
        /* 后端可能重启过、该会话本就不存在，忽略 */
      }
    },
    [currentSid],
  )

  const selectSession = useCallback((sid: string) => setCurrentSid(sid), [])

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        currentSid={currentSid}
        kb={kb}
        health={health}
        busy={streaming}
        onSelect={selectSession}
        onNew={createSession}
        onDelete={removeSession}
      />
      <ChatPanel
        messages={messages}
        streaming={streaming}
        onSend={send}
        onStop={stop}
      />
    </div>
  )
}
