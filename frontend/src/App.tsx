// frontend/src/App.tsx
// 应用主组件：管会话、管消息、编排流式状态。
// 视图层被拆到 components/ 下，这里只负责「状态 + 逻辑」。

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createSession as apiCreateSession,
  deleteDocument,
  deleteSession,
  fetchDocuments,
  fetchHealth,
  fetchMessages,
  fetchReady,
  fetchSessions,
  newSessionId,
  reindex,
  renameSession,
  sendFeedback,
  streamChat,
  uploadDocument,
} from './api'
import ChatPanel from './components/ChatPanel'
import Sidebar from './components/Sidebar'
import type {
  ChatMessage,
  Citation,
  DocumentsInfo,
  IndexReport,
  ReadyInfo,
  SessionSummary,
} from './types'

// ── 本地缓存 key ──────────────────────────────────────────────────
// 服务端已经持久化会话与消息（SQLite），这里只缓存「当前选中的会话」这类
// 纯界面状态。聊天记录不再依赖浏览器存储，换设备也能看到。
const LS_CURRENT = 'optics-rag-agent:current:v3'

const rid = (prefix: string) =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`

/** ISO 字符串可直接按字典序比较大小，省掉转 Date 的开销 */
const byRecent = (a: SessionSummary, b: SessionSummary) =>
  (b.last_active || '').localeCompare(a.last_active || '')

export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [messagesMap, setMessagesMap] = useState<Record<string, ChatMessage[]>>({})
  const [currentSid, setCurrentSid] = useState<string>(() => {
    try {
      return localStorage.getItem(LS_CURRENT) ?? ''
    } catch {
      return ''
    }
  })
  const [streaming, setStreaming] = useState(false)
  const [kb, setKb] = useState<DocumentsInfo | null>(null)
  const [health, setHealth] = useState<{ status: string; has_api_key: boolean } | null>(null)
  const [ready, setReady] = useState<ReadyInfo | null>(null)
  const [toast, setToast] = useState<{ kind: 'info' | 'error'; text: string } | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  // 用 ref 做并发闸门：setStreaming 是异步的，连点两次仍可能重复发
  const busyRef = useRef(false)
  // 记录每个会话最后一条用户提问，供「重新生成」使用
  const lastQuestionRef = useRef<Record<string, string>>({})

  const showToast = useCallback((text: string, kind: 'info' | 'error' = 'info') => {
    setToast({ kind, text })
    setTimeout(() => setToast(null), kind === 'error' ? 8000 : 4000)
  }, [])

  // ── 持久化当前会话选择 ──
  useEffect(() => {
    try {
      localStorage.setItem(LS_CURRENT, currentSid)
    } catch {
      /* 隐私模式下 localStorage 可能不可用 */
    }
  }, [currentSid])

  // ── 刷新知识库与就绪状态 ──
  const refreshKb = useCallback(async () => {
    try {
      const [docs, readyInfo] = await Promise.all([fetchDocuments(), fetchReady()])
      setKb(docs)
      setReady(readyInfo)
    } catch {
      /* 后端不可达时保留上一次的状态 */
    }
  }, [])

  const refreshSessions = useCallback(async () => {
    try {
      const remote = await fetchSessions()
      setSessions(remote.slice().sort(byRecent))
    } catch {
      /* 后端不可达时保持本地列表 */
    }
  }, [])

  // ── 首次加载：探活 + 知识库概况 + 服务端会话 ──
  useEffect(() => {
    fetchHealth()
      .then((h) => setHealth({ status: h.status, has_api_key: h.has_api_key }))
      .catch(() => setHealth({ status: 'down', has_api_key: false }))

    void refreshKb()
    void refreshSessions()
  }, [refreshKb, refreshSessions])

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

  // ── 选中会话时按需加载服务端历史 ──
  const selectSession = useCallback(
    async (sid: string) => {
      setCurrentSid(sid)
      if (messagesMap[sid]?.length) return
      try {
        const history = await fetchMessages(sid)
        const mapped: ChatMessage[] = history.map((m) => ({
          id: `srv-${m.id}`,
          serverId: m.id,
          role: m.role === 'user' ? 'user' : 'assistant',
          content: m.content,
          meta: (m.meta as ChatMessage['meta']) ?? undefined,
          citations: (m.meta?.citations as Citation[] | undefined) ?? undefined,
        }))
        setMessagesMap((prev) => ({ ...prev, [sid]: mapped }))
      } catch {
        /* 会话可能已被删除 */
      }
    },
    [messagesMap],
  )

  /** 拿到当前会话 id；没有就向服务端新建一个 */
  const ensureSession = useCallback(async () => {
    if (currentSid) return currentSid
    try {
      const created = await apiCreateSession()
      setSessions((prev) => [created, ...prev])
      setCurrentSid(created.id)
      return created.id
    } catch {
      // 服务端不可用时退化为本地会话，至少让界面能用
      const sid = newSessionId()
      const ts = new Date().toISOString().slice(0, 19)
      setSessions((prev) => [
        { id: sid, title: '新会话', created_at: ts, last_active: ts, turns: 0 },
        ...prev,
      ])
      setCurrentSid(sid)
      return sid
    }
  }, [currentSid])

  // ── 发送一条提问 ──
  const send = useCallback(
    async (text: string, sidOverride?: string) => {
      const trimmed = text.trim()
      if (!trimmed || busyRef.current) return
      busyRef.current = true

      const sid = sidOverride ?? (await ensureSession())
      lastQuestionRef.current[sid] = trimmed
      const asstId = rid('a')

      setMessagesMap((prev) => ({
        ...prev,
        [sid]: [
          ...(prev[sid] ?? []),
          { id: rid('u'), role: 'user', content: trimmed },
          { id: asstId, role: 'assistant', content: '', streaming: true, stages: [] },
        ],
      }))
      setStreaming(true)

      const ac = new AbortController()
      abortRef.current = ac

      // 轨迹数组里「保留最新一个 pending 占位」，其余按到达顺序排列
      const upsertStage = (item: NonNullable<ChatMessage['stages']>[number]) =>
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
            onPending: (label) => upsertStage({ stage: '__pending__', label, pending: true }),
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
                citations: d.citations ?? [],
                meta: {
                  retrieved_count: d.retrieved_count,
                  sub_questions: d.sub_questions,
                  rewritten_query: d.rewritten_query,
                  used_retrieval: d.used_retrieval,
                  elapsed: d.elapsed,
                  abstained: d.abstained,
                  abstain_reason: d.abstain_reason,
                },
              })),
            onError: (msg) =>
              patchMessage(sid, asstId, (m) => ({ ...m, streaming: false, error: msg })),
          },
          ac.signal,
        )
      } catch (e) {
        const err = e as Error
        const aborted = err.name === 'AbortError'
        patchMessage(sid, asstId, (m) => ({
          ...m,
          streaming: false,
          stages: (m.stages ?? []).filter((x) => !x.pending),
          // 主动停止：已吐出的内容保留，不标红
          error: aborted ? undefined : err.message,
        }))
        if (!aborted) showToast(err.message, 'error')
      } finally {
        setStreaming(false)
        busyRef.current = false
        abortRef.current = null
      }

      // 后端在 done 时才更新 turns / last_active，这里拉一次保持一致
      void refreshSessions()
    },
    [ensureSession, patchMessage, refreshSessions, showToast],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])

  /** 重新生成：删掉最后一条助手消息，用同一个问题再问一次 */
  const regenerate = useCallback(
    async (messageId: string) => {
      if (busyRef.current || !currentSid) return
      const question = lastQuestionRef.current[currentSid]
      if (!question) {
        showToast('找不到原始问题，无法重新生成', 'error')
        return
      }
      setMessagesMap((prev) => {
        const list = prev[currentSid] ?? []
        const idx = list.findIndex((m) => m.id === messageId)
        // 同时删掉被重新生成的那条回答和它前面的用户提问，避免出现孤立气泡
        const next = idx >= 0 ? [...list.slice(0, Math.max(0, idx - 1))] : list
        return { ...prev, [currentSid]: next }
      })
      await send(question, currentSid)
    },
    [currentSid, send, showToast],
  )

  // ── 反馈 ──
  const feedback = useCallback(
    async (messageId: string, rating: 'up' | 'down') => {
      if (!currentSid) return
      const list = messagesMap[currentSid] ?? []
      const idx = list.findIndex((m) => m.id === messageId)
      const answer = idx >= 0 ? list[idx].content : ''
      const question = lastQuestionRef.current[currentSid] ?? ''

      patchMessage(currentSid, messageId, (m) => ({ ...m, feedback: rating }))
      try {
        await sendFeedback({
          session_id: currentSid,
          rating,
          question,
          answer: answer.slice(0, 4000),
          comment: rating === 'down' ? '用户标记回答有问题' : '',
        })
        showToast(rating === 'up' ? '已记录：回答有用' : '已记录，会进入改进清单')
      } catch (e) {
        patchMessage(currentSid, messageId, (m) => ({ ...m, feedback: undefined }))
        showToast(`反馈提交失败：${(e as Error).message}`, 'error')
      }
    },
    [currentSid, messagesMap, patchMessage, showToast],
  )

  // ── 会话操作 ──
  const createNewSession = useCallback(async () => {
    if (busyRef.current) return
    try {
      const created = await apiCreateSession()
      setSessions((prev) => [created, ...prev])
      setCurrentSid(created.id)
      setMessagesMap((prev) => ({ ...prev, [created.id]: [] }))
    } catch (e) {
      showToast(`新建会话失败：${(e as Error).message}`, 'error')
    }
  }, [showToast])

  const removeSession = useCallback(
    async (sid: string) => {
      setSessions((prev) => {
        const rest = prev.filter((s) => s.id !== sid)
        if (sid === currentSid) setCurrentSid(rest[0]?.id ?? '')
        return rest
      })
      setMessagesMap((prev) => {
        const copy = { ...prev }
        delete copy[sid]
        return copy
      })
      try {
        // 同时清掉后端 LangGraph 里的对话记忆与数据库记录，
        // 否则下次同 id 提问模型还记得上文
        await deleteSession(sid)
      } catch {
        /* 后端可能重启过、该会话本就不存在，忽略 */
      }
    },
    [currentSid],
  )

  const rename = useCallback(
    async (sid: string, title: string) => {
      setSessions((prev) => prev.map((s) => (s.id === sid ? { ...s, title } : s)))
      try {
        await renameSession(sid, title)
      } catch (e) {
        showToast(`重命名失败：${(e as Error).message}`, 'error')
        void refreshSessions()
      }
    },
    [refreshSessions, showToast],
  )

  // ── 文档操作 ──
  const handleUpload = useCallback(
    async (file: File, subdir: string): Promise<IndexReport | void> => {
      const res = await uploadDocument(file, subdir)
      if (res.index) return res.index
      return undefined
    },
    [],
  )

  const handleDeleteDoc = useCallback(
    async (relPath: string) => {
      await deleteDocument(relPath)
    },
    [],
  )

  const handleReindex = useCallback(async (force: boolean) => {
    await reindex(force)
  }, [])

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        currentSid={currentSid}
        kb={kb}
        health={health}
        ready={ready}
        busy={streaming}
        onSelect={(sid) => void selectSession(sid)}
        onNew={() => void createNewSession()}
        onDelete={(sid) => void removeSession(sid)}
        onRename={(sid, title) => void rename(sid, title)}
        onRefreshKb={() => void refreshKb()}
        onUpload={handleUpload}
        onDeleteDoc={handleDeleteDoc}
        onReindex={handleReindex}
      />
      <ChatPanel
        messages={messages}
        streaming={streaming}
        onSend={(t) => void send(t)}
        onStop={stop}
        onFeedback={(id, r) => void feedback(id, r)}
        onRegenerate={(id) => void regenerate(id)}
      />
      {toast ? (
        <div className={`toast ${toast.kind}`} role="status">
          {toast.text}
        </div>
      ) : null}
    </div>
  )
}
