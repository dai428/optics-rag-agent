// frontend/src/components/ChatPanel.tsx
// 聊天主区：滚动消息列表 + 底部输入框
//
// 一个细节：流式逐字追加时会自动贴底，但如果用户手动往上滚（回看前文），
// 就不该再抢他的滚动位置。所以这里判断「是否已接近底部」再决定要不要跟。

import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent } from 'react'
import type { ChatMessage } from '../types'
import MessageBubble from './MessageBubble'

interface Props {
  messages: ChatMessage[]
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
  onFeedback: (messageId: string, rating: 'up' | 'down') => void
  onRegenerate: (messageId: string) => void
}

const EXAMPLES = [
  '测量矩阵条件数 κ2 的理论下界是多少？',
  'LCVR 相比传统机械旋转波片有什么优势？',
  '温度从 25℃ 升到 40℃，条件数恶化了多少？',
]

export default function ChatPanel({
  messages,
  streaming,
  onSend,
  onStop,
  onFeedback,
  onRegenerate,
}: Props) {
  const [text, setText] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)
  const pinnedRef = useRef(true)

  // 判断用户是否还在底部附近（阈值放宽到 80px，容忍小幅滚动）
  const handleScroll = () => {
    const el = scrollRef.current
    if (!el) return
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  useEffect(() => {
    const el = scrollRef.current
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight
  }, [messages])

  const submit = () => {
    const t = text.trim()
    if (!t || streaming) return
    onSend(t)
    setText('')
    pinnedRef.current = true
    if (taRef.current) taRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // isComposing：中文输入法选词时的回车不能当发送，否则拼音还没上屏就发出去了
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }

  return (
    <main className="chat">
      <div className="chat-scroll" ref={scrollRef} onScroll={handleScroll}>
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome-mark">◈</div>
            <h2>向你的科研资料库提问</h2>
            <p>
              背后是一套可溯源的 Agentic RAG：先改写问题、判断是否需要反问、必要时拆成子问题
              并行检索，再做交叉编码器精排；回答会附上可展开的原文证据，证据不足时明确拒答。
            </p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button
                  key={q}
                  className="example"
                  onClick={() => onSend(q)}
                  disabled={streaming}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <MessageBubble
              key={m.id}
              msg={m}
              onFeedback={onFeedback}
              onRegenerate={onRegenerate}
            />
          ))
        )}
      </div>

      <div className="composer">
        <div className="composer-box">
          <textarea
            ref={taRef}
            value={text}
            rows={1}
            aria-label="提问输入框"
            placeholder="问点什么，比如「条件数的理论下界是多少？」"
            onChange={handleChange}
            onKeyDown={handleKeyDown}
          />
          {streaming ? (
            <button className="btn-stop" onClick={onStop} title="中断本次生成">
              ■ 停止
            </button>
          ) : (
            <button className="btn-send" onClick={submit} disabled={!text.trim()}>
              发送 ↵
            </button>
          )}
        </div>
        <p className="composer-hint">
          Enter 发送 · Shift + Enter 换行 · 回答基于检索到的原文，引用可展开核对
        </p>
      </div>
    </main>
  )
}
