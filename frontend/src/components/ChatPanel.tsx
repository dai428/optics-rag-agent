// frontend/src/components/ChatPanel.tsx
// 聊天主区：滚动消息列表 + 底部输入框

import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent } from 'react'
import type { ChatMessage } from '../types'
import MessageBubble from './MessageBubble'

interface Props {
  messages: ChatMessage[]
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
}

const EXAMPLES = [
  'LCVR 的相位延迟量是怎么标定的？',
  '全 Stokes 偏振检测的基本原理是什么？',
  'DE 加权和 NSGA-II 两条优化路线各有什么优缺点？',
]

export default function ChatPanel({ messages, streaming, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)

  // 消息变化（含流式逐字追加）时贴底。用 scrollTop 而非 scrollIntoView，
  // 免得把整个页面也一起滚走。
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const submit = () => {
    const t = text.trim()
    if (!t || streaming) return
    onSend(t)
    setText('')
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
      <div className="chat-scroll" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome-mark">◈</div>
            <h2>向你的光学文献库提问</h2>
            <p>
              背后是一套 Agentic RAG：先改写问题、判断是否需要反问、必要时拆成子问题并行检索，
              再基于命中的文献片段作答。
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
          messages.map((m) => <MessageBubble key={m.id} msg={m} />)
        )}
      </div>

      <div className="composer">
        <div className="composer-box">
          <textarea
            ref={taRef}
            value={text}
            rows={1}
            placeholder="问点什么，比如「LCVR 延迟量标定的误差来源有哪些？」"
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
          Enter 发送 · Shift + Enter 换行 · 对话按会话隔离，模型会记住本会话上文
        </p>
      </div>
    </main>
  )
}
