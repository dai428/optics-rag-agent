// frontend/src/components/ChatPanel.tsx
// 聊天主区：滚动消息列表 + 底部输入框。
// 滚动管理抽到 useAutoScroll Hook；空态抽到 WelcomePanel（含实时指标）。

import { useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent } from 'react'
import type { ChatMessage, DocumentsInfo } from '../types'
import MessageBubble from './MessageBubble'
import WelcomePanel from './WelcomePanel'
import { useAutoScroll } from '../lib/useAutoScroll'

interface Props {
  messages: ChatMessage[]
  streaming: boolean
  onSend: (text: string) => void
  onStop: () => void
  onFeedback: (messageId: string, rating: 'up' | 'down') => void
  onRegenerate: (messageId: string) => void
  kb?: DocumentsInfo | null
}

export default function ChatPanel({
  messages,
  streaming,
  onSend,
  onStop,
  onFeedback,
  onRegenerate,
  kb,
}: Props) {
  const [text, setText] = useState('')
  const taRef = useRef<HTMLTextAreaElement>(null)
  const { scrollRef, handleScroll, pinnedRef } = useAutoScroll(messages)

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
          <WelcomePanel onSend={onSend} streaming={streaming} kb={kb} />
        ) : (
          messages.map((m) => (
            <MessageBubble key={m.id} msg={m} onFeedback={onFeedback} onRegenerate={onRegenerate} />
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
