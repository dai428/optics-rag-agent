// frontend/src/components/MessageBubble.tsx
// 单条消息：区分用户/助手；助手侧额外渲染执行轨迹、富文本正文、引用、反馈与错误。

import { useState } from 'react'
import { renderRich } from '../markdown'
import type { ChatMessage, DoneMeta } from '../types'
import CitationList from './CitationList'
import StageTrail from './StageTrail'

interface Props {
  msg: ChatMessage
  /** 点击赞/踩时回调；不传则不显示反馈按钮 */
  onFeedback?: (messageId: string, rating: 'up' | 'down') => void
  /** 重新生成这一轮回答 */
  onRegenerate?: (messageId: string) => void
}

export default function MessageBubble({ msg, onFeedback, onRegenerate }: Props) {
  const [copied, setCopied] = useState(false)

  if (msg.role === 'user') {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user">{msg.content}</div>
      </div>
    )
  }

  const hasTrail = (msg.stages?.length ?? 0) > 0
  const citations = msg.citations ?? []
  const finished = !msg.streaming && !msg.error

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(msg.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* 非 HTTPS 或浏览器拒绝时静默失败 */
    }
  }

  return (
    <div className="row row-ai">
      <div className="avatar">◈</div>
      <div className="bubble bubble-ai">
        {hasTrail && <StageTrail stages={msg.stages ?? []} streaming={msg.streaming} />}

        {msg.content ? (
          <div className="answer">{renderRich(msg.content)}</div>
        ) : (
          msg.streaming && (
            <div className="answer muted">
              <span className="dot-flash" />
              正在检索文献并组织回答…
            </div>
          )
        )}

        {msg.streaming && msg.content && <span className="caret" />}

        {/* 拒答提示：明确告诉用户「没答」而不是让空白显得像故障 */}
        {msg.meta?.abstained ? (
          <div className="abstain-banner">
            <b>未作答</b>
            <span>{msg.meta.abstain_reason || '知识库中没有足够证据'}</span>
          </div>
        ) : null}

        {msg.error && <div className="msg-error">⚠ {msg.error}</div>}

        {!msg.streaming && citations.length > 0 && <CitationList citations={citations} />}

        {!msg.streaming && msg.meta && <MetaLine meta={msg.meta} />}

        {finished && msg.content && (
          <div className="msg-actions">
            <button className="msg-action" onClick={copy} title="复制回答">
              {copied ? '已复制' : '复制'}
            </button>
            {onRegenerate && (
              <button
                className="msg-action"
                onClick={() => onRegenerate(msg.id)}
                title="用同一个问题重新生成"
              >
                重新生成
              </button>
            )}
            {onFeedback && (
              <>
                <button
                  className={`msg-action${msg.feedback === 'up' ? ' active' : ''}`}
                  onClick={() => onFeedback(msg.id, 'up')}
                  title="回答有用"
                >
                  👍
                </button>
                <button
                  className={`msg-action${msg.feedback === 'down' ? ' active' : ''}`}
                  onClick={() => onFeedback(msg.id, 'down')}
                  title="回答有问题（会记入改进清单）"
                >
                  👎
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** 回答下方的一行「本次做了什么」：检索条数 / 子问题 / 耗时 */
function MetaLine({ meta }: { meta: DoneMeta }) {
  const chips: string[] = []
  if (meta.retrieved_count) chips.push(`检索 ${meta.retrieved_count} 条资料`)
  if (meta.sub_questions && meta.sub_questions.length > 1) {
    chips.push(`拆分为 ${meta.sub_questions.length} 个子问题`)
  }
  if (meta.used_retrieval === false && !meta.abstained) chips.push('未触发检索（闲聊）')
  if (meta.elapsed) chips.push(`${meta.elapsed}s`)

  if (chips.length === 0 && !meta.rewritten_query) return null

  return (
    <div className="meta-line">
      {chips.map((c) => (
        <span key={c} className="meta-chip">
          {c}
        </span>
      ))}
      {meta.rewritten_query && (
        <span className="meta-query" title="模型改写后的检索问句">
          检索用问句：{meta.rewritten_query}
        </span>
      )}
    </div>
  )
}
