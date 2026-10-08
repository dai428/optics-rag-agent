// frontend/src/components/MessageBubble.tsx
// 单条消息：区分用户/助手；助手侧额外渲染执行轨迹、富文本正文、错误与元信息。

import { renderRich } from '../markdown'
import type { ChatMessage, DoneMeta } from '../types'
import StageTrail from './StageTrail'

export default function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === 'user') {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user">{msg.content}</div>
      </div>
    )
  }

  const hasTrail = (msg.stages?.length ?? 0) > 0

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

        {msg.error && <div className="msg-error">⚠ {msg.error}</div>}

        {!msg.streaming && msg.meta && <MetaLine meta={msg.meta} />}
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
  if (meta.used_retrieval === false) chips.push('未触发检索（闲聊）')
  if (meta.elapsed) chips.push(`${meta.elapsed}s`)

  if (chips.length === 0) return null

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
