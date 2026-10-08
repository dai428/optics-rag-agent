// frontend/src/components/StageTrail.tsx
// 执行轨迹：把「查询改写 → 澄清检查 → 问题拆分 → 资料检索 → 生成回答」显示成一条时间轴。
// 这是 Agentic RAG 相对朴素 RAG 最直观的差异 —— 用户能看见模型在做什么，而不是干等。

import type { StageItem } from '../types'

interface Props {
  stages: StageItem[]
  /** 是否还在流式中（决定是否有项处于「进行中」动画） */
  streaming?: boolean
}

export default function StageTrail({ stages, streaming }: Props) {
  if (stages.length === 0) return null

  return (
    <div className={`trail${streaming ? ' live' : ''}`}>
      {stages.map((s, i) => (
        <span
          key={`${s.stage}-${s.label}-${i}`}
          className={`trail-item ${s.pending ? 'pending' : 'done'}`}
          title={s.detail || s.label}
        >
          <span className="trail-icon">{s.pending ? '◌' : '✓'}</span>
          <span className="trail-label">{s.label}</span>
          {s.pending && <span className="trail-ellipsis" />}
          {!s.pending && s.elapsed != null && (
            <span className="trail-time">{s.elapsed}s</span>
          )}
        </span>
      ))}
    </div>
  )
}
