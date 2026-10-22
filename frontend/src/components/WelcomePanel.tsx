// frontend/src/components/WelcomePanel.tsx
// 首页空态：Hero 卖点 + 三条核心能力 + 知识库实时指标。
// 这是给面试官看的"第一屏"——把项目的工程亮点直接摆到脸上。

import type { DocumentsInfo } from '../types'

interface Props {
  onSend: (text: string) => void
  streaming: boolean
  kb?: DocumentsInfo | null
}

const EXAMPLES = [
  '测量矩阵条件数 κ2 的理论下界是多少？',
  'LCVR 相比传统机械旋转波片有什么优势？',
  '温度从 25℃ 升到 40℃，条件数恶化了多少？',
]

const CAPABILITIES = [
  {
    icon: '🔎',
    title: '可溯源检索',
    text: '向量 + BM25 混合召回，CrossEncoder 精排，回答附原文证据卡片',
  },
  {
    icon: '🤖',
    title: 'Agentic 流程',
    text: '查询改写 · 澄清反问 · 问题拆分 · 并行检索 · 预算兜底',
  },
  { icon: '🔬', title: '垂直领域', text: '面向 LCVR 偏振测量的科研语料：表格展开 + 来源加权' },
]

export default function WelcomePanel({ onSend, streaming, kb }: Props) {
  const card = (label: string, value: string) => (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  )

  return (
    <div className="welcome">
      <div className="welcome-mark">◈</div>
      <h2>面向你的科研资料库提问</h2>
      <p className="welcome-lead">
        一套<strong>可溯源</strong>的 Agentic RAG：改写问题、判断是否需要反问、必要时拆成子问题
        并行检索，再做交叉编码器精排；回答附可展开的原文证据，证据不足时明确拒答。
      </p>

      <div className="cap-row">
        {CAPABILITIES.map((c) => (
          <div className="cap-card" key={c.title}>
            <span className="cap-icon">{c.icon}</span>
            <strong className="cap-title">{c.title}</strong>
            <span className="cap-text">{c.text}</span>
          </div>
        ))}
      </div>

      {kb ? (
        <div className="stat-grid">
          {card('语料文档', `${kb.document_count} 篇`)}
          {card('向量切片', `${kb.chunks} 块`)}
          {card('父块', `${kb.parent_chunks} 个`)}
          {card('检索方式', kb.retrieval.mode.includes('hybrid') ? '混合检索' : kb.retrieval.mode)}
        </div>
      ) : null}

      <div className="examples">
        {EXAMPLES.map((q) => (
          <button key={q} className="example" onClick={() => onSend(q)} disabled={streaming}>
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}
