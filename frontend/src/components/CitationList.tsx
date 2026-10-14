// frontend/src/components/CitationList.tsx
// 引用卡片列表：把「回答里的结论」和「原文证据」连起来。
//
// 为什么值得单独做一个组件？
//   普通 RAG 演示只把文件名贴在回答下面，用户无法判断模型是不是编的。
//   这里给出可展开的原文片段（quote）、章节路径与相关度分数，
//   让「引用」从装饰变成可核验的证据。

import { useState } from 'react'
import type { Citation } from '../types'

/** 文件类型 → 显示标签 */
const TYPE_LABEL: Record<string, string> = {
  markdown: 'MD',
  text: 'TXT',
  csv: 'CSV',
  pdf: 'PDF',
  docx: 'DOCX',
}

export default function CitationList({ citations }: { citations: Citation[] }) {
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  if (citations.length === 0) return null

  return (
    <div className="citations">
      <div className="citations-head">
        <span className="citations-title">引用证据</span>
        <span className="citations-count">{citations.length} 条</span>
      </div>

      <div className="citation-items">
        {citations.map((c) => {
          const open = openIndex === c.index
          return (
            <div key={`${c.chunk_id}-${c.index}`} className={`citation${open ? ' open' : ''}`}>
              <button
                className="citation-head"
                onClick={() => setOpenIndex(open ? null : c.index)}
                aria-expanded={open}
                title={open ? '收起原文' : '展开原文片段'}
              >
                <span className="citation-index">[{c.index}]</span>
                <span className="citation-file">{c.source}</span>
                {c.page ? <span className="citation-page">第 {c.page} 页</span> : null}
                {c.file_type ? (
                  <span className="citation-type">{TYPE_LABEL[c.file_type] ?? c.file_type}</span>
                ) : null}
                <span className="citation-toggle">{open ? '收起' : '原文'}</span>
              </button>

              {c.section ? <div className="citation-section">{c.section}</div> : null}

              {open ? (
                <div className="citation-body">
                  <blockquote className="citation-quote">{c.quote}</blockquote>
                  <div className="citation-meta">
                    {typeof c.score === 'number' ? (
                      <span title="重排分数（含来源先验），越高越相关">
                        相关度 {c.score.toFixed(3)}
                      </span>
                    ) : null}
                    {typeof c.raw_score === 'number' ? (
                      <span title="CrossEncoder 原始分数，未叠加来源先验">
                        原始 {c.raw_score.toFixed(3)}
                      </span>
                    ) : null}
                    <span className="citation-chunk" title="父块 id，可用于定位原文">
                      {c.chunk_id}
                    </span>
                  </div>
                </div>
              ) : null}
            </div>
          )
        })}
      </div>
    </div>
  )
}
