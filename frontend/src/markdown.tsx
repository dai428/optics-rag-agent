// frontend/src/markdown.tsx
// 极简富文本渲染：只处理回答里最常见的几种行内/块级标记。
//
// 为什么不用 react-markdown？
//   回答来自我们自己的模型，格式可控；为几个标记引入一整套解析依赖不划算。
//   这里只做「粗体 / 行内代码 / 标题 / 无序·有序列表 / 空行分段」六件事。
//
// 安全性：所有文本先按纯字符串渲染，再由 React 转义，不注入 HTML，天然免疫 XSS。

import type { ReactNode } from 'react'

/** 处理 **粗体** 与 `行内代码`，其余原样保留 */
function inline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g
  let last = 0
  let hit: RegExpExecArray | null
  let n = 0

  while ((hit = re.exec(text)) !== null) {
    if (hit.index > last) nodes.push(text.slice(last, hit.index))
    const tok = hit[0]
    if (tok.startsWith('**')) {
      nodes.push(<strong key={`${keyPrefix}-b${n}`}>{tok.slice(2, -2)}</strong>)
    } else {
      nodes.push(<code key={`${keyPrefix}-c${n}`}>{tok.slice(1, -1)}</code>)
    }
    last = hit.index + tok.length
    n += 1
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes
}

export function renderRich(src: string): ReactNode {
  const out: ReactNode[] = []
  let buffer: string[] = []
  let ordered = false

  const flushList = (key: string) => {
    if (buffer.length === 0) return
    const items = buffer
    buffer = []
    out.push(
      ordered ? (
        <ol key={`l-${key}`} className="md-list">
          {items.map((t, i) => (
            <li key={i}>{inline(t, `${key}-${i}`)}</li>
          ))}
        </ol>
      ) : (
        <ul key={`l-${key}`} className="md-list">
          {items.map((t, i) => (
            <li key={i}>{inline(t, `${key}-${i}`)}</li>
          ))}
        </ul>
      ),
    )
  }

  src.split('\n').forEach((raw, idx) => {
    const line = raw.trimEnd()
    const key = String(idx)

    const bullet = line.match(/^\s*[-*·]\s+(.*)$/)
    if (bullet) {
      if (buffer.length > 0 && ordered) flushList(`x${key}`)
      ordered = false
      buffer.push(bullet[1])
      return
    }

    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/)
    if (numbered) {
      if (buffer.length > 0 && !ordered) flushList(`y${key}`)
      ordered = true
      buffer.push(numbered[1])
      return
    }

    flushList(`z${key}`)

    const heading = line.match(/^#{1,4}\s+(.*)$/)
    if (heading) {
      out.push(
        <h4 key={`h-${key}`} className="md-h">
          {inline(heading[1], `h${key}`)}
        </h4>,
      )
      return
    }

    if (line.trim() === '') {
      out.push(<div key={`sp-${key}`} className="md-gap" />)
      return
    }

    out.push(
      <p key={`p-${key}`} className="md-p">
        {inline(line, `p${key}`)}
      </p>,
    )
  })

  flushList('end')
  return out
}
