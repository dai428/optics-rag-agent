import type { ChatMessage, SessionSummary } from '../types'

export type ExportFormat = 'markdown' | 'json'

export function serializeSession(
  session: SessionSummary,
  messages: ChatMessage[],
  format: ExportFormat,
) {
  // Windows 文件名保留字符、控制字符与尾部句点都不进入下载文件名。
  const name =
    Array.from(session.title, (char) => (char.charCodeAt(0) < 32 ? '_' : char))
      .join('')
      .replace(/[<>:"/\\|?*]/g, '_')
      .replace(/[. ]+$/g, '')
      .slice(0, 80) || '会话'
  const filename = `rag-${name}-${session.id.replace(/[^a-zA-Z0-9_-]/g, '_')}.${format === 'json' ? 'json' : 'md'}`
  if (format === 'json') {
    return {
      filename,
      mime: 'application/json;charset=utf-8',
      content: JSON.stringify({ schema_version: 1, session, messages }, null, 2),
    }
  }
  const lines = [
    `# ${session.title.replace(/[\r\n]+/g, ' ') || '新会话'}`,
    '',
    `会话 ID：${session.id}`,
    `创建时间：${session.created_at}`,
    `最后活跃：${session.last_active}`,
    '',
  ]
  for (const message of messages) {
    lines.push(`## ${message.role === 'user' ? '用户' : '助手'}`, '', message.content, '')
    if (message.error) lines.push(`> 生成异常：${message.error}`, '')
    if (message.citations?.length) {
      lines.push('### 引用', '')
      for (const c of message.citations) {
        lines.push(
          `- [${c.index}] ${c.source}（${c.rel_path}）${c.page != null ? ` · 第 ${c.page} 页` : ''}`,
          `  章节：${c.section} · chunk：${c.chunk_id}`,
          ...c.quote.split(/\r?\n/).map((line) => `  > ${line}`),
          '',
        )
      }
    }
    if (message.meta)
      lines.push('### 检索元信息', '', '```json', JSON.stringify(message.meta, null, 2), '```', '')
  }
  return { filename, mime: 'text/markdown;charset=utf-8', content: lines.join('\n') }
}

export function downloadSession(
  session: SessionSummary,
  messages: ChatMessage[],
  format: ExportFormat,
) {
  const { filename, mime, content } = serializeSession(session, messages, format)
  const url = URL.createObjectURL(new Blob([content], { type: mime }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  try {
    anchor.click()
  } finally {
    anchor.remove()
    // 留一个任务周期给浏览器消费 URL，再释放内存。
    setTimeout(() => URL.revokeObjectURL(url), 0)
  }
}
