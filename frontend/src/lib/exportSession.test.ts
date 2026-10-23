import { afterEach, describe, expect, it, vi } from 'vitest'
import { downloadSession, serializeSession } from './exportSession'
import type { ChatMessage, SessionSummary } from '../types'

const session: SessionSummary = {
  id: 's-1',
  title: '光学/实验:*?',
  created_at: '2026-10-22T10:00:00',
  last_active: '2026-10-22T11:00:00',
  turns: 1,
}
const messages: ChatMessage[] = [
  { id: 'u', role: 'user', content: '条件数？' },
  {
    id: 'a',
    role: 'assistant',
    content: '示例测试回答 [1]',
    meta: { retrieved_count: 1, used_retrieval: true },
    citations: [
      {
        index: 1,
        source: 'fixture.pdf',
        rel_path: 'fixture.pdf',
        page: 2,
        section: '第二章',
        chunk_id: 'chunk-1',
        file_type: 'pdf',
        quote: '第一行\n第二行',
      },
    ],
  },
]

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('会话导出', () => {
  it('Markdown 保留问题、答案、引用定位与原文、检索信息', () => {
    const result = serializeSession(session, messages, 'markdown')
    expect(result.content).toContain('## 用户\n\n条件数？')
    expect(result.content).toContain('示例测试回答 [1]')
    expect(result.content).toContain('[1] fixture.pdf（fixture.pdf） · 第 2 页')
    expect(result.content).toContain('章节：第二章 · chunk：chunk-1')
    expect(result.content).toContain('  > 第一行\n  > 第二行')
    expect(result.content).toContain('"retrieved_count": 1')
    expect(result.mime).toContain('text/markdown')
  })

  it('JSON 可无损解析并包含 schema_version、会话和真实消息对象', () => {
    const result = serializeSession(session, messages, 'json')
    expect(JSON.parse(result.content)).toEqual({ schema_version: 1, session, messages })
    expect(result.filename).toMatch(/\.json$/)
  })

  it('清理 Windows 文件名特殊字符、控制字符和路径分隔符', () => {
    const result = serializeSession(
      { ...session, id: '../s', title: '\u0000a/b\\c:*?<>|". ' },
      [],
      'markdown',
    )
    expect(result.filename).toBe('rag-_a_b_c_______-___s.md')
  })

  it('空标题与失败消息有明确输出且不修改输入', () => {
    const input: ChatMessage[] = [
      { id: 'e', role: 'assistant', content: '部分回答', error: '连接中断' },
    ]
    const before = JSON.stringify(input)
    const result = serializeSession({ ...session, title: '' }, input, 'markdown')
    expect(result.filename).toContain('rag-会话')
    expect(result.content).toContain('生成异常：连接中断')
    expect(JSON.stringify(input)).toBe(before)
  })

  it('触发本地 Blob 下载并释放对象 URL 与临时节点', () => {
    vi.useFakeTimers()
    const create = vi.fn().mockReturnValue('blob:test')
    const revoke = vi.fn()
    vi.stubGlobal('URL', { createObjectURL: create, revokeObjectURL: revoke })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      expect(this.download).toMatch(/\.md$/)
      expect(this.href).toBe('blob:test')
    })
    downloadSession(session, messages, 'markdown')
    expect(create).toHaveBeenCalledWith(expect.any(Blob))
    expect(click).toHaveBeenCalledOnce()
    expect(document.querySelector('a[download]')).toBeNull()
    vi.runAllTimers()
    expect(revoke).toHaveBeenCalledWith('blob:test')
  })
})
