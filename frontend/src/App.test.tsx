import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'
import { downloadSession } from './lib/exportSession'
import type { SessionSummary } from './types'

vi.mock('./api')
vi.mock('./lib/exportSession', () => ({ downloadSession: vi.fn() }))
vi.mock('./components/DocumentsPanel', () => ({ default: () => <div>知识库测试占位</div> }))
vi.mock('./components/WelcomePanel', () => ({ default: () => <div>欢迎提问</div> }))

const first: SessionSummary = {
  id: 'session-a',
  title: '光学实验',
  created_at: '',
  last_active: '2026-10-22',
  turns: 1,
}
const second: SessionSummary = {
  ...first,
  id: 'session-b',
  title: 'React Notes',
  last_active: '2026-10-21',
}
const history = [{ id: 1, role: 'user', content: '已存问题', created_at: '', meta: null }]
const list = () => screen.getByRole('navigation', { name: '会话列表' })
const select = (title: string) => fireEvent.click(within(list()).getByText(title))

beforeEach(() => {
  vi.resetAllMocks()
  localStorage.clear()
  localStorage.setItem('optics-rag-agent:theme:v1', 'light')
  vi.mocked(api.fetchHealth).mockResolvedValue({ status: 'ok', has_api_key: true, sessions: 2 })
  vi.mocked(api.fetchSessions).mockResolvedValue([first, second])
  vi.mocked(api.fetchMessages).mockResolvedValue(history)
  vi.mocked(api.fetchReady).mockResolvedValue({ status: 'ready', checks: {} })
  vi.mocked(api.fetchDocuments).mockRejectedValue(new Error('测试未加载文档'))
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

async function mountSelected() {
  localStorage.setItem('optics-rag-agent:current:v3', first.id)
  render(<App />)
  await screen.findByText('已存问题')
  await within(list()).findByText(first.title)
}

describe('前端会话工作台集成', () => {
  it('恢复上次选中的会话并自动加载历史，空历史缓存不会反复请求', async () => {
    vi.mocked(api.fetchMessages).mockResolvedValue([])
    localStorage.setItem('optics-rag-agent:current:v3', first.id)
    render(<App />)
    await within(list()).findByText(first.title)
    await waitFor(() => expect(screen.getByLabelText('提问输入框')).toBeEnabled())
    select(first.title)
    expect(api.fetchMessages).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: '导出 JSON' })).toBeDisabled()
  })

  it('搜索标题和 ID，显示无结果并支持清除，不改变当前对话', async () => {
    await mountSelected()
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'SESSION-B' } })
    expect(within(list()).queryByText(first.title)).toBeNull()
    expect(within(list()).getByText(second.title)).toBeInTheDocument()
    expect(screen.getByText('已存问题')).toBeInTheDocument()
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: '找不到' } })
    expect(screen.getByText('没有匹配的会话')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '清除搜索' }))
    expect(within(list()).getByText(first.title)).toBeInTheDocument()
  })

  it('导出当前实际加载的消息，Markdown 和 JSON 共用当前会话数据', async () => {
    await mountSelected()
    fireEvent.click(screen.getByRole('button', { name: '导出 Markdown' }))
    fireEvent.click(screen.getByRole('button', { name: '导出 JSON' }))
    expect(downloadSession).toHaveBeenNthCalledWith(
      1,
      first,
      [expect.objectContaining({ content: '已存问题', serverId: 1 })],
      'markdown',
    )
    expect(downloadSession).toHaveBeenNthCalledWith(2, first, expect.any(Array), 'json')
  })

  it('历史失败可见、禁止发送及导出，重试后恢复', async () => {
    vi.mocked(api.fetchMessages)
      .mockRejectedValueOnce(new Error('HTTP 503'))
      .mockResolvedValue(history)
    localStorage.setItem('optics-rag-agent:current:v3', first.id)
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('HTTP 503')
    expect(screen.getByLabelText('提问输入框')).toBeDisabled()
    expect(screen.getByRole('button', { name: '导出 JSON' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '重试加载历史' }))
    await screen.findByText('已存问题')
    expect(screen.getByLabelText('提问输入框')).toBeEnabled()
    expect(api.fetchMessages).toHaveBeenCalledTimes(2)
  })

  it('快速切换时丢弃旧请求的迟到响应，不覆盖新会话', async () => {
    let resolveOld!: (value: typeof history) => void
    vi.mocked(api.fetchMessages).mockImplementation((sid) =>
      sid === first.id
        ? new Promise((resolve) => {
            resolveOld = resolve
          })
        : Promise.resolve([{ ...history[0], content: '新会话内容' }]),
    )
    render(<App />)
    await within(list()).findByText(first.title)
    select(first.title)
    expect(screen.getByRole('button', { name: '导出 JSON' })).toBeDisabled()
    select(second.title)
    await screen.findByText('新会话内容')
    await act(async () => {
      resolveOld(history)
    })
    expect(screen.queryByText('已存问题')).toBeNull()
    expect(screen.getByText('新会话内容')).toBeInTheDocument()
  })

  it('删除失败保留原会话和消息并显示错误', async () => {
    vi.mocked(api.deleteSession).mockRejectedValue(new Error('HTTP 500'))
    await mountSelected()
    fireEvent.click(screen.getByRole('button', { name: `删除会话 ${first.title}` }))
    await screen.findByText(/删除失败，原会话已保留/)
    expect(within(list()).getByText(first.title)).toBeInTheDocument()
    expect(screen.getByText('已存问题')).toBeInTheDocument()
  })

  it('取消确认不调用删除接口，成功删除才移除并加载下一会话', async () => {
    vi.mocked(api.deleteSession).mockResolvedValue({ status: 'deleted' })
    await mountSelected()
    vi.mocked(window.confirm).mockReturnValueOnce(false)
    fireEvent.click(screen.getByRole('button', { name: `删除会话 ${first.title}` }))
    expect(api.deleteSession).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: `删除会话 ${first.title}` }))
    await waitFor(() => expect(within(list()).queryByText(first.title)).toBeNull())
    expect(api.deleteSession).toHaveBeenCalledOnce()
    await waitFor(() => expect(api.fetchMessages).toHaveBeenCalledWith(second.id))
  })

  it('删除等待期间阻止重复删除和切换，成功后才更新列表', async () => {
    let finish!: (value: { status: string }) => void
    vi.mocked(api.deleteSession).mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    await mountSelected()
    const button = screen.getByRole('button', { name: `删除会话 ${first.title}` })
    fireEvent.click(button)
    fireEvent.click(button)
    select(second.title)
    expect(api.deleteSession).toHaveBeenCalledOnce()
    expect(button).toBeDisabled()
    expect(screen.getByText('已存问题')).toBeInTheDocument()
    expect(api.fetchMessages).not.toHaveBeenCalledWith(second.id)
    await act(async () => {
      finish({ status: 'deleted' })
    })
    await waitFor(() => expect(within(list()).queryByText(first.title)).toBeNull())
  })

  it('原有停止生成仍触发 AbortSignal、保留部分回答且释放新的交互锁', async () => {
    let streamSignal: AbortSignal | undefined
    vi.mocked(api.streamChat).mockImplementation(
      (_text, _sid, handlers, signal) =>
        new Promise((_resolve, reject) => {
          streamSignal = signal
          handlers.onToken?.('已生成的部分回答')
          signal?.addEventListener(
            'abort',
            () => reject(new DOMException('Aborted', 'AbortError')),
            { once: true },
          )
        }),
    )
    await mountSelected()
    fireEvent.change(screen.getByLabelText('提问输入框'), { target: { value: '继续提问' } })
    fireEvent.click(screen.getByRole('button', { name: /发送/ }))
    await screen.findByText('已生成的部分回答')
    expect(screen.getByRole('button', { name: '导出 JSON' })).toBeDisabled()
    expect(screen.getByRole('button', { name: `删除会话 ${first.title}` })).toBeDisabled()
    select(second.title)
    expect(api.fetchMessages).not.toHaveBeenCalledWith(second.id)
    fireEvent.click(screen.getByRole('button', { name: /停止/ }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /停止/ })).toBeNull())
    expect(streamSignal?.aborted).toBe(true)
    expect(screen.getByText('已生成的部分回答')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '导出 JSON' })).toBeEnabled()
    expect(screen.queryByText(/AbortError/)).toBeNull()
  })

  it('重命名失败保留原标题，不展示虚假的成功状态', async () => {
    vi.mocked(api.renameSession).mockRejectedValue(new Error('HTTP 503'))
    await mountSelected()
    fireEvent.doubleClick(within(list()).getByText(first.title))
    const input = screen.getByLabelText('重命名会话')
    fireEvent.change(input, { target: { value: '新标题' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    await screen.findByText(/重命名失败，原标题已保留/)
    expect(within(list()).getByText(first.title)).toBeInTheDocument()
    expect(within(list()).queryByText('新标题')).toBeNull()
  })
})
