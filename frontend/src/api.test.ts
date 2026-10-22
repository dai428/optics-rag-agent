// frontend/src/api.test.ts
// 手写 SSE 解析的测试。
//
// 为什么值得测：这段代码要处理「报文被网络切成两半」的边界 ——
// 如果不在 buffer 里等下一块，JSON.parse 会拿到半截字符串然后静默丢事件，
// 表现就是「偶尔少一个 stage / 答案突然断掉」，很难靠肉眼复现。
// 这里用假 fetch 造出「一条报文被切在两个 chunk 里」的场景。

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { streamChat } from './api'

/** 把若干段字符串做成一个 ReadableStream（模拟分块到达的网络响应） */
function makeStream(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c))
      controller.close()
    },
  })
}

function mockFetch(chunks: string[], init: ResponseInit = { status: 200 }) {
  const res = new Response(makeStream(chunks), init)
  // jsdom 的 Response 不一定带 body.getReader，这里直接给出我们自己的流
  Object.defineProperty(res, 'body', { value: makeStream(chunks) })
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(res))
}

const collect = () => {
  const events: string[] = []
  return {
    events,
    handlers: {
      onStage: (s: { label: string }) => events.push(`stage:${s.label}`),
      onPending: (l: string) => events.push(`pending:${l}`),
      onToken: (t: string) => events.push(`token:${t}`),
      onReset: (r: string) => events.push(`reset:${r}`),
      onDone: () => events.push('done'),
      onError: (m: string) => events.push(`error:${m}`),
    },
  }
}

beforeEach(() => {
  vi.restoreAllMocks()
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('streamChat SSE 解析', () => {
  it('依次分发 stage / pending / token / done', async () => {
    const payload = [
      'event: stage\ndata: {"stage":"rewrite","label":"查询改写","elapsed":0}\n\n',
      'event: pending\ndata: {"label":"澄清检查"}\n\n',
      'event: token\ndata: {"text":"你好"}\n\n',
      'event: done\ndata: {"answer":"你好","retrieved_count":5}\n\n',
    ].join('')

    mockFetch([payload])
    const { events, handlers } = collect()
    await streamChat('问题', 'sid', handlers)

    expect(events).toEqual(['stage:查询改写', 'pending:澄清检查', 'token:你好', 'done'])
  })

  it('报文被切在两个 chunk 中间时仍能正确解析', async () => {
    // 关键用例：第二条报文的前半段与后半段分属两个 chunk
    const part1 = 'event: stage\ndata: {"stage":"a","label":"开始处理"}\n\nevent: token\ndata: {"te'
    const part2 = 'xt":"完整内容"}\n\nevent: done\ndata: {"answer":"完整内容"}\n\n'

    mockFetch([part1, part2])
    const { events, handlers } = collect()
    await streamChat('问题', 'sid', handlers)

    // 第二条 token 的 JSON 跨了 chunk，必须被完整拼出来而不是丢掉
    expect(events).toEqual(['stage:开始处理', 'token:完整内容', 'done'])
  })

  it('连接结束但没收到 done 时抛错（避免界面永远卡在生成中）', async () => {
    mockFetch(['event: token\ndata: {"text":"半截"}\n\n'])
    const { handlers } = collect()

    await expect(streamChat('问题', 'sid', handlers)).rejects.toThrow('未收到完成事件')
  })

  it('HTTP 非 2xx 时抛出带状态码的错误', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('boom', { status: 500 })))
    const { handlers } = collect()

    await expect(streamChat('问题', 'sid', handlers)).rejects.toThrow(/HTTP 500/)
  })
})
