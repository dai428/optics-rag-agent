/**
 * 补充验收：主题切换、深色模式可读性、历史会话渲染、窄屏细节。
 *
 * 上一版探针里"主题切换 before=light after=light"是探针自身的问题：
 * 它在同一次 evaluate 里 click 后立刻读 data-theme，React 还没重渲染。
 * 这里改成跨语句、带等待的写法重新测。
 */

import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'

const BASE = process.argv[2] || 'http://127.0.0.1:8000'
const OUT = '.verify'
const CHROME =
  process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const PORT = 9335

mkdirSync(OUT, { recursive: true })
if (!existsSync(CHROME)) {
  console.error('找不到 Chrome')
  process.exit(1)
}

class CDP {
  constructor(ws) {
    this.ws = ws
    this.id = 0
    this.pending = new Map()
    this.listeners = new Set()
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id)
        this.pending.delete(msg.id)
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result)
      } else if (msg.method) {
        for (const fn of this.listeners) fn(msg)
      }
    })
  }
  static async connect(url) {
    const ws = new WebSocket(url)
    await new Promise((res, rej) => {
      ws.addEventListener('open', res, { once: true })
      ws.addEventListener('error', rej, { once: true })
    })
    return new CDP(ws)
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id
    const payload = { id, method, params }
    if (sessionId) payload.sessionId = sessionId
    this.ws.send(JSON.stringify(payload))
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id)
          reject(new Error(`超时 ${method}`))
        }
      }, 60000)
    })
  }
  on(fn) {
    this.listeners.add(fn)
  }
}

const userDataDir = resolve(OUT, 'chrome-profile-b')
mkdirSync(userDataDir, { recursive: true })

const probeVersion = async () => {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/json/version`)
    return (await r.json()).webSocketDebuggerUrl
  } catch {
    return null
  }
}

let browserWs = await probeVersion()
let chrome = null
if (!browserWs) {
  chrome = spawn(
    CHROME,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-first-run',
      '--hide-scrollbars',
      `--remote-debugging-port=${PORT}`,
      `--user-data-dir=${userDataDir}`,
      'about:blank',
    ],
    { stdio: 'ignore', detached: true },
  )
  chrome.unref()
  for (let i = 0; i < 80; i++) {
    browserWs = await probeVersion()
    if (browserWs) break
    await sleep(250)
  }
}
if (!browserWs) {
  console.error('调试端口未就绪')
  process.exit(1)
}

const cdp = await CDP.connect(browserWs)
const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' })
const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true })

await cdp.send('Page.enable', {}, sessionId)
await cdp.send('Runtime.enable', {}, sessionId)

const evaluate = async (expression) => {
  const r = await cdp.send(
    'Runtime.evaluate',
    { expression, returnByValue: true, awaitPromise: true },
    sessionId,
  )
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text)
  return r.result.value
}

const shot = async (name) => {
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' }, sessionId)
  const f = `${OUT}/${name}.png`
  writeFileSync(f, Buffer.from(data, 'base64'))
  return f
}

const setViewport = (width, height, mobile = false) =>
  cdp.send(
    'Emulation.setDeviceMetricsOverride',
    { width, height, deviceScaleFactor: 1, mobile },
    sessionId,
  )

const goto = async (url) => {
  await cdp.send('Page.navigate', { url }, sessionId)
  await sleep(2800)
}

const out = {}

/* ── 1. 主题切换（带等待） ── */
await setViewport(1440, 900)
await goto(BASE)

out.themeBefore = await evaluate(`document.documentElement.dataset.theme || '(无属性)'`)
out.toggleLabel = await evaluate(
  `(document.querySelector('.theme-toggle') || {}).getAttribute ? document.querySelector('.theme-toggle').getAttribute('aria-label') : null`,
)

await evaluate(`document.querySelector('.theme-toggle').click(); true`)
await sleep(900)
out.themeAfter = await evaluate(`document.documentElement.dataset.theme || '(无属性)'`)
out.localStorageTheme = await evaluate(
  `(localStorage.getItem('optics-rag-agent:theme:v1') || '(空)')`,
)
out.darkShot = await shot('check-dark')

/* 深色下的对比度抽查：正文/次要文字对背景 */
out.darkContrast = await evaluate(`(() => {
  const parse = (c) => {
    const m = c.match(/rgba?\\(([^)]+)\\)/);
    if (!m) return null;
    const p = m[1].split(',').map((x) => parseFloat(x));
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };
  const lum = ({ r, g, b }) => {
    const f = (v) => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const ratio = (a, b) => {
    const l1 = lum(a), l2 = lum(b);
    return ((Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05));
  };
  const bg = parse(getComputedStyle(document.body).backgroundColor);
  const pick = (sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const cs = getComputedStyle(el);
    const fg = parse(cs.color);
    if (!fg || !bg) return null;
    return { sel, ratio: Number(ratio(fg, bg).toFixed(2)), size: cs.fontSize };
  };
  return ['h2', '.welcome-lead', '.stat-label', '.example', '.composer-hint', '.session-item']
    .map(pick)
    .filter(Boolean);
})()`)

/* 关掉深色，回到亮色继续测 */
await evaluate(`document.querySelector('.theme-toggle').click(); true`)
await sleep(700)

/* ── 2. 历史会话渲染（桌面） ── */
await goto(BASE)
const SESSION_TITLE = process.env.SESSION_TITLE || '条件数'
const opened = await evaluate(`(() => {
  const needle = ${JSON.stringify(SESSION_TITLE)};
  const items = [...document.querySelectorAll('.session-item')];
  const t = items.find((el) => (el.textContent || '').includes(needle));
  if (!t) return false;
  t.click();
  return true;
})()`)
await sleep(3200)
out.sessionOpened = opened
out.messageCount = await evaluate(`document.querySelectorAll('.row').length`)
out.citationCount = await evaluate(`document.querySelectorAll('.citation').length`)
out.stageTrail = await evaluate(`document.querySelectorAll('.trail-step, .stage, .stages li').length`)
out.conversationShot = await shot('check-conversation-1440')

/* 空状态之外的错误提示是否残留 */
out.errorBanners = await evaluate(
  `[...document.querySelectorAll('[role="alert"]')].map((e) => (e.textContent || '').trim().slice(0, 40))`,
)

/* ── 3. 移动端会话页 ── */
await setViewport(390, 844, true)
await goto(BASE)
await evaluate(`(() => {
  const b = document.querySelector('.sidebar-toggle');
  if (b) b.click();
  return true;
})()`)
await sleep(500)
await evaluate(`(() => {
  const needle = ${JSON.stringify(SESSION_TITLE)};
  const items = [...document.querySelectorAll('.session-item')];
  const t = items.find((el) => (el.textContent || '').includes(needle));
  if (t) t.click();
  return true;
})()`)
await sleep(3000)
/* 关抽屉看内容 */
await evaluate(`(() => {
  const b = document.querySelector('.sidebar-toggle');
  if (b) b.click();
  return true;
})()`)
await sleep(600)
out.mobileConversationShot = await shot('check-conversation-390')
out.mobileOverflow = await evaluate(`(() => {
  const vw = document.documentElement.clientWidth;
  const bad = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.position === 'fixed' || cs.display === 'none') continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.right > vw + 1) {
      bad.push({
        tag: el.tagName.toLowerCase(),
        cls: (typeof el.className === 'string' ? el.className : '').split(/\\s+/).slice(0, 2).join('.'),
        right: Math.round(r.right),
        text: (el.textContent || '').trim().slice(0, 30),
      });
    }
  }
  return { overflowCount: bad.length, worst: bad.sort((a, b) => b.right - a.right).slice(0, 6) };
})()`)

/* 长代码块 / 表格在窄屏是否自己滚动而不撑破页面 */
out.mobileOverflow = out.mobileOverflow

writeFileSync(`${OUT}/check.json`, JSON.stringify(out, null, 2))

console.log('主题：', out.themeBefore, '→', out.themeAfter, '| localStorage =', out.localStorageTheme)
console.log('按钮 aria-label：', out.toggleLabel)
console.log('\n深色对比度（WCAG AA 正文需 ≥4.5，大字 ≥3）：')
for (const c of out.darkContrast) console.log(`  ${String(c.sel).padEnd(16)} ${String(c.ratio).padStart(5)}  ${c.size}`)
console.log('\n历史会话：打开=', out.sessionOpened, '消息行=', out.messageCount, '引用块=', out.citationCount, '轨迹=', out.stageTrail)
console.log('错误横幅：', out.errorBanners)
console.log('移动端会话页溢出：', out.mobileOverflow.overflowCount, out.mobileOverflow.worst)

if (chrome) chrome.kill()
process.exit(0)
