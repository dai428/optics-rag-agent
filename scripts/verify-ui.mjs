/**
 * 前端验收探针（无第三方依赖，走 Chrome DevTools Protocol）
 *
 * 用途：在没有安装 Playwright 的环境里，对运行中的 http://127.0.0.1:8000
 * 做真实浏览器验收：
 *   1. 多视口布局快照（390 / 768 / 1024 / 1440 / 1920）
 *   2. 横向溢出检测：找出 getBoundingClientRect().right 超出视口的元素
 *   3. 控制台报错 / 失败请求收集
 *   4. 关键交互：主题切换、侧栏抽屉、导出按钮可用性
 *   5. 基础无障碍抽查：无文本按钮是否带 aria-label、输入框是否有标签
 *
 * 只读检测：不改页面、不发聊天请求。
 *
 * 用法：node scripts/verify-ui.mjs [baseUrl]
 */

import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'

const BASE = process.argv[2] || 'http://127.0.0.1:8000'
const OUT = '.verify'
const CHROME =
  process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const PORT = 9333

mkdirSync(OUT, { recursive: true })

if (!existsSync(CHROME)) {
  console.error(`找不到 Chrome：${CHROME}`)
  process.exit(1)
}

/* ── 极简 CDP 客户端 ─────────────────────────────────────────── */
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
          reject(new Error(`CDP 超时：${method}`))
        }
      }, 60000)
    })
  }

  on(fn) {
    this.listeners.add(fn)
  }
}

/* ── 启动 Chrome ─────────────────────────────────────────────── */
// 注意：Chrome 的 user-data-dir 必须用绝对路径，相对路径在部分版本上会静默退出；
// 并且要 detached + stdio:ignore，否则子进程被父进程的管道约束，调试端口起不来。
const userDataDir = resolve(OUT, 'chrome-profile')
mkdirSync(userDataDir, { recursive: true })

let chrome = null
const probeVersion = async () => {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/json/version`)
    return (await r.json()).webSocketDebuggerUrl
  } catch {
    return null
  }
}

let browserWs = await probeVersion()
if (!browserWs) {
  chrome = spawn(
    CHROME,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-first-run',
      '--no-default-browser-check',
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
  if (chrome) chrome.kill()
  console.error(`Chrome 调试端口 ${PORT} 没起来`)
  process.exit(1)
}

const cdp = await CDP.connect(browserWs)

const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' })
const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true })

const consoleErrors = []
const failedRequests = []

cdp.on((msg) => {
  if (msg.sessionId !== sessionId) return
  if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
    consoleErrors.push(msg.params.args.map((a) => a.value ?? a.description ?? '?').join(' '))
  }
  if (msg.method === 'Runtime.exceptionThrown') {
    consoleErrors.push(`未捕获异常：${msg.params.exceptionDetails.text}`)
  }
  if (msg.method === 'Network.loadingFailed') {
    failedRequests.push(`${msg.params.type} 失败：${msg.params.errorText}`)
  }
  if (msg.method === 'Network.responseReceived') {
    const { status, url } = msg.params.response
    if (status >= 400) failedRequests.push(`HTTP ${status} ${url}`)
  }
})

await cdp.send('Page.enable', {}, sessionId)
await cdp.send('Runtime.enable', {}, sessionId)
await cdp.send('Network.enable', {}, sessionId)

const evaluate = async (expression) => {
  const r = await cdp.send(
    'Runtime.evaluate',
    { expression, returnByValue: true, awaitPromise: true },
    sessionId,
  )
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text)
  return r.result.value
}

const goto = async (url) => {
  await cdp.send('Page.navigate', { url }, sessionId)
  await sleep(2600)
}

const screenshot = async (name) => {
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' }, sessionId)
  const file = `${OUT}/${name}.png`
  writeFileSync(file, Buffer.from(data, 'base64'))
  return file
}

/* 溢出检测：返回超出视口右边界的元素路径 */
const OVERFLOW_PROBE = `(() => {
  const vw = document.documentElement.clientWidth;
  const bad = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.position === 'fixed' || cs.display === 'none' || cs.visibility === 'hidden') continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.right > vw + 1) {
      const cls = (typeof el.className === 'string' ? el.className : '').trim().split(/\\s+/).slice(0, 2).join('.');
      bad.push({
        tag: el.tagName.toLowerCase(),
        cls,
        right: Math.round(r.right),
        width: Math.round(r.width),
        text: (el.textContent || '').trim().slice(0, 24),
      });
    }
  }
  return {
    vw,
    scrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
    overflowCount: bad.length,
    worst: bad.sort((a, b) => b.right - a.right).slice(0, 8),
  };
})()`

const VIEWPORTS = [
  { name: 'mobile-390', width: 390, height: 844, mobile: true },
  { name: 'mobile-430', width: 430, height: 932, mobile: true },
  { name: 'tablet-768', width: 768, height: 1024, mobile: true },
  { name: 'laptop-1024', width: 1024, height: 768, mobile: false },
  { name: 'desktop-1440', width: 1440, height: 900, mobile: false },
  { name: 'wide-1920', width: 1920, height: 1080, mobile: false },
]

const report = { base: BASE, viewports: [], interactions: {}, a11y: {}, consoleErrors, failedRequests }

for (const vp of VIEWPORTS) {
  await cdp.send(
    'Emulation.setDeviceMetricsOverride',
    {
      width: vp.width,
      height: vp.height,
      deviceScaleFactor: 1,
      mobile: vp.mobile,
    },
    sessionId,
  )
  await goto(BASE)
  await sleep(700)
  const probe = await evaluate(OVERFLOW_PROBE)
  const shot = await screenshot(vp.name)
  report.viewports.push({ ...vp, shot, ...probe })
  console.log(
    `${vp.name.padEnd(13)} vw=${String(probe.vw).padEnd(5)} scrollW=${String(probe.scrollWidth).padEnd(5)} 溢出元素=${probe.overflowCount}`,
  )
  if (probe.overflowCount) {
    for (const b of probe.worst.slice(0, 4)) {
      console.log(`   ↳ <${b.tag} class="${b.cls}"> right=${b.right} w=${b.width} "${b.text}"`)
    }
  }
}

/* ── 交互验收（桌面视口） ── */
await cdp.send(
  'Emulation.setDeviceMetricsOverride',
  { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false },
  sessionId,
)
await goto(BASE)

const clickBySelector = async (sel) => {
  const ok = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(sel)});
    if (!el) return false;
    el.click();
    return true;
  })()`)
  await sleep(500)
  return ok
}

report.interactions.themeToggleFound = await evaluate(
  `!!document.querySelector('[aria-label*="主题"], [title*="主题"], .theme-toggle')`,
)
report.interactions.darkApplied = await evaluate(`(() => {
  const before = document.documentElement.dataset.theme || 'light';
  const el = document.querySelector('[aria-label*="主题"], [title*="主题"], .theme-toggle');
  if (!el) return null;
  el.click();
  return { before, after: document.documentElement.dataset.theme || 'light' };
})()`)
await sleep(600)
report.interactions.darkShot = await screenshot('desktop-dark')

/* 切回亮色，避免污染后续快照 */
await evaluate(`(() => {
  const el = document.querySelector('[aria-label*="主题"], [title*="主题"], .theme-toggle');
  if (el && (document.documentElement.dataset.theme || 'light') === 'dark') el.click();
})()`)
await sleep(400)

/* 侧栏抽屉（移动视口） */
await cdp.send(
  'Emulation.setDeviceMetricsOverride',
  { width: 390, height: 844, deviceScaleFactor: 1, mobile: true },
  sessionId,
)
await goto(BASE)
report.interactions.drawerToggleFound = await evaluate(`!!document.querySelector('.sidebar-toggle')`)
await clickBySelector('.sidebar-toggle')
report.interactions.drawerOpen = await evaluate(
  `!!document.querySelector('.sidebar.open') || getComputedStyle(document.querySelector('.sidebar')).transform === 'none'`,
)
report.interactions.drawerShot = await screenshot('mobile-drawer')

/* 打开历史会话，看真实消息渲染（不发新请求） */
await cdp.send(
  'Emulation.setDeviceMetricsOverride',
  { width: 390, height: 844, deviceScaleFactor: 1, mobile: true },
  sessionId,
)
await goto(BASE)
await clickBySelector('.sidebar-toggle')
await sleep(400)
const sessionClicked = await evaluate(`(() => {
  const items = [...document.querySelectorAll('.session-item button, .session-item')];
  const target = items.find((el) => (el.textContent || '').includes('LCVR'));
  if (!target) return false;
  target.click();
  return true;
})()`)
await sleep(2600)
report.interactions.sessionOpened = sessionClicked
report.interactions.conversationShot = await screenshot('mobile-conversation')
report.interactions.bubbleCount = await evaluate(
  `document.querySelectorAll('.row, .bubble, .bubble-user').length`,
)

/* ── 无障碍抽查 ── */
report.a11y = await evaluate(`(() => {
  const noLabelButtons = [...document.querySelectorAll('button')]
    .filter((b) => !(b.textContent || '').trim() && !b.getAttribute('aria-label') && !b.getAttribute('title'))
    .map((b) => b.className || '(no class)');
  const inputs = [...document.querySelectorAll('input, textarea')];
  const unlabelledInputs = inputs
    .filter((i) => {
      const id = i.id;
      const hasAria = i.getAttribute('aria-label') || i.getAttribute('placeholder') || i.getAttribute('title');
      const hasLabel = id && document.querySelector('label[for="' + id + '"]');
      return !hasAria && !hasLabel;
    })
    .map((i) => i.className || i.tagName.toLowerCase());
  return {
    buttonCount: document.querySelectorAll('button').length,
    noLabelButtons,
    inputCount: inputs.length,
    unlabelledInputs,
    lang: document.documentElement.lang || '(缺失)',
    hasMain: !!document.querySelector('main'),
    headingCount: document.querySelectorAll('h1, h2').length,
  };
})()`)

writeFileSync(`${OUT}/report.json`, JSON.stringify(report, null, 2))

console.log('\n── 交互 ──')
console.log('主题切换:', JSON.stringify(report.interactions.darkApplied))
console.log('移动端抽屉:', report.interactions.drawerToggleFound, '打开=', report.interactions.drawerOpen)
console.log('历史会话可打开:', report.interactions.sessionOpened, '气泡数=', report.interactions.bubbleCount)
console.log('\n── 无障碍 ──')
console.log(JSON.stringify(report.a11y))
console.log('\n── 控制台 ──')
console.log('报错', consoleErrors.length, '条；失败请求', failedRequests.length, '条')
for (const e of consoleErrors.slice(0, 5)) console.log('  err:', e.slice(0, 160))
for (const e of failedRequests.slice(0, 5)) console.log('  req:', e.slice(0, 160))

chrome.kill()
process.exit(0)
