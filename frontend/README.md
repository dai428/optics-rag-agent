# Frontend · 科研文献 Agent 前端

Vite + React 18 + TypeScript。在整条链路里，它负责把后端 LangGraph 的**规划过程**可视化，
并把回答里的每条结论和**可展开的原文证据**连起来。

```
浏览器 (React)
  │  POST /api/chat/stream   { message, session_id }
  ▼
FastAPI ── graph.astream(...) ── SSE 事件
  │
  │  stage / pending / token / reset / done / error
  ▼
本目录负责：分事件渲染时间轴、打字机正文、引用卡片、会话与知识库管理
```

## 快速开始

```bash
npm install
npm run dev        # 5173，/api 由 vite 代理到后端 8000

npm run build      # tsc --noEmit + vite build，产出 dist/ 交给 FastAPI 托管
npm run test       # Vitest（jsdom + React Testing Library）
npm run lint       # ESLint 10 flat config
npm run format     # Prettier
```

> 改完前端要重新 `npm run build` 才会体现在 8000 端口（FastAPI 托管的是 `dist/`）。

## 目录结构

```
src/
├─ App.tsx                  # 会话/消息状态编排：流式事件 → 不可变地 patch 单条消息
├─ api.ts                   # 接口封装 + 手写 SSE 解析（fetch ReadableStream）
├─ api.test.ts              # SSE 解析单测：含「报文被切在两个 chunk 中间」的边界
├─ types.ts                 # 与后端返回一一对应的类型，不做无意义改名
├─ markdown.tsx             # 轻量富文本渲染，不引 markdown 依赖，天然免疫 XSS
├─ markdown.test.tsx        # 富文本渲染单测
├─ styles.css               # 设计系统：CSS 变量 + 深浅两套主题
├─ lib/
│  ├─ useTheme.ts           # 深浅主题：跟随系统 → 手动切换后记忆到 localStorage
│  ├─ useAutoScroll.ts      # 流式贴底，但用户上滚时不抢滚动位置
│  ├─ time.ts               # 时间格式化纯函数（可单测，now 可注入）
│  └─ sessions.ts           # 会话排序纯函数
├─ components/
│  ├─ Sidebar.tsx           # 会话列表 + 知识库面板 + 状态灯 + 主题切换
│  ├─ ChatPanel.tsx         # 消息滚动区 + 输入框（中文输入法回车不误发）
│  ├─ WelcomePanel.tsx      # 空态：能力卡 + 知识库实时指标 + 示例问题
│  ├─ MessageBubble.tsx     # 单条消息：轨迹 / 正文 / 引用 / 拒答 / 反馈
│  ├─ StageTrail.tsx        # Agentic 执行轨迹时间轴
│  ├─ CitationList.tsx      # 引用卡片：可展开原文片段 + 分数
│  ├─ DocumentsPanel.tsx    # 上传 / 删除 / 重建索引
│  └─ ThemeToggle.tsx       # 主题切换（图标按钮，语义靠 aria-label）
└─ test/setup.ts            # 注入 jest-dom 匹配器
```

## 几个刻意的取舍

**不用 `EventSource`** —— 它只能发 GET、参数只能塞 URL，没法带请求体。问题文本走 POST body，
所以用 `fetch` 拿 `ReadableStream`，自己按 SSE 协议切报文。切分时最后一段可能被网络截断，
必须留在 buffer 里等下一块（`api.ts` 的 `streamChat`）。

**token 按节点过滤** —— 图里 6 个节点都调大模型，但只有 `clarify` / `respond` 的输出该给用户看；
`rewrite`、`decompose` 的输出是内部中间产物（比如「清晰」两个字）。多子问题并行时 token 会交错串台，
必须等 `respond` 汇总后再推。

**`done` 覆盖流式拼接** —— 后端在上游掐断时会重试并先发 `reset`。若中途重试过，流式片段可能重复，
所以最终以 `done` 里的权威答案覆盖，避免用户看到「半截 + 完整」两段重复答案。

**滚动位置归用户** —— 流式贴底只在「用户还在底部 80px 内」时发生；手动上滚回看前文时不再抢位置。

**不引 markdown 依赖** —— 回答来自自家模型、格式可控，只为几个标记装一整解析器不划算；
全部按纯字符串渲染，不注入 HTML。

## 测试

```bash
npm run test
```

覆盖纯逻辑（时间格式化、会话排序、富文本渲染）、**SSE 解析**与关键交互组件（主题切换）。
不追覆盖率数字，优先覆盖「改了容易悄悄坏」的地方 —— 比如 SSE 报文被网络切在两个 chunk 中间时，
必须在 buffer 里等下一块，否则 JSON 解析拿到半截字符串会静默丢事件，
表现就是「偶尔少一个 stage / 答案突然断掉」，靠肉眼几乎复现不出来。
