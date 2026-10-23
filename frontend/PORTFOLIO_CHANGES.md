# 前端求职作品增量说明

核验日期：2026-10-22（2026-10-23 随仓库提交复核）。范围：本仓库（本轮含一处后端数据契约修复，见第 2.D 节）。未读取 `.env`、私人语料或实际会话正文。

## 1. 原有能力（不是本次新做）

- React 18 + TypeScript + Vite；深浅主题、已有窄屏侧栏布局。
- POST fetch + ReadableStream 的 SSE 问答、分块报文拼接、阶段轨迹、token/reset/done；已有 AbortController 停止按钮和部分答案保留。
- 文档上传/删除/索引、引用原文展开、Markdown 回答渲染、反馈、重新生成。
- SQLite 服务端会话历史、新建会话、双击标题重命名、删除会话与模型记忆。
- 已有 Vitest：5 文件 / 23 测试；本次编辑前 `npm test` 与 `npm run lint` 均通过。

## 2. 本次三个增量

### A. 会话标题 / ID 搜索

- 新增本地子串过滤：去除首尾空格、不区分英文大小写，支持中文。
- 搜索作用于当前接口返回的会话摘要，不加载会话正文；明确显示“仅搜索标题和 ID”。
- 可见 / 总数量、无匹配空态与清除按钮；过滤不会改变当前选中的对话。
- 入口：[Sidebar](src/components/Sidebar.tsx)；纯函数：[filterSessions](src/lib/sessions.ts)。

### B. 当前会话导出 Markdown / JSON

- 聊天区工具栏提供两种本地下载；使用实际加载的消息数据，不生成演示回答。
- Markdown 保留用户问题、助手答案、引用文件/相对路径/页码/章节/chunk/原文和检索元信息；JSON 含 `schema_version: 1`、会话摘要、消息对象。
- Windows 文件名保留字符/控制字符清理；Blob URL 在下载触发后释放、临时节点移除。
- 空会话、历史未就绪、生成中、会话修改中禁用导出，避免导出过程中的不完整快照。
- 入口：[ChatPanel](src/components/ChatPanel.tsx)；序列化与下载：[serializeSession / downloadSession](src/lib/exportSession.ts)。

### C. 会话异步状态健壮性

- 修复刷新后 localStorage 选中会话只有 ID、没有自动加载历史的问题；改为统一 effect 加载。
- 区分“未加载”和“已加载但为空”的缓存；失败显示原因及重试按钮，不再当空会话默默略过。
- 历史未加载时禁止发送/导出；effect 清理标记忽略快速切换后的迟到响应，避免旧历史覆盖当前状态。
- 删除前确认；等待服务端成功才更新本地列表/缓存；失败保留原消息和会话并提示。重命名同样以接口成功返回标题为准。
- ref 同步闸门 + UI disabled：阻止重复会话修改；生成/会话修改期间阻止切换与删除，避免流式写入目标和用户看到的会话错位。
- 原有 AbortController 停止功能未重复实现；新增回归验证它与新的交互锁兼容。
- 实现：[App](src/App.tsx)、[Sidebar](src/components/Sidebar.tsx)、[mapHistory](src/lib/sessions.ts)。

### D. 历史会话引用证据丢失（前后端数据契约 bug）

这是本轮唯一一处改后端的地方，也是最值得在面试里讲的一条。

- 现象：流式回答刚结束时引用卡片正常显示；**重新打开这个会话，引用全部消失**，导出 Markdown 也没有引用块。
- 根因：`app/main.py` 的 `_record_turn` 入库时写的是 `citations=len(citations)`（数字），而前端 `mapHistory` 却按 `meta.citations` 是数组来读。数组和数字的可选链不会报错，只是静默拿到 `undefined` —— 典型的「不 crash、但数据没了」。
- 修法：后端 `_record_turn` 改为接收并入库**完整引用数组**，另存 `citation_count` 供统计；查询日志表仍存数字（`len()` 现算），两处口径不再漂移。前端 `mapHistory` 做兼容：`Array.isArray` 才当引用，旧数据里的数字直接忽略。
- 验证：后端 3 条回归测试（新数据存数组 / 日志仍存条数 / 空引用存 `[]`），前端 2 条（数字不被当数组 / 数组还原到顶层并从 `meta` 移除）；真机链路重跑一次问答后读历史接口，`meta.citations` 为 5 条数组；浏览器里打开历史会话，引用卡由 **0 条变为 5 条**。
- 入口：[main.py `_record_turn`](../../app/main.py)；[mapHistory](src/lib/sessions.ts)；[后端回归](../../tests/test_api.py)。

## 3. 接口依据

只读核对 `app/main.py` 的真实路由及 `app/storage.py` 查询，不改接口：

| 操作 | 现有接口 | 本次使用 |
| --- | --- | --- |
| 列表 | `GET /api/sessions` → `{ sessions }` | 本地过滤摘要 |
| 历史 | `GET /api/sessions/{sid}/messages` → `{ messages }` | 自动加载/重试/引用映射 |
| 重命名 | `PATCH /api/sessions/{sid}` → `{ title }` | 成功后更新标题 |
| 删除 | `DELETE /api/sessions/{sid}` | 成功后清理前端缓存 |
| 流式 | `POST /api/chat/stream` | 保留原协议及 AbortSignal |

本次自动化对 API 使用可控 mock，不调用模型、不修改运行中后端数据；所有测试问题/回答/引用都是明确的 fixture，不作真实业务数据展示。

## 4. 改动文件

修改：
- [App](src/App.tsx)：状态编排、自动恢复历史、修改锁、导出入口。
- [Sidebar](src/components/Sidebar.tsx)：搜索、数量/空态、确认/忙碌态与键盘事件隔离。
- [ChatPanel](src/components/ChatPanel.tsx)：导出工具栏、历史加载/错误/重试、发送禁用态。
- [sessions](src/lib/sessions.ts)：新增过滤与历史映射（含引用数组兼容），复用原有排序函数。
- [sessions 测试](src/lib/sessions.test.ts)：原有 2 条 + 新增 6 条。
- [types](src/types.ts)：`DoneMeta` 补 `citation_count` 说明。
- [样式](src/styles.css)：仅增加搜索/导出及状态样式；沿用深浅主题变量，工具栏和按钮组可换行，不声称重做整站响应式。
- [main.py](../../app/main.py)：`_record_turn` 入库完整引用数组（第 2.D 节）。
- [test_api.py](../../tests/test_api.py)：新增 3 条引用持久化回归。

新增：
- [会话导出工具](src/lib/exportSession.ts)。
- [导出测试](src/lib/exportSession.test.ts)：5 条。
- [App 集成测试](src/App.test.tsx)：10 条。
- [UI 验收脚本](../../scripts/verify-ui-detail.mjs)：CDP 无头浏览器跑真实页面，测主题切换、深色对比度、历史会话引用、窄屏溢出。
- [本说明](PORTFOLIO_CHANGES.md)。

依赖与 lockfile 未改。`npm run build` 重新生成现有被忽略的 `dist/` 构建产物。

## 5. 实测命令与结果

在前端目录执行：

```powershell
npm test
npm run build
npm run lint
npm run format:check
```

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node scripts\verify-ui-detail.mjs
```

- `npm test`：7 测试文件 / **44 测试全部通过**；原有 23，新增 **21**（搜索/历史映射 6、导出 5、App 集成 10）。
- `npm run build`：TypeScript 类型检查 + Vite 生产构建通过；46 modules transformed；JS 171.05 kB / gzip 57.18 kB，CSS 18.74 kB / gzip 4.46 kB。
- `npm run lint`：ESLint 通过，0 error / 0 warning。
- `npm run format:check`：通过，所有匹配文件符合 Prettier 格式。
- `pytest tests -q`：**86 passed**（原有 83 + 新增 3 条引用持久化回归）。
- `node scripts\verify-ui-detail.mjs`：真机无头 Chrome 跑真实页面。主题 `light → dark` 生效、`localStorage` 落盘；深色模式正文/次要文字对比度 7.42–15.56（WCAG AA 正文需 4.5）；历史会话引用卡 **5 条**（修复前 0 条）；移动端 390px 下横向溢出元素 **0**。

集成测试覆盖：首选历史恢复、空历史缓存、搜索/清空、双格式导出参数、加载错误与重试、延迟请求竞态、删除失败保留、取消/成功删除、删除等待时的重复操作与切换锁、原有停止生成兼容、重命名失败保留。集成测试 mock 文档面板与欢迎面板以隔离无关请求，实际渲染 Sidebar / ChatPanel / 消息。

中途发现并修复：文件名正则触发 ESLint `no-control-regex`，改为逐字符过滤控制字符；测试 DOM 缺系统主题 API，测试明确设置已存主题；下载测试补齐 `this: HTMLAnchorElement` 类型；后端 `_record_turn` 引用入库口径（第 2.D 节）。最终结论以末次完整 checks 为准，不以 shell 最后一条命令成功掩盖之前失败。

## 6. 本地启动与演示

```powershell
cd D:\Desktop\rag-demo
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

后端会直接托管已构建的 `frontend/dist/`，访问 `http://127.0.0.1:8000` 即可（本轮实测走的就是这条路径）。改前端后先 `cd frontend && npm run build` 再刷新。若要做热更新开发，另开 `cd frontend && npm run dev -- --host 127.0.0.1`，默认 `http://127.0.0.1:5173`，Vite 代理把 `/api` 转给 8000。

建议演示顺序：打开已有测试会话看引用证据 → 搜索标题/ID并清空 → 导出 Markdown / JSON 查看引用落盘 → 刷新看历史恢复 → 切深色主题 → 拉窄窗口看布局 → 最后用自动化测试讲解失败/竞态，而不是对现存个人会话做破坏性演示。

## 7. 边界与未验证项

- 搜索只覆盖当前列表里的标题/ID，不是正文全文检索；后端目前列表默认最多 200 会话，没有新增分页。
- 导出是“当前前端已加载会话”的快照，不是全库备份；历史接口默认最多返回按 ID 升序的前 500 条，本次没有改变这个后端限制，也没有声称全量历史导出。
- JSON 保留当前消息中可用的引用和元信息；服务端未返回的字段不补造。原有停止生成后的部分文本可能尚未持久化到服务端。
- 历史竞态防护忽略迟到响应，没有给 GET 历史请求新增 AbortController；网络传输仍可自然完成。
- 删除确认是原生 confirm；导出 Markdown 保留用户/模型原始正文，面向本地文件而非 HTML 发布。
- 浏览器/屏幕实测由 lead 统一负责；本子任务未声称已完成真浏览器、真实下载落盘、移动设备、上传/模型端到端测试。窄屏新增样式经过源码检查，实际视觉仍应浏览器验收。
- 原有长列表/历史分页、访问令牌 UI、重新生成历史问答关联等未扩入本轮。

## 8. 面试可以讲的真实贡献

> 我在已有 React + TypeScript RAG 前端上做增量，而不是把原有 SSE 和上传算成新开发：增加了会话搜索和双格式本地导出，处理引用数据保真与 Blob 生命周期；同时修复刷新恢复历史、请求竞态和删除失败丢状态，用可控延迟/错误 mock 做集成回归。测试从 23 条增加到 42 条，完成类型构建与 lint 验证。

可以展开说明：为什么用 `undefined` 区分未加载与空历史、effect 清理与迟到响应、同步 ref 防连点、先确认服务端成功再更新 UI 的取舍，以及为何明确说明搜索/导出的后端数据量限制。
