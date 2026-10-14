# 科研知识库 AI Agent 求职版改造计划

## 1. 项目定位

将现有光学 RAG Demo 改造成一个可公开复现、可上传文档、可验证引用、可持续评测、可一键部署的科研知识库 AI Agent。

核心叙事不是“接了大模型接口”，而是：

1. 能处理科研文档并增量构建索引；
2. 能用混合检索与重排找到证据；
3. 能返回结构化、可点击、可核验的引用；
4. 能用领域工具进行计算和事实核验；
5. 有独立测试集、自动化测试、监控和部署闭环。

## 2. 参考项目与取舍

- [Production RAG](https://github.com/sugeerth/production-rag)：参考上传、增量索引、拒答阈值、查询日志和反馈闭环。
- [Kotaemon](https://github.com/Cinnamon/kotaemon)：参考引用预览、文档集合、多用户隔离和多种推理流程；不照搬其大型框架。
- [R2R](https://github.com/SciPhi-AI/R2R)：参考文档管理、REST API 和生产级检索服务分层。
- [RAG Knowledge Search](https://github.com/usiere/rag-knowledge-search)：参考 FastAPI + React 的清晰目录与结构化 sources 响应。

项目坚持“小而完整”：保留现有 FastAPI、React、LangGraph、Chroma 与 BGE 技术栈，不引入微服务或知识图谱等暂时无法证明收益的复杂组件。

## 3. 目标架构

```text
React Web
├─ Chat：流式回答、引用卡片、反馈、重试
├─ Documents：上传、状态、删除、重新索引
└─ Observability：知识库状态与本地运行指标
        │
        ▼
FastAPI
├─ /api/chat/stream
├─ /api/documents
├─ /api/documents/upload
├─ /api/documents/{id}/reindex
├─ /api/feedback
└─ /health + /ready
        │
        ├─ Conversation Service（SQLite 持久会话）
        ├─ Ingestion Service（解析、切块、增量/原子索引）
        ├─ Retrieval Service（Dense + BM25 + RRF + Rerank）
        ├─ Citation Service（claim/source/chunk/quote）
        └─ Agent Graph
           ├─ 查询改写
           ├─ 澄清/拆分
           ├─ 知识库检索工具
           ├─ 光学参数计算工具
           └─ 证据核验工具
```

## 4. 分阶段交付

### Phase 0：正确性与可复现性

- 修复 package 模式第二轮导入故障；
- 查询改写真正进入检索；
- 区分会话状态与单轮状态，消除跨轮污染；
- 统一配置读取，进程环境变量优先于 `.env`；
- 修复测试入口和 Windows UTF-8 输出；
- 修复 README 数字漂移。

验收：同一会话连续提问 10 轮不串答案；根目录和容器启动方式一致；配置可由环境变量覆盖。

### Phase 1：文档与索引闭环

- 支持 Markdown、TXT、CSV、PDF、DOCX；
- 文档上传、列表、删除、重新索引；
- 内容 Hash、索引 manifest、增量更新；
- staging 构建后原子切换；
- 提供脱敏公开示例语料。

验收：上传文件后无需重启即可查询；修改同一文件只更新受影响内容；失败不会破坏旧索引。

### Phase 2：可信回答与 Agent 工具

- 返回结构化 citations；
- 前端引用卡片与原文定位；
- 相关性阈值与无证据拒答；
- 光学参数计算工具；
- 证据/数值核验工具；
- Prompt Injection 基础防护。

验收：回答中的关键结论能映射到 source、section、chunk_id 和 quote；无证据问题明确拒答。

### Phase 3：持久化与产品体验

- SQLite 会话、消息、文档、反馈；
- 历史查询、删除、导出；
- 上传与索引进度；
- 反馈、复制、重新生成；
- 移动端和无障碍改进。

### Phase 4：质量与交付

- pytest 单元/集成测试；
- 前端组件与端到端测试；
- 独立调参集/测试集；
- citation precision、拒答率、延迟和成本指标；
- Docker Compose 一键启动；
- GitHub Actions CI；
- README、架构图、演示视频脚本、面试追问材料。

## 5. 面试验收标准

1. 克隆仓库后可以使用脱敏语料一键运行；
2. 面试官上传一个 PDF 后可以立即提问；
3. 每个回答都有可点击证据；
4. 多轮对话不串状态，服务重启后历史仍在；
5. 自动化测试和评测结果可复现；
6. 能解释检索、重排、引用、拒答、索引一致性和部署中的关键取舍。
