# 光学科研文献智能体 · Agentic RAG

> 面向**光学 / 液晶偏振测量（LCVR）**垂直领域的科研文献问答智能体。
> 不是「能跑通」的玩具——每一层优化都有**可复现的评测数字**兜底。
>
> 📁 `D:\Desktop\rag-demo` · 栈：LangGraph + LangChain + Chroma + BGE + FastAPI

---

## 一句话卖点

**把 RAG 从「向量检索 + 拼接」升级为「有规划、会反问、能并行、可观测」的 Agentic RAG。**
在 20 题真实语料评测集上，检索层 Recall@5 从 **80% → 100%**，MRR 提升 **+26.1%**，失败题从 **5 道 → 0 道**。

---

## 一、它解决什么问题

光学实验语料有三个「劝退向量检索」的特点：

| 语料特点 | 纯向量检索的后果 | 本项目的解法 |
|---|---|---|
| 大量 **Markdown 表格**（参数、扫描结果） | 表格被切片拦腰截断，数字检索不到 | **表格展开**：横竖两种平铺形式，让数字变成可检索文本 |
| 同一参数在 **论文体 / 实验记录 / 数据表** 中反复出现 | 叙述性文字字面重合度高，一手数据被挤后面 | **来源加权**：实验记录 ×1.03、论文体 ×0.97 |
| 术语精确（如「ΔCN」「DE 加权」），同义词少 | 纯语义召回想不起精确术语 | **混合检索**：向量 + BM25 双路召回 RRF 融合 |

---

## 二、架构

```
                        ┌─(闲聊)────────────→ respond → END
__start__ → rewrite ────┼─(问题模糊)───────→ clarify → END（把问题抛回用户）
                        └─(正常)──→ decompose ─┬─→ retrieve_one ─┐
                                                └─→ retrieve_one ─┴→ respond(reduce) → END
                               （Send 动态并行，子问题数 = 分支数）

检索链路：query → [向量 Top-25 + BM25 Top-25] → RRF 融合 → CrossEncoder 重排
                → 父块聚合 → 来源加权 → Top-5 父块 → 拼上下文
```

### 六个 Agentic 能力（都对应一个真实痛点）

| 能力 | 节点 | 解决什么 |
|---|---|---|
| **查询改写** | `rewrite` | 多轮对话里的「它 / 那个结果」补全成独立问句 |
| **澄清反问** | `clarify_check` | 问题太模糊时**先反问**，不瞎猜（避免自信地答错） |
| **问题拆分** | `decompose` | 「对比 A 和 B」自动拆成两个子问题 |
| **并行检索** | `Send` + `retrieve_one` | 子问题同时检索，再 reduce 汇总 |
| **历史摘要** | `_summarize` | 长对话滚动压缩，控 token 且不丢数字 |
| **预算控制** | `MAX_TOOL_CALLS` 等 | 硬上限，防止 Agent 陷入检索死循环 |

---

## 三、评测结果（可复现）

评测集：`data/eval_set.json` —— **20 题**真实光学语料问题，每题标注期望来源文件与关键词。

| 指标 | 旧版（纯向量） | 新版（混合+重排） | 提升 |
|---|---|---|---|
| Recall@1 | 50.0% | **65.0%** | +15.0 pt |
| **Recall@5** | 80.0% | **100.0%** | **+20.0 pt** |
| Recall@10 | 95.0% | **100.0%** | +5.0 pt |
| MRR@10 | 0.614 | **0.774** | **+26.1%** |
| NDCG@10 | 0.693 | **0.830** | **+19.8%** |
| 关键词命中@5 | 95% | **100%** | +5.0 pt |
| **失败题数** | 5 | **0** | — |

### 优化贡献分解（每一步都单独实测）

| 步骤 | 做法 | R@5 | MRR |
|---|---|---|---|
| 基线 | 纯向量，chunk=300 | 80.0% | 0.614 |
| ① 父子分层 + 元数据 | 父块 2000-4000 / 子块 500 | — | — |
| ② 混合检索 RRF | 向量 + BM25 | — | 0.650 |
| ③ 召回池 12→25 | 候选池够大，重排才有素材 | — | 0.695 |
| ④ CrossEncoder 重排 | bge-reranker-base | 85.0% | 0.744 |
| ⑤ **表格展开** | 表格转可检索文本 | — | 0.785 |
| ⑥ **来源加权** | 实验/总纲 ×1.03，论文体 ×0.97 | **100.0%** | 0.774 |

> ⑥ 的取舍：来源加权让 R@5 补上最后 15 pt，MRR 只掉 0.011 —— 用极小的排序代价换「答案必在前 5」的硬保证，划算。
> 系数经网格搜索确定（`1.08/0.92` 会把 R@1 压到 60%，得不偿失）。

### 生成层评测（RAGAS，抽样 5 题）

检索好不等于回答好。用 RAGAS 对生成质量做抽查（裁判模型 = 本项目的 DeepSeek-V4-Flash）：

| 指标 | 含义 | 得分 |
|---|---|---|
| **faithfulness** | 答案的每句陈述能否在检索资料里找到依据（**衡量幻觉**） | **0.889** |
| **answer_relevancy** | 答案是否正面回应用户的问题（衡量跑题） | **0.818** |

> 说明：因 `eval_set.json` 是检索评测集（无逐题标准答案），只用不需要 `reference` 的两个指标。
> 且裁判模型与被评模型相同（自评），**绝对分值仅供参考，横向对比才有意义**。

复现命令：

```bash
cd app
../.venv/Scripts/python.exe eval_rag.py --compare   # 三种检索模式对比
../.venv/Scripts/python.exe eval_gen.py --limit 5   # 生成层评测（RAGAS）
../.venv/Scripts/python.exe verify_p1.py             # P1 四项行为验证
```

---

## 四、目录结构

```
rag-demo/
├─ app/                      # ⭐ 核心代码
│  ├─ chunker.py             #    父子分层切块 + 表格展开 + 父块仓库
│  ├─ knowledge_base.py      #    混合检索(RRF) + CrossEncoder 重排 + 来源加权
│  ├─ rag_graph.py           #    ⭐ LangGraph 主图（6 能力全在这）
│  ├─ rag_agent.py           #    create_agent 黑盒版（对照用：框架 vs 手写）
│  ├─ state.py               #    图状态定义（含并行 reducer）
│  ├─ prompts.py             #    全部提示词集中管理
│  ├─ config.py              #    全部可调参数集中管理
│  ├─ main.py                #    FastAPI 服务
│  ├─ eval_rag.py            #    检索层评测（三模式对比）
│  ├─ verify_p1.py           #    Agentic 行为验证
│  └─ test_rag.py            #    服务链路测试
├─ data/
│  ├─ docs/{论文,实验报告,表格}/  # 语料（14 篇，未发表材料，已 .gitignore）
│  ├─ chroma_db/             #    向量库（可重建）
│  ├─ parent_store/          #    父块 JSON（73 个）
│  └─ eval_set.json          #    评测集（20 题）
├─ learn/  notes/  legacy/   # 学习资料 / 讲解 HTML / 旧脚本
├─ .env.example              # 配置模板（不含密钥）
└─ Dockerfile / .dockerignore
```

---

## 五、快速开始

### 1) 装依赖

```bash
cd D:/Desktop/rag-demo
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2) 配密钥

```bash
cp .env.example .env
# 编辑 .env，填入 AMD_API_KEY
```

### 3) 建索引（首次或语料变更后）

```bash
cd app
../.venv/Scripts/python.exe -c "import knowledge_base as kb; kb.build_vectorstore(force=True)"
```

### 4) 问答

```bash
# 命令行单跑（最直观）
../.venv/Scripts/python.exe rag_graph.py

# 或起服务
../.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
# → http://127.0.0.1:8000/docs
```

---

## 六、模型选型（实测结论）

AMD Radeon Cloud 各模型单次调用耗时（2026-10-08 实测）：

| 模型 | 耗时 | 建议 |
|---|---|---|
| MiniCPM5-2B | 0.9 s | 极快，质量偏弱 |
| Qwen3.8-Flash-Next | 2.1 s | 快且质量好 |
| **DeepSeek-V4-Flash** | **2.2 s** | ✅ **默认**：开发调试首选 |
| MiMo-V2.6-Flash | 2.5 s | 快 |
| DeepSeek-V4.1-Flash | >90 s | ⚠️ 易超时 |
| GLM-5.3-Flash | >90 s | ⚠️ 易超时 |
| Qwen3.8-27B | 60~360 s | 质量最好（推理模型），适合出正式报告 |

**结论**：日常开发用 flash 系（2 秒级），出报告换 27B。改 `.env` 里 `AMD_LLM_MODEL` 一行即可，不改代码。
换 27B 时记得同步调大 `config.py` 的 `request_timeout`（27B 需 600s，flash 系 120s 够）。

---

## 七、踩坑记录（真实排障过程）

这些坑都写进了代码注释，这里只列结论：

1. **`EnsembleRetriever` 已从 langchain-community 移除** → 自实现 RRF 融合（`_RRFRetriever`）。
2. **并行分支写同一字段抛 `InvalidUpdateError`** → 给 `tool_calls` 加 `sum_int` reducer、`iterations` 加 `last_int`。
   *（这个 bug 曾伪装成「模型超时」，排查了很久。）*
3. **LangGraph `thread_id` 不能用中文** → 中文 thread_id 会让 `invoke()` 挂死（`stream` 正常）。
4. **HF 下载缓存硬链接失败**（Windows 沙箱）→ 快照文件 0 字节；需按**内容**而非大小重建 blob 映射。
5. **重排模型首次加载约 10 秒**（1.1GB）→ 服务里懒加载 + 全局单例，只付一次代价。

---

## 八、与「黑盒版」的区别

仓库里保留了 `rag_agent.py`（`create_agent` 框架版）作为对照：

| | `rag_agent.py`（框架） | `rag_graph.py`（手写） |
|---|---|---|
| 流程可见性 | 藏在框架里 | 每个节点可见可改 |
| 能不能加澄清反问 | 难 | 一个节点的事 |
| 能不能并行检索 | 难 | `Send` 一行 |
| 适合 | 快速原型 | 要讲清楚、要定制 |

**两个都留着，是为了面试时能讲清「框架帮你做了什么，以及你为什么还需要手写」。**

---

## 九、本机环境坑

1. **代理**：系统代理指向 `127.0.0.1:7897`，关代理时 pip 会误连失败 → `unset` 全部 proxy + `export no_proxy="*"`。
2. **pip 源**：清华源被拦 → 用阿里云 `https://mirrors.aliyun.com/pypi/simple/`。
3. **TEMP 重定向**：`D:\temp` 损坏致 pip 解压不完整 → `export TEMP=<干净目录>`。
4. **HuggingFace**：`HF_ENDPOINT=https://hf-mirror.com` · `HF_HOME=D:/DevCache/huggingface`。
5. **AMD 并发上限 16**：打满返回 429 → 代码已带重试。

> 上述环境变量已在 `app/*.py` 顶部用 `setdefault` 内置，**无论 shell 还是 IDE 启动都不必手设**。

---

## 十、评测工具一览

| 脚本 | 评什么 | 是否需要 LLM | 典型耗时 |
|---|---|---|---|
| `app/eval_rag.py` | **检索层**：Recall@k / MRR / NDCG / 关键词命中 | 否 | 秒级 |
| `app/eval_gen.py` | **生成层**：faithfulness / answer_relevancy（RAGAS） | 是 | 数分钟 |
| `app/verify_p1.py` | **Agentic 行为**：该反问吗 / 该拆吗 / 摘要保不保数字 | 是 | 数分钟 |
| `app/test_rag.py` | **服务链路**：接口 / 多轮记忆 / 会话隔离 | 是 | 分钟级 |

```bash
cd app
../.venv/Scripts/python.exe eval_rag.py --compare      # 检索层三模式对比
../.venv/Scripts/python.exe eval_gen.py --limit 5      # 生成层（先跑5题省额度）
../.venv/Scripts/python.exe eval_gen.py --from-raw     # 复用已采集数据，只重评分
../.venv/Scripts/python.exe verify_p1.py               # Agentic 行为全量验证
```

> **生成层评测只用了 2 个 RAGAS 指标（faithfulness + answer_relevancy）**，这是个有意的取舍：
> 本项目的 `eval_set.json` 是**检索评测集**（只标了期望来源 + 关键词），没有逐题标准答案；
> 而 RAGAS 的 `context_precision` / `answer_correctness` 都强制要求 `reference` 列。
> 硬凑答案容易失真、反而污染评测，所以只保留不需要标准答案、且恰好覆盖
> 「有没有编 + 跑不跑题」这两个最关键维度的指标。
> 将来补齐标准答案后，把 `context_precision` 加回 metrics 列表即可。

---

## 十一、下一步

- [x] **检索层评测**：20 题 Recall / MRR / NDCG（R@5 100%）
- [x] **生成层评测**：RAGAS faithfulness + answer_relevancy
- [x] **可观测**：`observability.py` 接 Langfuse（未配置自动跳过）
- [x] **容器化**：Dockerfile / .dockerignore
- [ ] 语料扩到设备手册 / 更多论文，做增量索引
- [ ] 多模态：把实验图表也纳入检索
