// frontend/src/types.ts
// 全局类型定义：与后端 /api/** 的返回结构一一对应

/** done 事件携带的元信息 */
export interface DoneMeta {
  /** 本轮累计命中资料条数 */
  retrieved_count?: number
  /** 拆分出的子问题（未拆分时为空数组） */
  sub_questions?: string[]
  /** 查询改写后的检索问句 */
  rewritten_query?: string | null
  /** 是否真的检索了资料（闲聊时为 false） */
  used_retrieval?: boolean
  /** 后端统计的总耗时（秒） */
  elapsed?: number
}

/** 一条聊天消息（仅存前端，见 App.tsx 的 localStorage 说明） */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  /** 流式进行中：true 时显示闪烁光标 */
  streaming?: boolean
  /** 出错时的原因，渲染在气泡下方 */
  error?: string
  /** 本轮回答的元信息 */
  meta?: DoneMeta
  /** 本轮的执行轨迹（节点时间轴），用于「过程可解释」 */
  stages?: StageItem[]
}

/** 左栏会话列表项（字段与后端 /api/sessions 的返回一一对应） */
export interface SessionSummary {
  /** 会话 id，同时用作 LangGraph 的 thread_id（后端字段名就叫 id） */
  id: string
  title: string
  /** ISO 字符串，形如 2026-10-08T18:41:12（无时区，按本地时间解析） */
  created_at: string
  last_active: string
  turns: number
}

/** 一个图节点：已完成（stage）或正在执行（pending 预告） */
export interface StageItem {
  stage: string
  label: string
  detail?: string
  elapsed?: number
  /** true = 这是「下一步预告」，还没跑完 */
  pending?: boolean
}

/** GET /api/documents —— 知识库概况 */
export interface DocumentsInfo {
  /** 语料文件名列表 */
  documents: string[]
  /** 向量库切片数 */
  chunks: number
  /** 父块数（分层切块的父块仓库） */
  parent_chunks: number
  /** 嵌入模型名 */
  embed_model: string
  /** 检索配置（混合检索权重 + 重排模型） */
  retrieval: {
    mode: string
    dense_weight: number
    sparse_weight: number
    rerank_model: string | null
    retrieval_k: number
  }
  chroma_dir: string
}
