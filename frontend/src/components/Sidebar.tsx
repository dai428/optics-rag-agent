// frontend/src/components/Sidebar.tsx
// 左侧栏：品牌头 + 会话列表 + 知识库概况

import type { DocumentsInfo, SessionSummary } from '../types'

interface Props {
  sessions: SessionSummary[]
  currentSid: string
  kb: DocumentsInfo | null
  health: { status: string; has_api_key: boolean } | null
  busy: boolean
  onSelect: (sid: string) => void
  onNew: () => void
  onDelete: (sid: string) => void
}

/** 今天只显示时分，更早显示月/日 —— 列表窄，不占地方。
 *  入参是后端给的 ISO 字符串（无时区），JS 会按本地时间解析。 */
function fmtTime(iso: string): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const sameDay = d.toDateString() === new Date().toDateString()
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  return sameDay ? `${hh}:${mm}` : `${d.getMonth() + 1}/${d.getDate()}`
}

export default function Sidebar({
  sessions,
  currentSid,
  kb,
  health,
  busy,
  onSelect,
  onNew,
  onDelete,
}: Props) {
  const apiOk = health?.has_api_key === true

  return (
    <aside className="sidebar">
      <header className="sidebar-head">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <div className="brand-text">
            <h1>科研文献 Agent</h1>
            <p>LCVR 液晶偏振光学 · 检索增强问答</p>
          </div>
        </div>
        <button className="btn-new" onClick={onNew} disabled={busy} title="开启一段新对话">
          ＋ 新建会话
        </button>
      </header>

      <nav className="session-list">
        {sessions.length === 0 && (
          <p className="empty-tip">还没有会话，直接在右侧提问即可开始</p>
        )}
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`session-item${s.id === currentSid ? ' active' : ''}`}
            onClick={() => onSelect(s.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onSelect(s.id)}
          >
            <div className="session-main">
              <span className="session-title">{s.title || '新会话'}</span>
              <span className="session-sub">
                {fmtTime(s.last_active)}
                {s.turns > 0 && ` · ${s.turns} 轮`}
              </span>
            </div>
            <button
              className="session-del"
              title="删除会话（同时清除模型记忆）"
              onClick={(e) => {
                e.stopPropagation()
                onDelete(s.id)
              }}
            >
              ×
            </button>
          </div>
        ))}
      </nav>

      <footer className="sidebar-foot">
        <div className="kb-card">
          <div className="kb-row">
            <span>语料文件</span>
            <b>{kb ? kb.documents.length : '—'}</b>
          </div>
          <div className="kb-row">
            <span>向量切片</span>
            <b>{kb ? kb.chunks : '—'}</b>
          </div>
          <div className="kb-row">
            <span>检索方式</span>
            <b>{kb ? (kb.retrieval.mode.includes('hybrid') ? '混合检索' : kb.retrieval.mode) : '—'}</b>
          </div>
          <div className="kb-row">
            <span>重排模型</span>
            <b>{kb?.retrieval.rerank_model ? '已启用' : '未启用'}</b>
          </div>
        </div>
        <div className={`status ${apiOk ? 'ok' : 'bad'}`}>
          <span className="status-dot" />
          {apiOk ? '模型服务正常' : '模型未配置（缺 API Key）'}
        </div>
      </footer>
    </aside>
  )
}
