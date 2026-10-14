// frontend/src/components/Sidebar.tsx
// 左侧栏：品牌头 + 会话列表 + 知识库面板 + 状态灯

import { useState } from 'react'
import type { DocumentsInfo, IndexReport, ReadyInfo, SessionSummary } from '../types'
import DocumentsPanel from './DocumentsPanel'

interface Props {
  sessions: SessionSummary[]
  currentSid: string
  kb: DocumentsInfo | null
  health: { status: string; has_api_key: boolean } | null
  ready: ReadyInfo | null
  busy: boolean
  onSelect: (sid: string) => void
  onNew: () => void
  onDelete: (sid: string) => void
  onRename: (sid: string, title: string) => void
  onRefreshKb: () => void
  onUpload: (file: File, subdir: string) => Promise<IndexReport | void>
  onDeleteDoc: (relPath: string) => Promise<void>
  onReindex: (force: boolean) => Promise<void>
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
  ready,
  busy,
  onSelect,
  onNew,
  onDelete,
  onRename,
  onRefreshKb,
  onUpload,
  onDeleteDoc,
  onReindex,
}: Props) {
  const [editing, setEditing] = useState<string>('')
  const [draft, setDraft] = useState<string>('')
  const [open, setOpen] = useState(false)

  const apiOk = health?.has_api_key === true
  const backendDown = health?.status === 'down'
  const indexStale = ready?.checks?.index?.ok === false || kb?.index?.stale === true

  const commitRename = (sid: string) => {
    const title = draft.trim()
    if (title) onRename(sid, title)
    setEditing('')
    setDraft('')
  }

  return (
    <>
      <button
        className="sidebar-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? '收起侧栏' : '展开侧栏'}
        aria-expanded={open}
      >
        ☰
      </button>
      <aside className={`sidebar${open ? ' open' : ''}`}>
        <header className="sidebar-head">
          <div className="brand">
            <span className="brand-mark">◈</span>
            <div className="brand-text">
              <h1>科研知识库 Agent</h1>
              <p>可溯源问答 · 混合检索 + 引用核验</p>
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
              onKeyDown={(e) => {
                // 键盘可达性：Enter 与空格都应触发（原先只处理了 Enter）
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelect(s.id)
                }
              }}
            >
              <div className="session-main">
                {editing === s.id ? (
                  <input
                    className="session-rename"
                    value={draft}
                    autoFocus
                    aria-label="重命名会话"
                    onChange={(e) => setDraft(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    onBlur={() => commitRename(s.id)}
                    onKeyDown={(e) => {
                      e.stopPropagation()
                      if (e.key === 'Enter') commitRename(s.id)
                      if (e.key === 'Escape') setEditing('')
                    }}
                  />
                ) : (
                  <span
                    className="session-title"
                    onDoubleClick={(e) => {
                      e.stopPropagation()
                      setEditing(s.id)
                      setDraft(s.title || '')
                    }}
                    title="双击重命名"
                  >
                    {s.title || '新会话'}
                  </span>
                )}
                <span className="session-sub">
                  {fmtTime(s.last_active)}
                  {s.turns > 0 && ` · ${s.turns} 轮`}
                </span>
              </div>
              <button
                className="session-del"
                title="删除会话（同时清除模型记忆）"
                aria-label={`删除会话 ${s.title || ''}`}
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
          <DocumentsPanel
            kb={kb}
            onRefresh={onRefreshKb}
            onUpload={onUpload}
            onDelete={onDeleteDoc}
            onReindex={onReindex}
            busy={busy}
          />

          <div className={`status ${backendDown ? 'bad' : apiOk ? 'ok' : 'warn'}`}>
            <span className="status-dot" />
            {backendDown
              ? '后端未连接'
              : apiOk
                ? indexStale
                  ? '模型正常 · 索引待更新'
                  : '模型与索引正常'
                : '模型未配置（缺 API Key）'}
          </div>
        </footer>
      </aside>
    </>
  )
}
