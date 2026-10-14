// frontend/src/components/DocumentsPanel.tsx
// 知识库面板：上传、列表、删除、重建索引、索引进度。
//
// 设计取舍：不做独立路由页，而是侧栏内的可折叠面板。
// 演示场景下用户最常做的是「传一份材料 → 立刻提问」，放在同一屏里路径最短。

import { useRef, useState } from 'react'
import type { DocumentsInfo, IndexReport } from '../types'

interface Props {
  kb: DocumentsInfo | null
  onRefresh: () => void
  onUpload: (file: File, subdir: string) => Promise<IndexReport | void>
  onDelete: (relPath: string) => Promise<void>
  onReindex: (force: boolean) => Promise<void>
  busy: boolean
}

export default function DocumentsPanel({
  kb,
  onRefresh,
  onUpload,
  onDelete,
  onReindex,
  busy,
}: Props) {
  const [open, setOpen] = useState(false)
  const [subdir, setSubdir] = useState('')
  const [status, setStatus] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [working, setWorking] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const documents = kb?.documents ?? []
  const index = kb?.index

  const pick = () => fileRef.current?.click()

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return
    setWorking(true)
    setError('')
    try {
      for (const file of Array.from(files)) {
        setStatus(`正在上传并重建索引：${file.name}（首次重建需加载嵌入模型，请耐心等待）`)
        const report = await onUpload(file, subdir.trim())
        if (report && !report.ok) {
          setError(`索引未更新：${report.warnings.join('；') || '未知原因'}`)
          setStatus('文件已保存，但索引构建失败')
        } else if (report) {
          setStatus(
            `索引已更新：${report.parsed_files} 个文件 / ${report.parent_count} 父块 / ` +
              `${report.child_count} 子块（${report.elapsed_seconds}s）`,
          )
        } else {
          setStatus('上传完成')
        }
      }
      onRefresh()
    } catch (e) {
      setError((e as Error).message)
      setStatus('')
    } finally {
      setWorking(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const handleReindex = async (force: boolean) => {
    setWorking(true)
    setError('')
    setStatus(force ? '正在全量重建索引…' : '正在检查语料变化…')
    try {
      await onReindex(force)
      onRefresh()
      setStatus('索引已更新')
    } catch (e) {
      setError((e as Error).message)
      setStatus('')
    } finally {
      setWorking(false)
    }
  }

  const statusTone =
    index?.status === 'ok' ? 'ok' : index?.status === 'missing' ? 'bad' : 'warn'

  return (
    <section className={`doc-panel${open ? ' open' : ''}`}>
      <button
        className="doc-panel-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>知识库</span>
        <span className="doc-panel-badge">{documents.length} 篇</span>
        <span className="doc-panel-caret">{open ? '▾' : '▸'}</span>
      </button>

      {!open ? (
        <div className="kb-card compact">
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
          <div className={`kb-row idx ${statusTone}`}>
            <span>索引状态</span>
            <b>{indexLabel(index?.status)}</b>
          </div>
        </div>
      ) : (
        <div className="doc-panel-body">
          <div className={`index-banner ${statusTone}`}>
            <b>{indexLabel(index?.status)}</b>
            <span>{index?.detail || '—'}</span>
            {index?.built_at ? <span className="index-time">构建于 {index.built_at}</span> : null}
          </div>

          <div className="doc-toolbar">
            <input
              ref={fileRef}
              type="file"
              multiple
              accept=".md,.markdown,.txt,.csv,.pdf,.docx"
              onChange={(e) => void handleFiles(e.target.files)}
              style={{ display: 'none' }}
            />
            <button className="btn-mini" onClick={pick} disabled={working || busy}>
              上传文档
            </button>
            <button
              className="btn-mini"
              onClick={() => void handleReindex(false)}
              disabled={working || busy}
              title="仅在语料变化时重建"
            >
              检查更新
            </button>
            <button
              className="btn-mini"
              onClick={() => void handleReindex(true)}
              disabled={working || busy}
              title="忽略增量判断，强制全量重建"
            >
              全量重建
            </button>
          </div>

          <input
            className="doc-subdir"
            value={subdir}
            onChange={(e) => setSubdir(e.target.value)}
            placeholder="可选：上传到子目录，如 实验报告"
            aria-label="上传子目录"
          />

          {working || status ? <div className="doc-status">{status}</div> : null}
          {error ? <div className="doc-error">⚠ {error}</div> : null}

          <ul className="doc-list">
            {documents.length === 0 && (
              <li className="doc-empty">
                还没有文档。上传 .md / .txt / .csv / .pdf / .docx 后即可提问。
              </li>
            )}
            {documents.map((rel) => (
              <li key={rel} className="doc-item">
                <span className="doc-name" title={rel}>
                  {rel}
                </span>
                <button
                  className="doc-del"
                  title="删除该文档并重建索引"
                  disabled={working || busy}
                  onClick={async () => {
                    setWorking(true)
                    setError('')
                    try {
                      await onDelete(rel)
                      onRefresh()
                      setStatus(`已删除 ${rel} 并重建索引`)
                    } catch (e) {
                      setError((e as Error).message)
                    } finally {
                      setWorking(false)
                    }
                  }}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>

          <div className="doc-foot">
            支持 .md / .txt / .csv / .pdf / .docx，单文件上限 20 MB
          </div>
        </div>
      )}
    </section>
  )
}

function indexLabel(status?: string): string {
  switch (status) {
    case 'ok':
      return '已同步'
    case 'stale':
      return '有变更未重建'
    case 'missing':
      return '尚未建立'
    case 'incompatible':
      return '需重建'
    default:
      return '未知'
  }
}
