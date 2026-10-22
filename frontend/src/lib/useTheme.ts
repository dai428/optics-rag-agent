// frontend/src/lib/useTheme.ts
// 深浅主题管理 Hook。
//  - 首次默认跟随系统（prefers-color-scheme）
//  - 用户手动切换后记忆到 localStorage
//  - 返回 { theme, toggleTheme }，并负责给 <html> 挂 data-theme 属性

import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

const LS_THEME = 'optics-rag-agent:theme:v1'

function systemTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function loadTheme(): Theme {
  try {
    const saved = localStorage.getItem(LS_THEME)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* localStorage 不可用时回退 */
  }
  return systemTheme()
}

export function useTheme(): { theme: Theme; toggleTheme: () => void } {
  const [theme, setTheme] = useState<Theme>(loadTheme)

  // 手册切到 dark 后，跟随系统的监听就没意义了；只有"未手动指定"才随系统变。
  useEffect(() => {
    const saved = (() => {
      try {
        return localStorage.getItem(LS_THEME)
      } catch {
        return null
      }
    })()
    if (saved !== null) return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = (e: MediaQueryListEvent) => setTheme(e.matches ? 'dark' : 'light')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      localStorage.setItem(LS_THEME, theme)
    } catch {
      /* 忽略 */
    }
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'light' ? 'dark' : 'light'))
  }, [])

  return { theme, toggleTheme }
}
