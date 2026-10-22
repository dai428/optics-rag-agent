// frontend/src/components/ThemeToggle.tsx
// 深浅主题切换按钮：侧栏品牌区右上角。
// 只显示图标（空间紧张），语义靠 aria-label / title 提供，兼顾无障碍。

import type { Theme } from '../lib/useTheme'

interface Props {
  theme: Theme
  onToggle: () => void
}

export default function ThemeToggle({ theme, onToggle }: Props) {
  const next = theme === 'dark' ? '浅色' : '深色'
  return (
    <button
      className="theme-toggle"
      onClick={onToggle}
      aria-label={`切换到${next}模式`}
      title={`切换到${next}模式`}
    >
      <span className="theme-icon" aria-hidden>
        {theme === 'dark' ? '☀' : '◐'}
      </span>
    </button>
  )
}
