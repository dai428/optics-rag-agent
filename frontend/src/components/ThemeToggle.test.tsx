// frontend/src/components/ThemeToggle.test.tsx
// 主题切换按钮：渲染浅色时提示切到深色，点击触发 onToggle。

import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ThemeToggle from './ThemeToggle'

describe('ThemeToggle', () => {
  it('浅色主题下显示「深色」提示', () => {
    render(<ThemeToggle theme="light" onToggle={() => {}} />)
    expect(screen.getByTitle('切换到深色模式')).toBeInTheDocument()
  })

  it('深色主题下显示「浅色」提示', () => {
    render(<ThemeToggle theme="dark" onToggle={() => {}} />)
    expect(screen.getByTitle('切换到浅色模式')).toBeInTheDocument()
  })

  it('点击触发 onToggle', () => {
    const fn = vi.fn()
    render(<ThemeToggle theme="light" onToggle={fn} />)
    fireEvent.click(screen.getByRole('button'))
    expect(fn).toHaveBeenCalledTimes(1)
  })
})
