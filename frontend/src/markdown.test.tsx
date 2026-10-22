// frontend/src/markdown.test.tsx
// 富文本渲染器的纯逻辑测试：验证粗体、行内代码、列表、标题被正确转换。

import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { renderRich } from './markdown'

function renderText(src: string) {
  return render(<div>{renderRich(src)}</div>)
}

describe('renderRich', () => {
  it('渲染粗体', () => {
    const { container } = renderText('这是**重点**内容')
    const strong = screen.getByText('重点')
    expect(strong.tagName).toBe('STRONG')
    // 段落整体文本应保留粗体两侧的内容
    expect(container.querySelector('.md-p')?.textContent).toBe('这是重点内容')
  })

  it('渲染行内代码（不注入 HTML）', () => {
    renderText('命令 `rm -rf` 危险')
    const code = screen.getByText('rm -rf')
    expect(code.tagName).toBe('CODE')
  })

  it('渲染无序列表', () => {
    renderText('- 第一项\n- 第二项')
    const items = screen.getAllByRole('listitem')
    expect(items).toHaveLength(2)
  })

  it('渲染有序列表', () => {
    renderText('1. 选项A\n2. 选项B')
    const items = screen.getAllByRole('listitem')
    expect(items).toHaveLength(2)
  })

  it('渲染标题为 h4', () => {
    renderText('## 小节标题')
    expect(screen.getByText('小节标题').tagName).toBe('H4')
  })

  it('空行产生分段（不报错）', () => {
    renderText('第一段\n\n第二段')
    expect(screen.getByText('第一段')).toBeInTheDocument()
    expect(screen.getByText('第二段')).toBeInTheDocument()
  })
})
