// frontend/vitest.config.ts
// Vitest 配置：复用 vite.config.ts 的插件，额外加 jsdom 环境与 jest-dom matchers。
// 单独一个文件，避免与 Vite 的 build/dev 行为互相干扰。

import { mergeConfig } from 'vite'
import { defineConfig } from 'vitest/config'
import viteConfig from './vite.config'

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      exclude: ['node_modules', 'dist'],
    },
  }),
)
