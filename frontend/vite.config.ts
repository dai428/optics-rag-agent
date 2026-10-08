import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时：Vite dev server(5173) 把 /api 代理到后端(8000)，避开跨域。
// 生产时：npm run build 产出 dist/，由 FastAPI 直接托管，前后端同源，无需代理。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
