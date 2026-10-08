// frontend/src/main.tsx
// 应用入口：把 <App/> 挂到 index.html 的 #root 上

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles.css'

const rootEl = document.getElementById('root')
if (!rootEl) throw new Error('找不到 #root 节点')

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
