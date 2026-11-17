// frontend/src/main.jsx
// ─────────────────────────────────────────────────────────
// React entry point.
// Mounts the App component into the #root div in index.html.
// StrictMode helps catch bugs during development.
// ─────────────────────────────────────────────────────────

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { loadApiBase } from './apiBase'

loadApiBase().finally(() => {
  createRoot(document.getElementById('root')).render(
    <StrictMode>
      <App />
    </StrictMode>
  )
})