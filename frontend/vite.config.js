// vite.config.js
// Tells Vite to use the Tailwind plugin so CSS classes work
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),   // ← adds Tailwind support
  ],
  server: {
    port: 5173,
    // Proxy API calls to FastAPI backend
    // So frontend can call /api/... instead of http://localhost:8000/api/...
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})