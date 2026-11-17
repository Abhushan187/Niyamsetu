// frontend/src/App.jsx
// ─────────────────────────────────────────────────────────
// Root component — sets up routing and auth context.
//
// Route structure — every route actually registered below:
//   /                 → redirects to /login
//   /login            → Login page (public)
//   /chat             → Chat page (requires login)
//   /graph            → GR relationship graph (requires login)
//   /registry         → Document registry table (requires login)
//   /search           → Retrieval inspector (requires login)
//   /summaries        → Generated GR summaries (requires login)
//   /admin/upload     → Upload GRs (admin only)
//   /admin/knowledge  → Knowledge base: embed + summarize (admin only)
//   /admin/analytics  → Query stats and logs (admin only)
//   /admin/users      → User management (admin only)
//   *                 → anything else redirects to /login
// ─────────────────────────────────────────────────────────

import { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { backendUp } from './apiBase'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import AdminRoute from './components/AdminRoute'
import Layout from './components/Layout'

// Pages
import Login   from './pages/Login'
import Chat    from './pages/Chat'
import Graph   from './pages/Graph'
import Registry from './pages/Registry'
import Search   from './pages/Search'
import Upload   from './pages/Upload'
import Users    from './pages/Users'
import Summaries from './pages/Summaries'
import Analytics from './pages/Analytics'
import KnowledgeBase from './pages/KnowledgeBase'

export default function App() {
  // Backend reachability, checked on mount and every 20 seconds.
  // Starts true so the banner does not flash before the first check.
  const [online, setOnline] = useState(true)

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      const up = await backendUp()
      if (!cancelled) setOnline(up)
    }
    check()
    const id = setInterval(check, 20000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return (
    // BrowserRouter enables URL-based navigation
    <BrowserRouter>
      {!online && (
        <div className="fixed top-0 inset-x-0 z-50 bg-red-600 text-white text-center text-sm py-2 px-4 pointer-events-none">
          The server is offline. Please try again later.
        </div>
      )}
      {/* AuthProvider wraps everything so all pages can useAuth() */}
      <AuthProvider>
        <Routes>

          {/* ── Public routes ─────────────────────────── */}
          {/* Root → redirect to login */}
          <Route path="/" element={<Navigate to="/login" replace />} />

          {/* Login page — only page accessible without token */}
          <Route path="/login" element={<Login />} />

          {/* ── User routes (login required) ──────────── */}
          <Route path="/chat" element={
            <ProtectedRoute><Layout><Chat /></Layout></ProtectedRoute>
          } />

          <Route path="/graph" element={
            <ProtectedRoute><Layout><Graph /></Layout></ProtectedRoute>
          } />

          <Route path="/registry" element={
            <ProtectedRoute><Layout><Registry /></Layout></ProtectedRoute>
          } />

          <Route path="/search" element={
            <ProtectedRoute><Layout><Search /></Layout></ProtectedRoute>
          } />

          <Route path="/summaries" element={
            <ProtectedRoute><Layout><Summaries /></Layout></ProtectedRoute>
          } />

          {/* ── Admin routes ───────────────────────────── */}
          <Route path="/admin/upload" element={
            <AdminRoute><Layout><Upload /></Layout></AdminRoute>
          } />

          <Route path="/admin/knowledge" element={
            <AdminRoute><Layout><KnowledgeBase /></Layout></AdminRoute>
          } />

          <Route path="/admin/analytics" element={
            <AdminRoute><Layout><Analytics /></Layout></AdminRoute>
          } />

          <Route path="/admin/users" element={
            <AdminRoute><Layout><Users /></Layout></AdminRoute>
          } />

          {/* Catch-all — unknown URLs redirect to login */}
          <Route path="*" element={<Navigate to="/login" replace />} />

        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}