// frontend/src/context/AdminContext.jsx
// Shared admin state — pending summary count + batch job progress.
// Sidebar reads this to show a live badge from ANY page.
// KnowledgeBase reads/writes this to trigger and display the batch job.

import { createContext, useContext, useState, useEffect, useRef } from 'react'
import client from '../api/client'
import { useAuth } from './AuthContext'

const AdminContext = createContext(null)

export function AdminProvider({ children }) {
  const { isAdmin } = useAuth()

  const [pendingCount, setPendingCount] = useState(0)
  const [batchState,   setBatchState]   = useState(null)

  const pendingPollRef = useRef(null)
  const batchPollRef   = useRef(null)
  const batchRunningRef = useRef(false) // avoids stale-closure issue inside setInterval

  useEffect(() => {
    batchRunningRef.current = !!batchState?.running
  }, [batchState])

  useEffect(() => {
    if (!isAdmin) {
      setPendingCount(0)
      setBatchState(null)
      return
    }
    loadPending()
    checkBatchStatus()
    startPendingPolling()
    return () => { stopPendingPolling(); stopBatchPolling() }
  }, [isAdmin])

  const loadPending = async () => {
    try {
      const res = await client.get('/summary/pending')
      setPendingCount(res.data.total ?? (res.data.pending || []).length)
    } catch { /* silent — badge just won't show */ }
  }

  // 60s, not 15s. This timer runs for the whole session on every admin
  // page, so it is the only poller that hits the backend continuously —
  // the embed and batch pollers only run while their job is active. At 15s
  // it produced four access-log lines a minute forever, drowning the
  // server terminal. The value it fetches is a badge count that changes
  // only when a summary finishes, and the batch poller already refreshes
  // that count on completion, so a slower tick loses nothing.
  const PENDING_POLL_MS = 60000

  const startPendingPolling = () => {
    if (pendingPollRef.current) return
    pendingPollRef.current = setInterval(() => {
      // Skip while a batch job is running — batch polling already keeps count fresh then
      if (!batchRunningRef.current) loadPending()
    }, PENDING_POLL_MS)
  }

  const stopPendingPolling = () => {
    if (pendingPollRef.current) { clearInterval(pendingPollRef.current); pendingPollRef.current = null }
  }

  const checkBatchStatus = async () => {
    try {
      const res = await client.get('/summary/batch-status')
      setBatchState(res.data.state)
      if (res.data.state?.running) startBatchPolling()
    } catch { /* silent */ }
  }

  const startBatchPolling = () => {
    if (batchPollRef.current) return
    batchPollRef.current = setInterval(async () => {
      try {
        const res = await client.get('/summary/batch-status')
        setBatchState(res.data.state)
        if (!res.data.state?.running) {
          stopBatchPolling()
          loadPending() // refresh true count once job finishes
        }
      } catch { /* silent */ }
    }, 3000)
  }

  const stopBatchPolling = () => {
    if (batchPollRef.current) { clearInterval(batchPollRef.current); batchPollRef.current = null }
  }

  /**
   * Starts batch summarization.
   *
   * @param {string[]|null} filenames  null/omitted → every pending document
   *                                   (unchanged default). A list → only
   *                                   those documents.
   *
   * The body is omitted entirely in the default case rather than sent as
   * {filenames: null}, so the request stays byte-identical to what this
   * function sent before selection existed.
   */
  const startBatchSummarize = async (filenames = null) => {
    try {
      const res = filenames?.length
        ? await client.post('/summary/generate-batch', { filenames })
        : await client.post('/summary/generate-batch')
      if (res.data.success) {
        setBatchState(res.data.state)
        startBatchPolling()
      }
      return res.data
    } catch (err) {
      return { success: false, message: err.response?.data?.detail || 'Could not start summarization.' }
    }
  }

  return (
    <AdminContext.Provider value={{
      pendingCount, batchState, startBatchSummarize, refreshPending: loadPending,
    }}>
      {children}
    </AdminContext.Provider>
  )
}

export function useAdmin() {
  return useContext(AdminContext)
}