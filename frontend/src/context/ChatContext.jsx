// frontend/src/context/ChatContext.jsx
// Shared state between Sidebar (session list) and Chat (messages).
// Both components read from here instead of duplicating state.

import { createContext, useContext, useState, useEffect } from 'react'
import client from '../api/client'
import { useAuth } from './AuthContext'

const ChatContext = createContext(null)

export function ChatProvider({ children }) {
  const { isLoggedIn } = useAuth()

  const [sessions,        setSessions]        = useState([])
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [messages,        setMessages]        = useState([])
  const [history,         setHistory]         = useState([])
  const [sessionLoading,  setSessionLoading]  = useState(false)

  // Load sessions when user logs in
  useEffect(() => {
    if (isLoggedIn) loadSessions()
    else { setSessions([]); setActiveSessionId(null); setMessages([]); setHistory([]) }
  }, [isLoggedIn])

  const loadSessions = async () => {
    try {
      const res = await client.get('/sessions/')
      setSessions(res.data.sessions || [])
    } catch { /* silent */ }
  }

  const createSession = async () => {
    try {
      const res = await client.post('/sessions/', { title: 'New Chat' })
      const s   = res.data.session
      setSessions(prev => [s, ...prev])
      setActiveSessionId(s._id)
      setMessages([])
      setHistory([])
      return s._id
    } catch { return null }
  }

  const loadSession = async (sessionId) => {
    if (sessionId === activeSessionId) return
    setSessionLoading(true)
    try {
      const res = await client.get(`/sessions/${sessionId}`)
      const s   = res.data.session
      setActiveSessionId(sessionId)
      setMessages(s.messages || [])
      setHistory((s.messages || []).map(m => ({ role: m.role, content: m.content })))
    } catch { /* silent */ }
    finally { setSessionLoading(false) }
  }

  const deleteSession = async (sessionId) => {
    try {
      await client.delete(`/sessions/${sessionId}`)
      setSessions(prev => prev.filter(s => s._id !== sessionId))
      if (activeSessionId === sessionId) {
        setActiveSessionId(null); setMessages([]); setHistory([])
      }
    } catch { /* silent */ }
  }

  const renameSession = async (sessionId, title) => {
    try {
      await client.patch(`/sessions/${sessionId}`, { title })
      setSessions(prev => prev.map(s => s._id === sessionId ? { ...s, title } : s))
    } catch { /* silent */ }
  }

  const pinSession = async (sessionId) => {
    const target = sessions.find(s => s._id === sessionId)
    if (!target) return
    const newPinned = !target.pinned

    // Optimistic update, sorted pinned-first then by recency
    setSessions(prev => {
      const updated = prev.map(s => s._id === sessionId ? { ...s, pinned: newPinned } : s)
      return [...updated].sort((a, b) => {
        if (!!b.pinned !== !!a.pinned) return (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0)
        return new Date(b.updated_at) - new Date(a.updated_at)
      })
    })

    try {
      await client.patch(`/sessions/${sessionId}/pin`, { pinned: newPinned })
    } catch {
      // Revert on failure
      setSessions(prev => prev.map(s => s._id === sessionId ? { ...s, pinned: target.pinned } : s))
    }
  }

  // ── One exchange = one user bubble + one assistant bubble ──────────
  //
  // This used to be a single append-only `appendMessages`, which Chat.jsx
  // called twice per question: once with a "..." placeholder and again
  // with the real answer. Appending cannot replace, so a finished exchange
  // left FOUR entries — question, "...", question again, answer — and the
  // stray "..." bubble rendered above the real one. The display filter
  // that was supposed to hide it only matched while `loading` was true and
  // only at the very last index, so it stopped matching the moment the
  // answer arrived.
  //
  // Split into start/complete/fail so the placeholder is written once and
  // then overwritten in place. There is no path that appends a second
  // assistant bubble for the same question.

  /** Index of the in-flight placeholder, or -1. */
  const pendingIndex = (list) => {
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].role === 'assistant' && list[i].pending) return i
    }
    return -1
  }

  const replacePending = (list, message) => {
    const i = pendingIndex(list)
    // No placeholder to replace (an interrupted or already-resolved
    // request) — append rather than clobbering an unrelated message.
    if (i === -1) return [...list, message]
    const next = [...list]
    next[i] = message
    return next
  }

  /**
   * Adds the user's question plus the assistant placeholder that renders
   * as the loading state.
   *
   * `history` is deliberately NOT touched here. It is the conversation
   * context sent to the LLM, and a placeholder is not part of the
   * conversation — the old code put "..." into it, so every later question
   * carried a fake assistant turn saying "...".
   */
  const startExchange = (userMsg) => {
    setMessages(prev => [
      ...prev,
      userMsg,
      { role: 'assistant', content: '', pending: true },
    ])
  }

  /**
   * Swaps the placeholder for the real answer and commits the exchange to
   * history — only now, once there is a genuine answer to record.
   */
  const completeExchange = (userMsg, assistantMsg) => {
    setMessages(prev => replacePending(prev, assistantMsg))
    setHistory(prev => [
      ...prev,
      { role: 'user',      content: userMsg.content      },
      { role: 'assistant', content: assistantMsg.content },
    ])
  }

  /**
   * Swaps the placeholder for an error bubble.
   *
   * History stays untouched: a failed request produced no assistant turn,
   * and feeding the error text back as context would have the model
   * answering about the error on the next question. This is also what
   * clears the placeholder on failure — previously a failed request left
   * "..." on screen permanently, which is the other way the stray bubble
   * was seen.
   */
  const failExchange = (errorText) => {
    setMessages(prev => replacePending(prev, {
      role:    'assistant',
      content: errorText,
      error:   true,
    }))
  }

  const updateSessionTitle = (sessionId, title) => {
    setSessions(prev => prev.map(s => s._id === sessionId ? { ...s, title } : s))
  }

  return (
    <ChatContext.Provider value={{
      sessions, activeSessionId, messages, history,
      sessionLoading, setMessages,
      loadSessions, createSession, loadSession,
      deleteSession, renameSession, pinSession,
      startExchange, completeExchange, failExchange,
      updateSessionTitle,
      setActiveSessionId,
    }}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  return useContext(ChatContext)
}