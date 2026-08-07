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

  const appendMessages = (userMsg, assistantMsg) => {
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setHistory(prev => [
      ...prev,
      { role: 'user',      content: userMsg.content      },
      { role: 'assistant', content: assistantMsg.content },
    ])
  }

  // Overwrites the trailing assistant entry instead of pushing a new one.
  // Chat.jsx appends an optimistic "..." placeholder alongside the user
  // message, then calls this once the real answer lands. Calling
  // appendMessages a second time instead would push a duplicate user turn
  // and strand the placeholder in both messages and history.
  const replaceLastAssistant = (assistantMsg) => {
    setMessages(prev => {
      const last = prev.length - 1
      if (last < 0 || prev[last].role !== 'assistant') return prev
      const next = [...prev]
      next[last] = assistantMsg
      return next
    })

    // history is the trimmed {role, content} view sent to the backend —
    // keep it in step so the next request carries the real answer rather
    // than the placeholder.
    setHistory(prev => {
      const last = prev.length - 1
      if (last < 0 || prev[last].role !== 'assistant') return prev
      const next = [...prev]
      next[last] = { role: 'assistant', content: assistantMsg.content }
      return next
    })
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
      appendMessages, replaceLastAssistant, updateSessionTitle,
      setActiveSessionId,
    }}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  return useContext(ChatContext)
}