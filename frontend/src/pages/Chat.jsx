// frontend/src/pages/Chat.jsx
// Session management handled by ChatContext + Sidebar.
// This page only handles sending messages and displaying them.

import { useRef, useEffect, useState } from 'react'
import { useChat } from '../context/ChatContext'
import client from '../api/client'
import CitationPill from '../components/CitationPill'

// Optimistic assistant bubble shown between send and response. Named so the
// send path and the display filter can never disagree on its exact text.
const PLACEHOLDER = '...'

export default function Chat() {
  const {
    messages, history, activeSessionId,
    appendMessages, replaceLastAssistant, updateSessionTitle, createSession,
    setActiveSessionId,
  } = useChat()

  const [input,          setInput]          = useState('')
  const [loading,        setLoading]        = useState(false)
  const [error,          setError]          = useState('')
  const [graphPanel,     setGraphPanel]     = useState(false)
  const [graphRelations, setGraphRelations] = useState([])

  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const fetchGraphRelations = async (citations) => {
    if (!citations?.length) return
    const relations = []
    const seen = new Set()
    for (const c of citations) {
      const grId = c.file.replace('.pdf', '')
      if (seen.has(grId)) continue
      seen.add(grId)
      try {
        const res = await client.get(`/graph/${grId}`)
        if (res.data.relationships?.length)
          relations.push(...res.data.relationships)
      } catch { /* no relations */ }
    }
    if (relations.length) { setGraphRelations(relations); setGraphPanel(true) }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    // Auto-create session if none active
    let sessionId = activeSessionId
    if (!sessionId) {
      sessionId = await createSession()
      if (!sessionId) { setError('Could not create session.'); return }
    }

    setInput('')
    setError('')
    setGraphPanel(false)

    // Optimistic render: show the user's message immediately, with a
    // placeholder assistant turn that replaceLastAssistant() overwrites
    // in place once the response (or an error) arrives.
    const userMsg = { role: 'user', content: query }
    appendMessages(userMsg, { role: 'assistant', content: PLACEHOLDER })

    setLoading(true)

    try {
      const res = await client.post('/query/chat', { query, history })
      const { answer, citations, elapsed_sec, language } = res.data

      const assistantMsg = {
        role: 'assistant', content: answer,
        citations:   citations   || [],
        elapsed_sec: elapsed_sec || 0,
        language:    language    || 'english',
      }

      // Overwrite the placeholder in place. Calling appendMessages again
      // here would push a second copy of userMsg and leave the "..."
      // stranded in both the rendered list and the history sent upstream.
      replaceLastAssistant(assistantMsg)

      // Save to MongoDB session
      try {
        await client.post(`/sessions/${sessionId}/messages`, {
          user_msg:      { role: 'user', content: query },
          assistant_msg: assistantMsg,
        })
        // Update title in sidebar if it was auto-titled
        if (query.length > 0) updateSessionTitle(sessionId, query.slice(0, 50))
      } catch { /* non-critical */ }

      if (citations?.length) fetchGraphRelations(citations)

    } catch (err) {
      const detail = err.response?.data?.detail || 'Something went wrong.'
      setError(detail)
      // Resolve the placeholder on the failure path too — otherwise a "..."
      // bubble stays on screen once loading clears, and the next request
      // ships "..." upstream as if it were a real assistant turn.
      replaceLastAssistant({ role: 'assistant', content: `⚠️ ${detail}` })
    } finally {
      setLoading(false)
    }
  }

  const suggestions = [
    'What is this Government Resolution about?',
    'या शासन निर्णयाचा विषय काय आहे?',
    'Who signed this resolution?',
    'What are the key provisions?',
    'What is the GR number and date?',
    'Does this GR supersede any previous order?',
  ]

  const relationColor = { supersedes: '#EF4444', amends: '#FACC15', refers_to: '#6FB3FF' }

  // Hide the optimistic placeholder while the request is in flight — the
  // dedicated "Searching GR documents..." block below covers that state.
  // This only ever matches the trailing entry, which is exactly where
  // replaceLastAssistant() writes, so the placeholder is always either
  // filtered out or already overwritten.
  const displayMessages = messages.filter(
    (m, i) => !(m.role === 'assistant' && m.content === PLACEHOLDER && loading && i === messages.length - 1)
  )

  return (
    <div style={{
      display: 'flex', height: '100%',
      fontFamily: 'Inter, sans-serif', position: 'relative', overflow: 'hidden',
    }}>

      {/* ── MAIN CHAT ──────────────────────────────── */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
        marginRight: graphPanel ? '320px' : '0',
        transition: 'margin-right 0.25s ease',
      }}>

        {/* Messages */}
        <div style={{
          flex: 1, overflowY: 'auto',
          padding: '24px 32px',
          display: 'flex', flexDirection: 'column', gap: '20px',
        }}>

          {/* Empty state */}
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', padding: '48px 24px' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🏛️</div>
              <h2 style={{ color: '#FF6B00', fontSize: '1.4rem', margin: '0 0 8px' }}>
                नमस्कार! Hello!
              </h2>
              <p style={{ color: '#555', fontSize: '0.9rem', margin: '0 0 28px' }}>
                Ask anything about uploaded Government Resolutions.<br />
                English and Marathi both supported.
              </p>
              <div style={{
                display: 'flex', flexWrap: 'wrap', gap: '8px',
                justifyContent: 'center', maxWidth: '580px', margin: '0 auto',
              }}>
                {suggestions.map((s, i) => (
                  <button key={i} onClick={() => setInput(s)} style={{
                    background: '#1A1D2E', border: '1px solid #2A2D3E',
                    borderRadius: '20px', color: '#B0B4C8',
                    padding: '7px 14px', fontSize: '0.82rem',
                    cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                  }}
                    onMouseEnter={e => e.target.style.borderColor = '#FF6B00'}
                    onMouseLeave={e => e.target.style.borderColor = '#2A2D3E'}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message list */}
          {displayMessages.map((msg, i) => (
            <div key={i}>
              {msg.role === 'user' ? (
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <div style={{
                    background: '#FF6B00', color: 'white',
                    padding: '12px 18px', borderRadius: '18px 18px 4px 18px',
                    maxWidth: '70%', fontSize: '0.95rem', lineHeight: '1.5',
                  }}>
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                  <div style={{
                    width: '36px', height: '36px', background: '#1E3A5F',
                    borderRadius: '50%', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', fontSize: '1rem', flexShrink: 0, marginTop: '2px',
                  }}>🏛️</div>
                  <div style={{ maxWidth: '75%' }}>
                    <div style={{
                      background: '#1A1D2E', border: '1px solid #2A2D3E',
                      borderRadius: '4px 18px 18px 18px',
                      padding: '14px 18px', color: '#E8EAF0',
                      fontSize: '0.95rem', lineHeight: '1.65', whiteSpace: 'pre-wrap',
                    }}>
                      {msg.content}
                    </div>
                    {msg.citations?.length > 0 && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '8px' }}>
                        {msg.citations.map((c, ci) => (
                          <CitationPill key={ci} file={c.file} page={c.page} />
                        ))}
                      </div>
                    )}
                    <div style={{ color: '#444', fontSize: '0.72rem', marginTop: '6px', fontFamily: 'monospace' }}>
                      ⏱ {msg.elapsed_sec}s
                      {msg.language && (
                        <span style={{ marginLeft: '8px' }}>
                          · {msg.language === 'marathi' ? '🇮🇳 Marathi' : '🇬🇧 English'}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Loading */}
          {loading && (
            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
              <div style={{
                width: '36px', height: '36px', background: '#1E3A5F',
                borderRadius: '50%', display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: '1rem',
              }}>🏛️</div>
              <div style={{
                background: '#1A1D2E', border: '1px solid #2A2D3E',
                borderRadius: '4px 18px 18px 18px',
                padding: '14px 18px', color: '#555', fontSize: '0.88rem',
              }}>
                Searching GR documents...
              </div>
            </div>
          )}

          {error && (
            <div style={{
              background: '#2A0F0F', border: '1px solid #EF4444',
              borderRadius: '8px', padding: '12px 16px',
              color: '#EF4444', fontSize: '0.85rem',
            }}>
              ❌ {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{
          borderTop: '1px solid #2A2D3E', padding: '14px 24px', background: '#0F1117',
        }}>
          <form onSubmit={sendMessage} style={{ display: 'flex', gap: '10px' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type in English or Marathi… / मराठीत प्रश्न विचारा…"
              disabled={loading}
              style={{
                flex: 1, background: '#1A1D2E', border: '1px solid #2A2D3E',
                borderRadius: '12px', padding: '12px 18px', color: '#E8EAF0',
                fontSize: '0.95rem', outline: 'none', fontFamily: 'Inter, sans-serif',
              }}
              onFocus={e => e.target.style.borderColor = '#FF6B00'}
              onBlur={e  => e.target.style.borderColor = '#2A2D3E'}
            />
            <button
              type="submit" disabled={loading || !input.trim()}
              style={{
                background: loading || !input.trim() ? '#3A1800' : '#FF6B00',
                color: 'white', border: 'none', borderRadius: '12px',
                padding: '12px 20px', fontSize: '1rem',
                cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
                fontWeight: '600', fontFamily: 'Inter, sans-serif',
              }}
            >➤</button>
          </form>
        </div>
      </div>

      {/* ── GR GRAPH PANEL ─────────────────────────── */}
      {graphPanel && (
        <div style={{
          position: 'fixed', right: 0, top: '56px',
          width: '320px', height: 'calc(100vh - 56px)',
          background: '#1A1D2E', borderLeft: '1px solid #2A2D3E',
          zIndex: 50, display: 'flex', flexDirection: 'column',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '16px 18px', borderBottom: '1px solid #2A2D3E',
          }}>
            <div>
              <div style={{ color: '#E8EAF0', fontWeight: '600', fontSize: '0.95rem' }}>
                🕸️ GR Relationships
              </div>
              <div style={{ color: '#555', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                {graphRelations.length} connection{graphRelations.length !== 1 ? 's' : ''} found
              </div>
            </div>
            <button onClick={() => setGraphPanel(false)} style={{
              background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '1.1rem',
            }}>✕</button>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
            {graphRelations.map((rel, i) => (
              <div key={i} style={{
                background: '#13151F', border: '1px solid #2A2D3E',
                borderRadius: '8px', padding: '12px 14px', marginBottom: '8px',
              }}>
                <div style={{
                  display: 'inline-block',
                  color: relationColor[rel.relation] || '#888',
                  border: `1px solid ${relationColor[rel.relation] || '#888'}`,
                  borderRadius: '12px', padding: '2px 10px',
                  fontSize: '0.72rem', fontFamily: 'monospace',
                  fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase',
                }}>{rel.relation}</div>
                <div style={{ fontSize: '0.82rem', color: '#E8EAF0', marginBottom: '6px' }}>
                  <span style={{ color: '#FF6B00', fontFamily: 'monospace' }}>{rel.source}</span>
                  <span style={{ color: '#555', margin: '0 6px' }}>→</span>
                  <span style={{ color: rel.target === 'Unknown' ? '#555' : '#6FB3FF', fontFamily: 'monospace' }}>
                    {rel.target}
                  </span>
                </div>
                {rel.snippet && (
                  <div style={{
                    color: '#555', fontSize: '0.75rem', lineHeight: '1.4', fontStyle: 'italic',
                    borderTop: '1px solid #2A2D3E', paddingTop: '6px', marginTop: '4px',
                  }}>"{rel.snippet.slice(0, 120)}…"</div>
                )}
              </div>
            ))}
          </div>
          <div style={{ padding: '12px 16px', borderTop: '1px solid #2A2D3E', display: 'flex', gap: '12px' }}>
            {Object.entries(relationColor).map(([type, color]) => (
              <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
                <span style={{ color: '#555', fontSize: '0.7rem', fontFamily: 'monospace' }}>{type}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}