// frontend/src/pages/KnowledgeBase.jsx
// Admin-only control panel for the AI knowledge base.
// Embedding and summarization are two independent, explicit actions.
// Live pending/progress state lives in AdminContext so Sidebar can show it too.
//
// Selection model:
//   Nothing selected → the buttons act on ALL documents, exactly as before.
//   One or more selected → the same buttons narrow to the selection and a
//   Delete action appears. Selection is the only thing that changes what
//   the buttons target; there is no separate "scoped mode" to toggle.

import { useState, useEffect, useRef } from 'react'
import client from '../api/client'
import { useAdmin } from '../context/AdminContext'

export default function KnowledgeBase() {
  const [files,      setFiles]      = useState([])
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState('')
  const [notice,     setNotice]     = useState('')
  const [embedState, setEmbedState] = useState(null)
  const [selected,   setSelected]   = useState([])      // filenames
  const [confirming, setConfirming] = useState(null)     // {filenames} | null
  const [deleting,   setDeleting]   = useState(false)
  const pollRef = useRef(null)

  const { pendingCount, batchState, startBatchSummarize, refreshPending } = useAdmin()

  useEffect(() => {
    loadFiles()
    checkStatus()
    return () => stopPolling()
  }, [])

  const loadFiles = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await client.get('/upload/list')
      const list = res.data.files || []
      setFiles(list)
      // Drop selections for documents that no longer exist, so a stale
      // filename can never be sent to embed/summarize/delete.
      const names = new Set(list.map(f => f.filename))
      setSelected(prev => prev.filter(n => names.has(n)))
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load documents.')
    } finally {
      setLoading(false)
    }
  }

  const checkStatus = async () => {
    try {
      const res = await client.get('/embed/status')
      setEmbedState(res.data)
      if (res.data.state?.running) startPolling()
    } catch { /* silent */ }
  }

  const startPolling = () => {
    if (pollRef.current) return
    pollRef.current = setInterval(async () => {
      try {
        const res = await client.get('/embed/status')
        setEmbedState(res.data)
        if (!res.data.state?.running) {
          stopPolling()
          loadFiles()
        }
      } catch { /* silent */ }
    }, 3000)
  }

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  // ── Selection helpers ───────────────────────────────────
  const toggleOne = (filename) => {
    setSelected(prev =>
      prev.includes(filename)
        ? prev.filter(n => n !== filename)
        : [...prev, filename]
    )
  }

  const allSelected = files.length > 0 && selected.length === files.length
  // Some-but-not-all drives the indeterminate checkbox state below.
  const someSelected = selected.length > 0 && !allSelected

  const toggleAll = () => {
    setSelected(allSelected ? [] : files.map(f => f.filename))
  }

  const clearSelection = () => setSelected([])

  // ── Actions ─────────────────────────────────────────────
  const startEmbedding = async () => {
    setError(''); setNotice('')
    try {
      // No selection → no body at all, so this is the same request the
      // page sent before selection existed: embed everything, rebuild.
      const res = selected.length
        ? await client.post('/embed/start', { filenames: selected })
        : await client.post('/embed/start')

      if (!res.data.success) {
        setError(res.data.message || 'Could not start embedding.')
        return
      }
      setEmbedState({ success: true, state: res.data.state, vector_ready: false })
      startPolling()
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not start embedding.')
    }
  }

  const handleSummarize = async () => {
    setError(''); setNotice('')
    const res = await startBatchSummarize(selected.length ? selected : null)
    if (!res.success) setError(res.message || 'Could not start summarization.')
  }

  const confirmDelete = (filenames) => {
    setError(''); setNotice('')
    setConfirming({ filenames })
  }

  const runDelete = async () => {
    if (!confirming) return
    const { filenames } = confirming
    setDeleting(true)
    setError(''); setNotice('')

    try {
      // One request for the whole selection — the backend loads and
      // rewrites the FAISS index once, instead of once per document.
      const res = filenames.length === 1
        ? await client.delete(`/upload/${encodeURIComponent(filenames[0])}`)
        : await client.post('/upload/delete', { filenames })

      const data = res.data

      if (data.success) {
        setNotice(data.message)
      } else {
        // The backend reports leftovers rather than hiding them — surface
        // them verbatim, because a partial delete needs manual attention.
        setError(
          data.message ||
          `Delete did not complete cleanly for ${filenames.join(', ')}.`
        )
      }

      setSelected(prev => prev.filter(n => !filenames.includes(n)))
      setConfirming(null)
      await loadFiles()
      refreshPending?.()
      await checkStatus()
    } catch (err) {
      setError(err.response?.data?.detail || 'Delete failed.')
      setConfirming(null)
    } finally {
      setDeleting(false)
    }
  }

  const notEmbeddedCount = files.filter(f => !f.embedded).length
  const state      = embedState?.state
  const isRunning  = state?.running
  const isBatchRunning = batchState?.running
  const busy = isRunning || isBatchRunning || deleting

  const hasSelection = selected.length > 0

  // With a selection the buttons act on it, so they must not be disabled
  // by the corpus-wide "nothing pending" counts — re-embedding an already
  // embedded document, or re-summarizing one, is a legitimate request.
  //
  // The embed button is no longer gated on notEmbeddedCount at all. It used
  // to disable itself once every document was embedded, which made a full
  // rebuild unreachable exactly when it is most wanted: right after
  // deleting documents, when everything still on the list is embedded and
  // the index needs rebuilding from what remains.
  const embedDisabled     = isRunning
  const summarizeDisabled = isBatchRunning || (!hasSelection && pendingCount === 0)

  // Nothing pending and nothing selected means the click is a deliberate
  // rebuild of an already-complete index, so say that rather than implying
  // there is new work to do.
  const embedLabel = hasSelection
    ? `⚙️ Embed ${selected.length} Selected`
    : (notEmbeddedCount === 0 ? '⚙️ Rebuild Vector Store' : '⚙️ Build Vector Store')

  const btn = (enabled, bg, fg) => ({
    background: enabled ? bg : '#2A2D3E',
    color: enabled ? fg : '#6A6F84',
    border: 'none', borderRadius: '10px',
    padding: '11px 22px', fontSize: '0.88rem', fontWeight: '600',
    cursor: enabled ? 'pointer' : 'not-allowed',
    fontFamily: 'Inter, sans-serif', whiteSpace: 'nowrap',
  })

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      {/* Header */}
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px', fontWeight: '700' }}>
          🗄️ Knowledge Base
        </h1>
        <p style={{ color: '#555', fontSize: '0.85rem', margin: 0 }}>
          Manage which GR documents are indexed and searchable by the AI.
        </p>
      </div>

      {error && (
        <div style={{
          background: '#2A0F0F', border: '1px solid #EF4444', borderRadius: '8px',
          padding: '12px 16px', color: '#EF4444', fontSize: '0.85rem', marginBottom: '20px',
          whiteSpace: 'pre-wrap',
        }}>
          ❌ {error}
        </div>
      )}

      {notice && (
        <div style={{
          background: '#0F2A1A', border: '1px solid #4ADE80', borderRadius: '8px',
          padding: '12px 16px', color: '#4ADE80', fontSize: '0.85rem', marginBottom: '20px',
          whiteSpace: 'pre-wrap',
        }}>
          ✓ {notice}
        </div>
      )}

      {/* Actions panel */}
      <div style={{
        background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '14px',
        padding: '20px 22px', marginBottom: '24px',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px',
        }}>
          <div>
            <div style={{ color: '#E8EAF0', fontSize: '0.95rem', fontWeight: '600', marginBottom: '4px' }}>
              Vector Store &amp; Summaries
            </div>
            <div style={{ color: hasSelection ? '#FF6B00' : '#555', fontSize: '0.8rem' }}>
              {hasSelection ? (
                <>
                  {selected.length} document{selected.length !== 1 ? 's' : ''} selected — actions apply to the selection.
                  <button
                    onClick={clearSelection}
                    style={{
                      background: 'none', border: 'none', color: '#6FB3FF',
                      cursor: 'pointer', fontSize: '0.8rem', padding: '0 0 0 8px',
                      fontFamily: 'Inter, sans-serif', textDecoration: 'underline',
                    }}
                  >clear</button>
                </>
              ) : (
                <>
                  {notEmbeddedCount > 0
                    ? `${notEmbeddedCount} document${notEmbeddedCount !== 1 ? 's' : ''} not yet embedded.`
                    : 'All uploaded documents are embedded.'}
                  {pendingCount > 0 && ` · ${pendingCount} awaiting summary.`}
                </>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <button
              onClick={startEmbedding}
              disabled={embedDisabled}
              style={btn(!embedDisabled, '#FF6B00', 'white')}
            >
              {isRunning ? '⏳ Embedding…' : embedLabel}
            </button>

            <button
              onClick={handleSummarize}
              disabled={summarizeDisabled}
              style={btn(!summarizeDisabled, '#FACC15', '#1A1D2E')}
            >
              {isBatchRunning
                ? '⏳ Summarizing…'
                : hasSelection
                  ? `📋 Summarize ${selected.length} Selected`
                  : '📋 Summarize All'}
            </button>

            {hasSelection && (
              <button
                onClick={() => confirmDelete(selected)}
                disabled={busy}
                style={btn(!busy, '#EF4444', 'white')}
              >
                🗑 Delete {selected.length} Selected
              </button>
            )}
          </div>
        </div>

        {/* Embed progress bar */}
        {state && (state.running || state.last_status !== 'idle') && (
          <div style={{ marginTop: '16px' }}>
            <div style={{ background: '#0F1117', borderRadius: '8px', height: '8px', overflow: 'hidden' }}>
              <div style={{
                width: `${state.progress || 0}%`, height: '100%',
                background: state.last_status === 'failed' ? '#EF4444' : '#FF6B00',
                transition: 'width 0.4s ease',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px' }}>
              <span style={{ color: '#7A7F94', fontSize: '0.78rem' }}>{state.last_message}</span>
              <span style={{ color: '#555', fontSize: '0.78rem', fontFamily: 'monospace' }}>{state.progress || 0}%</span>
            </div>
          </div>
        )}

        {/* Batch summary progress bar */}
        {batchState && (batchState.running || batchState.last_status !== 'idle') && (
          <div style={{ marginTop: '16px' }}>
            <div style={{ background: '#0F1117', borderRadius: '8px', height: '8px', overflow: 'hidden' }}>
              <div style={{
                width: `${batchState.progress || 0}%`, height: '100%',
                background: batchState.last_status === 'failed' ? '#EF4444' : '#FACC15',
                transition: 'width 0.4s ease',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px' }}>
              <span style={{ color: '#7A7F94', fontSize: '0.78rem' }}>{batchState.last_message}</span>
              <span style={{ color: '#555', fontSize: '0.78rem', fontFamily: 'monospace' }}>{batchState.progress || 0}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Document list header + select all */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px',
      }}>
        {files.length > 0 && (
          <input
            type="checkbox"
            checked={allSelected}
            // Indeterminate is not an attribute — it only exists on the DOM
            // node, so it has to be set through a ref callback.
            ref={el => { if (el) el.indeterminate = someSelected }}
            onChange={toggleAll}
            aria-label="Select all documents"
            style={{
              width: '16px', height: '16px', accentColor: '#FF6B00',
              cursor: 'pointer', flexShrink: 0,
            }}
          />
        )}
        <div style={{ color: '#B0B4C8', fontSize: '0.9rem', fontWeight: '600' }}>
          Documents ({files.length})
          {hasSelection && (
            <span style={{ color: '#FF6B00', fontWeight: '500' }}> · {selected.length} selected</span>
          )}
        </div>
      </div>

      {loading && (
        <div style={{ color: '#555', fontSize: '0.85rem', padding: '20px 0' }}>Loading…</div>
      )}

      {!loading && files.length === 0 && (
        <div style={{ color: '#555', fontSize: '0.85rem', padding: '20px 0', fontStyle: 'italic' }}>
          No documents uploaded yet. Go to Upload GR to add some.
        </div>
      )}

      {!loading && files.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {files.map((f) => {
            const isSel = selected.includes(f.filename)
            return (
              // Keyed by filename, not index: after a delete the array
              // shifts, and index keys would make React reuse a row's
              // checkbox state for a different document.
              <div key={f.filename} style={{
                background: isSel ? '#211A2E' : '#1A1D2E',
                border: `1px solid ${isSel ? '#FF6B00' : '#2A2D3E'}`,
                borderRadius: '10px',
                padding: '12px 16px', display: 'flex', alignItems: 'center',
                justifyContent: 'space-between', gap: '12px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
                  <input
                    type="checkbox"
                    checked={isSel}
                    onChange={() => toggleOne(f.filename)}
                    aria-label={`Select ${f.filename}`}
                    style={{
                      width: '16px', height: '16px', accentColor: '#FF6B00',
                      cursor: 'pointer', flexShrink: 0,
                    }}
                  />
                  <span style={{ fontSize: '1.2rem', flexShrink: 0 }}>📄</span>
                  <div style={{ overflow: 'hidden' }}>
                    <div style={{
                      color: '#E8EAF0', fontSize: '0.85rem', fontWeight: '600',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {f.filename}
                    </div>
                    <div style={{ color: '#555', fontSize: '0.72rem', fontFamily: 'monospace', marginTop: '2px' }}>
                      {f.page_count ?? '?'} pages
                      {!f.exists_on_disk && (
                        <span style={{ color: '#EF4444', marginLeft: '8px' }}>⚠ missing from disk</span>
                      )}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>
                  <span style={{
                    fontSize: '0.7rem', padding: '3px 10px', borderRadius: '12px',
                    fontFamily: 'monospace',
                    color: f.embedded ? '#4ADE80' : '#FACC15',
                    background: f.embedded ? '#0F2A1A' : '#2A2410',
                    border: `1px solid ${f.embedded ? '#4ADE8033' : '#FACC1533'}`,
                  }}>
                    {f.embedded ? '✓ Embedded' : '○ Pending'}
                  </span>

                  <button
                    onClick={() => confirmDelete([f.filename])}
                    disabled={busy}
                    title={`Delete ${f.filename} and all of its data`}
                    style={{
                      background: 'none',
                      border: `1px solid ${busy ? '#2A2D3E' : '#EF444455'}`,
                      borderRadius: '8px', padding: '5px 10px',
                      color: busy ? '#3A3D4E' : '#EF4444',
                      cursor: busy ? 'not-allowed' : 'pointer',
                      fontSize: '0.75rem', fontFamily: 'Inter, sans-serif',
                    }}
                  >🗑</button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Delete confirmation ─────────────────────────────
          Deletion removes the PDF, its vectors, its metadata, its summary
          and its graph entries — none of which can be undone from the UI,
          so it is spelled out before the click that does it. */}
      {confirming && (
        <div style={{
          position: 'fixed', inset: 0, background: '#000000AA',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
        }}>
          <div style={{
            background: '#1A1D2E', border: '1px solid #EF4444', borderRadius: '14px',
            padding: '24px 26px', maxWidth: '520px', width: '90%',
          }}>
            <div style={{ color: '#E8EAF0', fontSize: '1.05rem', fontWeight: '700', marginBottom: '10px' }}>
              Delete {confirming.filenames.length} document{confirming.filenames.length !== 1 ? 's' : ''}?
            </div>

            <div style={{
              maxHeight: '160px', overflowY: 'auto', background: '#0F1117',
              border: '1px solid #2A2D3E', borderRadius: '8px',
              padding: '10px 12px', marginBottom: '14px',
            }}>
              {confirming.filenames.map(n => (
                <div key={n} style={{
                  color: '#B0B4C8', fontSize: '0.8rem', fontFamily: 'monospace',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>{n}</div>
              ))}
            </div>

            <div style={{ color: '#7A7F94', fontSize: '0.82rem', lineHeight: '1.6', marginBottom: '18px' }}>
              This permanently removes, for each document:
              <ul style={{ margin: '6px 0 0', paddingLeft: '18px' }}>
                <li>the uploaded PDF file</li>
                <li>its chunks in the vector index</li>
                <li>its MongoDB metadata record</li>
                <li>its generated summary, if any</li>
                <li>its nodes and relationships in the GR graph</li>
              </ul>
              <div style={{ marginTop: '8px', color: '#555' }}>
                The vector index is updated in place — the other documents keep
                their embeddings and are not re-embedded.
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setConfirming(null)}
                disabled={deleting}
                style={{
                  background: '#2A2D3E', color: '#B0B4C8', border: 'none',
                  borderRadius: '10px', padding: '10px 18px', fontSize: '0.85rem',
                  cursor: deleting ? 'not-allowed' : 'pointer', fontFamily: 'Inter, sans-serif',
                }}
              >Cancel</button>
              <button
                onClick={runDelete}
                disabled={deleting}
                style={{
                  background: deleting ? '#5A1F1F' : '#EF4444', color: 'white', border: 'none',
                  borderRadius: '10px', padding: '10px 18px', fontSize: '0.85rem', fontWeight: '600',
                  cursor: deleting ? 'not-allowed' : 'pointer', fontFamily: 'Inter, sans-serif',
                }}
              >{deleting ? 'Deleting…' : 'Delete permanently'}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
