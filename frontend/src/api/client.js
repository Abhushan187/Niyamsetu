// frontend/src/api/client.js
// ─────────────────────────────────────────────────────────
// Axios instance — all API calls go through here.
//
// Why one central client?
//   If the backend URL changes, you change it here once.
//   Auth token is automatically attached to every request.
//   401 errors (token expired) are handled here globally.
// ─────────────────────────────────────────────────────────

import axios from 'axios'
import { apiUrl } from '../apiBase'

// Create axios instance pointing to our FastAPI backend
// The proxy in vite.config.js forwards /api calls to localhost:8000
const client = axios.create({
  baseURL: '/api',          // all requests go to /api/...
  headers: {
    'Content-Type': 'application/json',
  },
})

// ── Request interceptor ───────────────────────────────────
// Runs before every request is sent.
// Reads the JWT token from localStorage and adds it to the header.
// This is why every protected route works automatically —
// you never manually add the token in individual API calls.
client.interceptors.request.use(
  (config) => {
    // Backend address is resolved at runtime (see apiBase.js). With an
    // empty base this stays '/api', which the Vite dev proxy handles.
    config.baseURL = apiUrl('') + '/api'
    const token = localStorage.getItem('niyamsetu_token')
    if (token) {
      // Bearer token format required by FastAPI OAuth2
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// ── Response interceptor ──────────────────────────────────
// Runs after every response is received.
// If the server returns 401 (token expired or invalid),
// clear stored credentials and redirect to login page.
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired — clear storage and force re-login
      localStorage.removeItem('niyamsetu_token')
      localStorage.removeItem('niyamsetu_user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default client