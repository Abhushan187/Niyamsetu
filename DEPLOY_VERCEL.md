# Deploying the frontend on Vercel (backend on this machine via Cloudflare tunnel)

## 1. Vercel project settings
1. Import the repository in Vercel.
2. Root directory: `frontend`
3. Framework preset: Vite
4. Build command: `npm run build`
5. Output directory: `dist`

## 2. Allow the Vercel domain in the backend
1. In `backend/.env` set `CORS_ORIGINS` to your real Vercel domain
   (do not copy the commented placeholder from `.env.example` unchanged):
   `CORS_ORIGINS=https://YOUR-APP.vercel.app`
2. Restart the backend.

## 3. Run the backend on localhost only
From the `backend` folder:

    uvicorn main:app --host 127.0.0.1 --port 8000

## 4. Start the tunnel

    cloudflared tunnel --url http://localhost:8000

Copy the `https://SOMETHING.trycloudflare.com` address it prints.

The login limit is per visitor because the backend reads CF-Connecting-IP only from the local tunnel.

## 5. Reviewer link
Send the reviewer:

    https://YOUR-APP.vercel.app/?api=https://SOMETHING.trycloudflare.com

The browser remembers the address, so later visits work without `?api=`
until the tunnel address changes.

## 6. After the viva
Stop the tunnel (Ctrl+C in the cloudflared window). The site will then show
"The server is offline" banner.

## 7. Never expose MongoDB or Ollama
Only tunnel the backend port (8000). Never run cloudflared (or open a
firewall port) for MongoDB (27017) or Ollama (11434).
