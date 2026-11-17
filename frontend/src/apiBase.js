const ALLOWED = /^(https:\/\/[a-z0-9-]+\.trycloudflare\.com|http:\/\/localhost:\d+)$/;
let base = "";

export async function loadApiBase() {
  const p = new URLSearchParams(window.location.search).get("api");
  if (p && ALLOWED.test(p)) localStorage.setItem("apiBase", p);
  base = localStorage.getItem("apiBase") || "";
  if (!base) {
    try {
      const r = await fetch("/config.json", { cache: "no-store" });
      base = (await r.json()).apiBase || "";
    } catch (e) {}
  }
  if (!base) base = "";
}

export const apiUrl = (path) => base + path;

export async function backendUp() {
  try {
    const r = await fetch(apiUrl("/health"), { cache: "no-store" });
    if (!r.ok) return false;
    const data = await r.json();
    return data?.ok === true;
  } catch (e) {
    return false;
  }
}
