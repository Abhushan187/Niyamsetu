# backend/config.py
# ─────────────────────────────────────────────────────────
# Central configuration for Niyamsetu backend.
# All settings are loaded from the .env file.
# Every other module imports from here — never hardcode values elsewhere.
# ─────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── MongoDB ───────────────────────────────────────────
    # Connection string from Atlas (stored in .env, never hardcoded)
    MONGODB_URL: str
    # Name of the database inside your Atlas cluster
    DATABASE_NAME: str = "niyamsetu"

    # ── JWT Authentication ────────────────────────────────
    # Secret key used to sign tokens — anyone with this can forge tokens
    # So it stays in .env and never touches GitHub
    JWT_SECRET: str
    # How many hours before a login token expires (user gets logged out)
    JWT_EXPIRE_HOURS: int = 24

    # ── Ollama (local LLM) ────────────────────────────────
    # Where Ollama is running — always localhost for on-premise
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Model used to generate embeddings (converts text to vectors)
    EMBEDDING_MODEL: str = "nomic-embed-text"
    # Model used to generate answers (the actual LLM)
    LLM_MODEL: str = "phi4-mini"
    # Hard ceiling on a single LLM call, in seconds.
    # Without this the Ollama client defaults to timeout=None — an unbounded
    # wait. A client that gave up long ago leaves the generation running, and
    # because Ollama serialises on the model, that orphan blocks every later
    # request until it finishes on its own.
    # Set just above the harness's 600s per-question timeout so the caller
    # still governs normal runs and this only fires as a backstop.
    LLM_TIMEOUT: float = 620.0
    # Context window pinned explicitly, in tokens.
    # Left unset, Ollama sizes the context PER REQUEST from the prompt length
    # and restarts llama-server whenever the tier changes (observed: -c 512,
    # -c 2048, -c 4096 for the same model). That restart is a fresh CPU
    # allocation of the weights (~2.3 GiB, mmap is off on the CPU path) plus
    # the KV cache, and on a loaded machine it fails outright:
    #   ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer
    #   graph_reserve: failed to allocate compute buffers      -> HTTP 500
    # Pinning removes the resize, so the model loads once and stays put.
    # 8192 (not 4096) because the chat path never truncates: TOP_K=15 chunks
    # of CHUNK_SIZE=800 plus source headers and CONTEXT_WINDOW=6 history turns
    # reach ~16k characters, which is denser than 4096 tokens in Devanagari.
    # Ollama runs with --context-shift, so a prompt that overflows a pinned
    # context is NOT an error — it silently drops the front of the prompt,
    # i.e. retrieved chunks, while citations still reference them.
    LLM_NUM_CTX: int = 8192
    # How long Ollama keeps the model resident after a request.
    # Ollama's default is 5m, so any normal pause between questions unloads
    # the model and the next question pays a full reload — another chance to
    # hit the allocation failure above. 30m spans idle gaps in a chat session.
    LLM_KEEP_ALIVE: str = "30m"

    # ── RAG settings ──────────────────────────────────────
    # How many characters per chunk when splitting PDFs
    CHUNK_SIZE: int = 800
    # How many characters overlap between chunks (prevents cutting mid-sentence)
    CHUNK_OVERLAP: int = 150
    # How many chunks to retrieve per query
    TOP_K: int = 12
    # How many previous chat turns to include for context
    CONTEXT_WINDOW: int = 6

    # ── Local file storage ────────────────────────────────
    # Base data directory — relative to backend/ folder
    DATA_DIR: str = "./data"

    # ── Computed paths (derived from DATA_DIR) ────────────
    # These are properties, not .env variables
    # They give you Path objects you can use directly in code
    @property
    def GRDOCS_PATH(self) -> Path:
        # Where uploaded GR PDFs are stored
        return Path(self.DATA_DIR) / "grdocs"

    @property
    def VECTORSTORE_PATH(self) -> Path:
        # Where FAISS index files are saved
        return Path(self.DATA_DIR) / "vectorstore"

    @property
    def SUMMARIES_PATH(self) -> Path:
        # Where generated summary files are saved
        return Path(self.DATA_DIR) / "summaries"

    class Config:
        # Tells pydantic to read values from the .env file automatically
        env_file = ".env"
        # If a variable is in .env but not defined above, ignore it
        extra = "ignore"


# ── Single instance used across the entire app ────────────
# Every file does: from config import settings
# Then uses: settings.LLM_MODEL, settings.MONGODB_URL etc.
settings = Settings()

# ── Ensure data directories exist on startup ──────────────
# Creates the folders if they don't exist yet
# exist_ok=True means no error if folder already exists
settings.GRDOCS_PATH.mkdir(parents=True, exist_ok=True)
settings.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
settings.SUMMARIES_PATH.mkdir(parents=True, exist_ok=True)