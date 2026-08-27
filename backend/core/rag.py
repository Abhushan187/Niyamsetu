# backend/core/rag.py
# ─────────────────────────────────────────────────────────
# RAG (Retrieval Augmented Generation) query engine.
#
# Flow for every chat message:
#   1. Detect language (English or Marathi)
#   2. Search FAISS for relevant chunks
#   3. Extract citations from chunk metadata
#   4. Build prompt with context + history + query
#   5. Send to LLM via Ollama
#   6. Return answer + citations + timing
#
# Called by:
#   api/query.py → passes user query and chat history here
# ─────────────────────────────────────────────────────────

import sys
import os
import time
import asyncio

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from core.hybrid_search import hybrid_search
from core.language import detect_language, format_chat_history, clean_text
from core.trace import trace


# Do NOT pass think=False here to speed qwen3 up — it does not work.
# Measured on qwen3:4b, same prompt, both with a warm prompt cache:
#
#   think ON  : 36.1s, 181 generated tokens, answer "Divisional Commissioner"
#   think OFF : 40.1s, 181 generated tokens, answer = the full reasoning
#               monologue, ending in a stray "</think>" then the answer
#
# The model reasons either way. think=False only moves the chain of thought
# out of the response's `thinking` field and into `content`, so the token
# count — and therefore the time, at ~5 tok/s on CPU — is unchanged, while
# the answer text becomes unusable for the capture harness.
#
# Leaving thinking ON is both faster and cleaner here.


def get_llm() -> ChatOllama:
    """
    Creates a ChatOllama instance pointing to local Ollama server.
    temperature=0 means deterministic responses — same question
    gets consistent answers, important for government documents.
    """
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
        # Pinned so Ollama never resizes the context between requests — see
        # LLM_NUM_CTX in config.py for why a resize is what actually OOMs.
        # langchain-ollama folds this into the Ollama "options" payload.
        num_ctx=settings.LLM_NUM_CTX,
        # Keeps the model resident across idle gaps, so a question after a
        # pause reuses the loaded context instead of forcing a reload.
        keep_alive=settings.LLM_KEEP_ALIVE,
        # client_kwargs goes straight to ollama.Client, which forwards it to
        # httpx.Client. Ollama's own default is timeout=None, so without this
        # a generation can run indefinitely.
        # Closing the socket is what actually stops the work: Ollama ends the
        # request and frees the model, so the next queued request can start.
        # Only bounds total generation together with stream=False at the
        # invoke() call below — see the note there before changing either.
        client_kwargs={"timeout": settings.LLM_TIMEOUT},
    )


def build_citations(docs: list) -> list:
    """
    Extracts citation information from retrieved document chunks.
    Deduplicates — same file+page combination appears only once.

    Args:
        docs : list of LangChain Document objects from retrieval

    Returns:
        List of citation dicts:
        [
            {
                "file": "GR_2024_transfer.pdf",
                "page": "4",           ← displayed as human-readable page number
                "preview": "First 200 chars of the chunk..."
            },
            ...
        ]
    """
    citations = []
    seen      = set()   # tracks file+page combos we've already added

    for doc in docs:
        filename = doc.metadata.get("source_file", "Unknown Document")

        # LangChain pages are 0-indexed internally
        # We add 1 so users see "Page 4" instead of "Page 3"
        raw_page = doc.metadata.get("page", None)
        if raw_page is not None:
            page_display = str(int(raw_page) + 1)
        else:
            page_display = "?"

        # Create a unique key for deduplication
        key = f"{filename}_p{page_display}"

        if key not in seen:
            seen.add(key)
            citations.append({
                "file":    filename,
                "page":    page_display,
                # Preview helps user understand why this source was cited
                "preview": doc.page_content[:200].strip(),
            })

    return citations


def build_prompt(
    query: str,
    context: str,
    history_text: str,
    language: str,
) -> str:
    """
    Builds the full prompt sent to the LLM.
    Two versions — English and Marathi — with different instructions.

    The prompt has 4 parts:
        1. Role instruction  → tells LLM what it is
        2. Rules             → how to behave
        3. Conversation history → previous turns for context
        4. Retrieved context → the actual GR text chunks
        5. Current question  → what to answer right now

    Args:
        query        : current user question
        context      : concatenated FAISS chunks (the GR text)
        history_text : formatted previous conversation turns
        language     : 'english' or 'marathi'

    Returns:
        Complete prompt string ready to send to LLM
    """
    if language == "marathi":
        return f"""तुम्ही महाराष्ट्र शासन निर्णय (Government Resolution) तज्ञ आहात.

नियम:
- फक्त खालील संदर्भातून उत्तर द्या
- पूर्ण उत्तर अनेकदा वेगवेगळ्या स्रोतांत विभागलेले असते — एका स्रोतात एकूण संख्या किंवा नियम, तर दुसऱ्यात तपशील किंवा अपवाद. पहिला संबंधित स्रोत सापडल्यावर थांबू नका; उर्वरित स्रोतांतही संबंधित आकडे, नावे, तारखा किंवा अटी आहेत का ते तपासा
- उत्तरात कोणताही विशिष्ट आकडा देण्यापूर्वी, तो आकडा सांगणारे नेमके वाक्य संदर्भातून जसेच्या तसे उद्धृत करा, आणि मगच अंतिम उत्तर द्या
- या विशिष्ट प्रश्नाचे उत्तर म्हणून संदर्भात स्पष्टपणे न आलेला कोणताही आकडा सांगू नका. संदर्भात इतर आकडे असतात — बैठकीचा क्रमांक, कलम क्रमांक, वर्ष, दिनांक — ते उत्तर नाहीत. आकडा कशाची संख्या आहे ते तपासा
- माहिती नसल्यास सांगा: "हे माहिती दिलेल्या दस्तऐवजात आढळली नाही."
- उत्तर मराठीत द्या
- स्पष्ट उत्तर द्या; संक्षिप्त ठेवा, पण इतर स्रोतातील संबंधित तपशील वगळू नका

मागील संवाद:
{history_text if history_text else "नाही"}

शासन निर्णयातील संदर्भ:
{context}

प्रश्न: {query}

उत्तर:"""

    else:
        return f"""You are an expert assistant for Maharashtra Government Resolution (GR) documents.

Rules:
- Answer ONLY from the provided context below
- A complete answer is often split across sources — one may give a total or a rule, another the breakdown, exception, or detail. Do not stop at the first source that looks relevant; check the remaining sources for numbers, names, dates, or conditions that add to or qualify it
- If your answer includes a specific number, first identify and quote the exact phrase from the context that states that number, then give your final answer
- Do not state a number that does not appear explicitly in the context as the answer to this specific question. The context contains other numbers — meeting numbers, section and clause numbers, years, dates, GR reference numbers — that are NOT the answer. Check what each number counts before using it
- If the answer is not in the context, say exactly: "This information was not found in the uploaded documents."
- Be precise; stay brief, but never omit a relevant detail found in another source
- Use bullet points when listing multiple items
- Always refer to specific GR details when available

Previous conversation:
{history_text if history_text else "None"}

Context from GR documents:
{context}

Current question: {query}

Answer:"""


async def query(
    user_query: str,
    chat_history: list,
    top_k: int = None,
    language: str = None,
) -> dict:
    """
    Main RAG pipeline — called for every chat message.

    Args:
        user_query   : the question the user typed
        chat_history : list of previous {"role", "content"} turns
        top_k        : chunks to retrieve (defaults to settings.TOP_K)
        language     : force a language, or None for auto-detect

    Returns dict:
        success      : bool
        answer       : the LLM's response text
        citations    : list of {file, page, preview}
        language     : detected or forced language
        elapsed_sec  : how long the full pipeline took
        query        : the original question (for logging)
    """
    start_time = time.time()

    if top_k is None:
        top_k = settings.TOP_K

    # ── Step 1: Detect language ───────────────────────────
    if language is None:
        language = detect_language(user_query)

    # ── Step 2: Retrieve (hybrid: dense + BM25, fused by RRF) ──
    # hybrid_search() runs FAISS and BM25 independently over the same
    # corpus and fuses the two rankings. It returns list[Document], the
    # identical shape vectorstore.search() returned, so Steps 3-4 below
    # are untouched — only the ORDER and membership of the chunks change.
    #
    # Still synchronous, and for the same reasons as before: the first
    # call loads the FAISS index from disk and builds the BM25 index, and
    # every call embeds the query via a blocking Ollama HTTP request
    # (~8s cold, ~0.3s warm). The added BM25 half needs no embedding call
    # and is CPU-only. All of it stalls the event loop, so it stays on a
    # worker thread exactly as search() did.
    #
    # Degrades rather than fails: if the dense side errors, hybrid_search
    # returns the lexical ranking alone instead of raising, so a
    # momentarily unreachable Ollama downgrades retrieval quality rather
    # than breaking the chat. Empty list still means "store not ready",
    # which the guard below handles unchanged.
    trace("RETRIEVAL start", f"top_k={top_k} language={language} mode=hybrid")
    docs = await asyncio.to_thread(hybrid_search, user_query, top_k=top_k)
    retrieval_sec = round(time.time() - start_time, 2)
    trace("RETRIEVAL done ", f"{len(docs)} chunk(s) in {retrieval_sec}s")

    # If no docs returned, vector store is not ready
    if not docs:
        elapsed = round(time.time() - start_time, 2)
        no_docs_message = (
            "⚠️ माहिती संच अद्याप तयार नाही. कृपया प्रशासकाला शासन निर्णय दस्तऐवज अपलोड करण्यास सांगा."
            if language == "marathi"
            else "⚠️ Vector store is not ready. Please ask admin to embed the GR documents first."
        )
        return {
            "success":     False,
            "answer":      no_docs_message,
            "citations":   [],
            "language":    language,
            "elapsed_sec": elapsed,
            "query":       user_query,
        }

    # ── Step 3: Build context string ─────────────────────
    # Join all retrieved chunks into one context block
    # Each chunk separated by a divider so LLM can distinguish them
    context_parts = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("source_file", "Unknown")
        page  = doc.metadata.get("page", "?")
        # Label each chunk with its source for better LLM grounding
        context_parts.append(
            f"[Source {i}: {fname}, Page {page}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 4: Extract citations ─────────────────────────
    citations = build_citations(docs)

    # ── Step 5: Format history ────────────────────────────
    history_text = format_chat_history(
        chat_history,
        context_window=settings.CONTEXT_WINDOW,
    )

    # ── Step 6: Build prompt ──────────────────────────────
    prompt = build_prompt(
        query=user_query,
        context=context,
        history_text=history_text,
        language=language,
    )

    # ── Step 7: Call LLM ──────────────────────────────────
    try:
        llm      = get_llm()
        # Marks the retrieval → generation boundary. If the terminal sits on
        # RETRIEVAL done the stall is in FAISS/embedding; if it sits here it
        # is Ollama generating (or queued behind another generation).
        trace(
            "LLM      start",
            f"model={settings.LLM_MODEL} prompt={len(prompt)} chars "
            f"timeout={settings.LLM_TIMEOUT:.0f}s",
        )
        # llm.invoke() is the SYNCHRONOUS entrypoint — ChatOllama drives
        # ollama.Client (blocking httpx), pinning the event loop for the full
        # generation. This runs directly on the request path, so a single
        # chat message froze the whole server. Offloaded to a worker thread;
        # exceptions still propagate to the except block below unchanged.
        # stream=False is what makes settings.LLM_TIMEOUT an actual ceiling.
        # ChatOllama streams even for .invoke() (chat_models.py: "stream":
        # kwargs.pop("stream", True)), and httpx's timeout is per-read — it
        # bounds the GAP BETWEEN TOKENS, not total time. A slow-but-steady
        # generation therefore never trips it and runs unbounded. Non-streaming
        # keeps the connection silent until the answer is complete, so the read
        # timeout covers the whole call. Nothing is streamed to the caller
        # today — _generate() aggregates the stream anyway — so this changes no
        # observable behaviour.
        response = await asyncio.to_thread(
            llm.invoke, [HumanMessage(content=prompt)], stream=False
        )
        answer   = response.content.strip()
        trace("LLM      done ", f"{round(time.time() - start_time, 2)}s total, {len(answer)} chars")

    # Separated from the generic handler below because the cause is the
    # opposite one: Ollama is up and working, just not fast enough. Reporting
    # this as "is Ollama running?" sends you looking in the wrong place.
    except httpx.TimeoutException:
        elapsed = round(time.time() - start_time, 2)
        return {
            "success":     False,
            "answer":      (f"⚠️ LLM timed out after {settings.LLM_TIMEOUT:.0f}s "
                            f"({settings.LLM_MODEL}). The request was aborted so it "
                            f"cannot block the next one."),
            "citations":   citations,
            "language":    language,
            "elapsed_sec": elapsed,
            "query":       user_query,
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        return {
            "success":     False,
            "answer":      f"⚠️ LLM error: {str(e)}. Is Ollama running with {settings.LLM_MODEL}?",
            "citations":   citations,
            "language":    language,
            "elapsed_sec": elapsed,
            "query":       user_query,
        }

    elapsed = round(time.time() - start_time, 2)

    return {
        "success":     True,
        "answer":      answer,
        "citations":   citations,
        "language":    language,
        "elapsed_sec": elapsed,
        "query":       user_query,
    }