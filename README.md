# JengaVest — Backend

The AI + market-data backend for **JengaVest**, an AI financial-analyst and
portfolio-tracker. It's a **FastAPI** service that exposes:

- an **agentic AI analyst** powered by Claude (tool use, streaming, and a
  retrieval-augmented-generation pipeline over uploaded PDFs), and
- **live market data** — stock quotes, price history, symbol search, and
  market news — sourced from Finnhub and Twelve Data.

The companion Next.js frontend lives in `../frontend` and talks to this service
over HTTP via `NEXT_PUBLIC_API_URL`.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment variables](#environment-variables)
- [Getting API keys](#getting-api-keys)
- [Running locally](#running-locally)
- [API reference](#api-reference)
- [The AI agent](#the-ai-agent)
- [The RAG pipeline](#the-rag-pipeline)
- [Market-data providers](#market-data-providers)
- [Deployment](#deployment)
- [Configuration & tuning](#configuration--tuning)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Overview

At a high level the backend does three things:

1. **Answers financial questions** through an agent loop. A user message is
   routed to a task-specific prompt, sent to Claude with a set of tools, and the
   model decides whether to fetch live stock data, search the web, or search the
   user's uploaded documents before answering. Responses can be **streamed**
   token-by-token.
2. **Serves market data** the frontend renders — a searchable stock catalogue,
   live quotes (equities + crypto), historical price series with multiple
   timeframes, and portfolio-aware news.
3. **Indexes uploaded PDFs** into a hybrid (vector + keyword) search index so the
   agent can answer questions grounded in those documents, with page-level
   citations.

---

## Architecture

```
                       ┌──────────────────────────────────────────┐
   Next.js frontend    │                FastAPI (main.py)          │
   ───────────────►    │  /chat  /chat/stream  /upload  /health    │
                       │  /stocks  /stocks/{t}/quote  /history      │
                       │  /stocks/quotes  /stocks/search  /news     │
                       └───────┬───────────────┬──────────────┬────┘
                               │               │              │
                     ┌─────────▼──────┐ ┌──────▼───────┐ ┌────▼─────────┐
                     │  agent.py      │ │  stocks.py   │ │  rag.py      │
                     │  Claude agent  │ │  market data │ │  hybrid RAG  │
                     │  + tool loop   │ │              │ │  Chroma+BM25 │
                     └───┬────────┬───┘ └──────┬───────┘ └──────┬───────┘
                         │        │            │                │
              ┌──────────▼─┐ ┌────▼──────┐ ┌───▼────────┐ ┌─────▼──────┐
              │ Anthropic  │ │ tools.py  │ │  Finnhub   │ │  ChromaDB  │
              │  (Claude)  │ │ web/quote │ │ TwelveData │ │  + BM25    │
              └────────────┘ └───────────┘ └────────────┘ └────────────┘
```

**Request flow for a chat message:**

1. `POST /chat` (or `/chat/stream`) receives `{ message, conversation_history }`.
2. `workflows.route_query` classifies the message (portfolio / news / document /
   rebalancing / general) and picks a task prompt.
3. `cache.build_cached_prompt` wraps the main system prompt with Anthropic
   **prompt caching**; `cache.build_messages` assembles the message list.
4. `agent.run_agent` / `agent.stream_agent` calls Claude with the tools. While
   the model requests tools, `tools.execute_tool` runs them and the loop
   continues; the final text is returned/streamed.
5. Citations are collected from the model's web-search result blocks and RAG
   tool results, de-duplicated, and returned alongside the answer.

---

## Tech stack

| Concern            | Choice                                             |
| ------------------ | -------------------------------------------------- |
| Web framework      | FastAPI + Uvicorn                                  |
| LLM                | Anthropic Claude — `claude-haiku-4-5` (fast)       |
| Agent features     | Tool use, server-side web search, SSE streaming, prompt caching |
| Vector search      | ChromaDB (persistent, cosine)                      |
| Keyword search     | BM25 (`rank-bm25`)                                 |
| PDF parsing        | `pypdf`                                            |
| Tokenisation       | `tiktoken` (`cl100k_base`)                         |
| Stock quotes/news  | Finnhub REST API                                   |
| Price history/crypto | Twelve Data REST API                             |
| Config             | `python-dotenv`                                    |

---

## Project structure

```
backend/
  main.py          # FastAPI app: all HTTP endpoints, CORS, startup
  agent.py         # Claude agent — run_agent (blocking) + stream_agent (SSE), tool loop, citations
  tools.py         # Tool schemas + implementations (stock data, web search, RAG search, drive stub)
  prompts.py       # System prompt + task prompts (portfolio, news, document, rebalancing)
  workflows.py     # route_query (intent → prompt), ticker extraction, chaining/parallel helpers
  cache.py         # Anthropic prompt-cache wrapper + message builder
  rag.py           # RAG pipeline: PDF ingest, chunking, ChromaDB, BM25, hybrid fusion
  stocks.py        # Market data: catalogue, quotes, batch quotes, history, search, news, crypto
  requirements.txt # Python dependencies
  .env.example     # Required environment variables
  documents/       # Uploaded PDFs (indexed by the RAG pipeline)
  chroma_db/       # Persistent ChromaDB store (auto-created)
  bm25_index.json  # Persisted BM25 index (auto-created)
```

---

## Prerequisites

- **Python 3.10+**
- API keys for **Anthropic**, **Finnhub**, and **Twelve Data** (all have free
  tiers — see below)

---

## Environment variables

Create a `.env` file in the `backend/` folder (copy `.env.example`):

```dotenv
ANTHROPIC_API_KEY=sk-ant-...      # Claude — the AI analyst
FINNHUB_API_KEY=...               # live equity quotes, symbol search, news
TWELVE_DATA_API_KEY=...           # price history + crypto quotes/history
```

| Variable              | Used by                                   | Required for                          |
| --------------------- | ----------------------------------------- | ------------------------------------- |
| `ANTHROPIC_API_KEY`   | `agent.py`                                 | `/chat`, `/chat/stream`               |
| `FINNHUB_API_KEY`     | `stocks.py`, `tools.py`                    | quotes, `/stocks/search`, `/news`     |
| `TWELVE_DATA_API_KEY` | `stocks.py`                                | `/stocks/{t}/history`, crypto quotes  |

The stock endpoints don't need the Anthropic key, and vice-versa — each part
degrades independently.

---

## Getting API keys

- **Anthropic** — <https://console.anthropic.com> → API Keys. Pay-as-you-go;
  Haiku is inexpensive.
- **Finnhub** — <https://finnhub.io> → free tier (60 calls/min) covers real-time
  US quotes, symbol search, and market/company news.
- **Twelve Data** — <https://twelvedata.com> → free "Basic" plan (800
  credits/day) covers `time_series` history and crypto for US markets.

---

## Running locally

```bash
cd backend

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env                 # then paste your keys into .env

uvicorn main:app --reload --port 8000
```

Then:

- Interactive API docs (Swagger): <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>
- Point the frontend at it: set `NEXT_PUBLIC_API_URL=http://localhost:8000` in
  `../frontend/.env.local`

Quick smoke tests:

```bash
curl http://localhost:8000/stocks/AAPL/quote
curl "http://localhost:8000/stocks/AAPL/history?range=1M"
curl "http://localhost:8000/news?symbols=AAPL,MSFT"
```

---

## API reference

Base URL: `http://localhost:8000` (or your deployed host).

### `GET /`
Service metadata and endpoint index.

### `GET /health`
Returns agent status, RAG readiness, and the list of registered tools.

```json
{
  "status": "healthy",
  "agent_status": { "anthropic_client": "OK", "finnhub": "OK", "chromadb": "OK" },
  "rag_ready": true,
  "available_tools": ["get_stock_data", "search_financial_news", "search_documents", "save_report_to_drive"]
}
```

### `POST /chat`
Non-streaming chat. Runs the full agent loop and returns the final answer.

Request:
```json
{
  "message": "How is my portfolio doing?",
  "conversation_history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Response:
```json
{
  "response": "Your portfolio is up ...",
  "citations": ["reuters.com", "AAPL_Annual_Report_2024.pdf p.12"],
  "tools_used": ["get_stock_data", "web_search"],
  "query_type": "portfolio"
}
```

### `POST /chat/stream`
Streaming chat via **Server-Sent Events**. Same request body as `/chat`. Emits:

```
data: {"type": "delta", "text": "Your "}
data: {"type": "delta", "text": "portfolio "}
data: {"type": "sources", "citations": ["reuters.com"]}
data: {"type": "done"}
```

`{"type": "error", "message": "..."}` is emitted on failure. The endpoint uses a
sync generator so Starlette flushes each chunk immediately (see
[Troubleshooting](#troubleshooting)).

### `POST /upload`
Upload a PDF (`multipart/form-data`, field `file`). The file is saved to
`documents/`, chunked, embedded into ChromaDB, and added to the BM25 index.

Response:
```json
{ "success": true, "filename": "AAPL_2024.pdf", "message": "Document uploaded and indexed successfully", "chunks_indexed": 142 }
```

### `GET /stocks?q=<query>`
The tradable catalogue (30 curated equities/ETFs + 6 crypto). Optional `q`
filters by ticker or name. Prices are **not** included here — fetch quotes
separately.

```json
{ "stocks": [ { "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology" }, ... ] }
```

### `GET /stocks/quotes?symbols=AAPL,MSFT,BTC`
Live quotes for several tickers at once (used to show prices in the Markets
list). Cached ~60s per symbol. Failed symbols come back with `price: null`.

### `GET /stocks/search?q=<query>`
Searches the **full symbol universe** via Finnhub (plus the local crypto
catalogue), not just the bundled list.

```json
{ "results": [ { "ticker": "KO", "name": "Coca-Cola Co", "sector": "Common Stock" }, ... ] }
```

### `GET /stocks/{ticker}/quote`
Live quote for one ticker. Equities → Finnhub; crypto → Twelve Data.

```json
{
  "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
  "price": 232.14, "change": -1.8, "previous_close": 236.4,
  "high": 235.1, "low": 231.0, "open": 234.7,
  "timestamp": "2026-07-30T14:03:00+00:00"
}
```

### `GET /stocks/{ticker}/history?range=1D|1W|1M|1Y|5Y`
Historical price series via Twelve Data `time_series`.

```json
{ "ticker": "AAPL", "range": "1M", "points": [ { "t": "2026-07-01", "value": 228.4 }, ... ] }
```

Range → interval mapping: `1D`→5min, `1W`→30min, `1M`→1day, `1Y`→1day, `5Y`→1week.

### `GET /news?symbols=AAPL,BTC`
Portfolio-aware market news: recent **company news** for each held equity plus
**general market** headlines, merged, de-duped and sorted. Cached 5 minutes.
Omit `symbols` for general news only.

```json
{
  "articles": [
    { "headline": "...", "summary": "...", "source": "Reuters", "url": "https://...",
      "image": "https://...", "datetime": 1753876800, "related": "AAPL" }
  ]
}
```

---

## The AI agent

Defined in `agent.py`.

- **Model:** `claude-haiku-4-5-20251001` — chosen for low latency. `MAX_TOKENS = 4000`,
  `TEMPERATURE = 0.7`, and **extended thinking is disabled** so text starts
  streaming immediately.
- **Routing:** `workflows.route_query` keyword-classifies each message into
  `portfolio | news | document | rebalancing | general` and selects the matching
  task prompt from `prompts.py`. The prompts enforce plain-text output (no
  markdown) and forbid inline citations, since the UI renders sources as chips.
- **Tools** (`tools.py`), offered to the model each turn:
  - `get_stock_data(tickers)` — live quotes/profiles via Finnhub
  - `search_financial_news(query)` — hands off to Claude's server-side web search
  - `search_documents(query, n_results)` — hybrid RAG search over uploaded PDFs
  - `save_report_to_drive(title, content)` — **stub** that returns "unavailable"
  - plus Anthropic's built-in `web_search` server tool
- **Agent loop:** the model may call tools; `execute_tool` runs them, results are
  appended, and the loop repeats until the model returns a final answer. This is
  implemented for both `run_agent` (blocking) and `stream_agent` (streaming).
- **Citations:** collected from `web_search_tool_result` blocks and RAG results
  (rendered as e.g. `reuters.com` or `report.pdf p.12`), de-duplicated in order.
- **Prompt caching:** `cache.build_cached_prompt` wraps the system prompt with an
  `ephemeral` cache_control block to cut latency/cost on repeat calls.

---

## The RAG pipeline

Defined in `rag.py`. A hybrid retrieval system over uploaded PDFs.

1. **Ingest** (`ingest_documents`) — reads every PDF in `documents/`, extracts
   text per page with `pypdf`.
2. **Chunk** (`chunk_text_recursive`) — token-based chunks of **500 tokens** with
   **50-token overlap** (via `tiktoken` `cl100k_base`), keeping `document_name`
   and `page_number` metadata.
3. **Vector index** (`index_chunks_in_chromadb`) — chunks upserted into a
   persistent ChromaDB collection (`financial_documents`, cosine space).
4. **Keyword index** (`BM25Index`) — a `BM25Okapi` index persisted to
   `bm25_index.json`.
5. **Hybrid search** (`hybrid_search`) — queries both, normalises scores, and
   fuses them with weights **0.7 vector / 0.3 lexical**, returning the top
   results with page citations.

The pipeline is initialised on startup (`initialize_rag_pipeline`) and refreshed
whenever a new PDF is uploaded.

> Note: ChromaDB and the BM25 index persist to disk, but the `documents/` folder
> is ephemeral on most hosting platforms — uploaded PDFs won't survive a redeploy
> unless you attach persistent storage.

---

## Market-data providers

| Data                    | Provider    | Endpoint used                    | Free tier? |
| ----------------------- | ----------- | -------------------------------- | ---------- |
| Equity quote            | Finnhub     | `/quote`                         | ✅         |
| Symbol search           | Finnhub     | `/search`                        | ✅         |
| Company & market news   | Finnhub     | `/company-news`, `/news`         | ✅         |
| Price history           | Twelve Data | `/time_series`                   | ✅         |
| Crypto quote & history  | Twelve Data | `/quote`, `/time_series` (BASE/USD) | ✅       |

`stocks.py` routes automatically: tickers in the crypto set (`BTC, ETH, SOL, XRP,
DOGE, ADA`) use Twelve Data with a `BASE/USD` symbol; everything else is treated
as an equity. Quotes are cached ~60s and news ~5min to respect rate limits.

> Finnhub's **historical candles** (`/stock/candle`) are a *paid* feature, which is
> why price history uses Twelve Data instead.

---

## Deployment

The service runs anywhere that can run a Python web process. It's currently
deployed on **Render**.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment:** set `ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, and
  `TWELVE_DATA_API_KEY` in the host's env settings.

**CORS:** `main.py` allows `http://localhost:3000`, `http://localhost:3001`, and
any `*.vercel.app` origin. Add your production frontend domain there if it's
hosted elsewhere.

> On free tiers the instance **sleeps when idle**; the first request after a
> sleep can take 30–60s to cold-start. Consider a keep-alive ping or a small paid
> instance for demos.

---

## Configuration & tuning

Key knobs and where they live:

| Setting                 | File        | Constant                          |
| ----------------------- | ----------- | --------------------------------- |
| Model / temperature / max tokens | `agent.py`  | `MODEL`, `TEMPERATURE`, `MAX_TOKENS` |
| Chunk size / overlap    | `rag.py`    | `CHUNK_SIZE`, `CHUNK_OVERLAP`     |
| Hybrid search weights   | `rag.py`    | `fuse_results` (0.7 / 0.3)        |
| Quote cache TTL         | `stocks.py` | `QUOTE_TTL`                       |
| News cache TTL          | `stocks.py` | `NEWS_TTL`                        |
| Timeframe → interval    | `stocks.py` | `RANGE_MAP`                       |
| Crypto set              | `stocks.py` | `CRYPTO`                          |
| Query routing keywords  | `workflows.py` | `route_query`                  |

---

## Troubleshooting

- **`/chat/stream` returns everything at once (no streaming).** The endpoint must
  hand `StreamingResponse` a *sync* generator so Starlette runs it in a threadpool
  and flushes each chunk; wrapping the blocking Claude stream in an `async`
  generator blocks the event loop and buffers the whole response. (Already fixed
  in `main.py`.)
- **Quotes/history return HTTP 502.** The upstream provider is unreachable or a
  key is missing/invalid. Confirm `FINNHUB_API_KEY` / `TWELVE_DATA_API_KEY` are
  set. The frontend falls back to a bundled snapshot / synthetic series in this case.
- **AI replies with an API error.** Check `ANTHROPIC_API_KEY` and your Anthropic
  account credit.
- **Prices look fake / wrong.** Make sure you're not using a Finnhub *sandbox*
  token — those return randomised data.
- **RAG returns nothing.** No PDFs indexed yet — upload one via `/upload` (or drop
  files into `documents/` and restart).

---

## Limitations

- **Auth:** none. The service is stateless and assumes a trusted frontend; add
  auth before exposing it publicly.
- **Persistence:** uploaded PDFs live on local disk and don't survive redeploys on
  ephemeral hosts.
- **`save_report_to_drive`** is an intentional stub (returns "unavailable").
- **Crypto news:** Finnhub has no crypto company-news, so crypto is covered by
  general market headlines only.
- **session management:** No session management. Stateless and single-tenant — no login, sessions, or per-user data; uploaded documents and the RAG index are shared across all callers. Add auth and per-user scoping before a public/multi-user deployment.
