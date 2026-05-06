# Financial Content Intelligence Pipeline

A Python pipeline that scrapes financial news and market pages, extracts named
entities (currencies, central banks, indices, commodities, people, etc.),
scores **per-entity sentiment** with cited source spans, runs QA / conflict
checks, and produces three Markdown reports — Trader Brief, Analyst Report,
and Executive Summary — grounded in the scraped content.

> **Market intelligence only. NOT financial advice.**

---

## Features

- **Generic scraper** — works across Yahoo Finance, Reuters, the Federal
  Reserve, IMF, and similar sites without site-specific selectors.
- **Realistic browser headers + retries** — survives 403 / connection-reset
  responses from sites that block default Python User-Agents.
- **Per-entity sentiment** — one LLM call per entity, scored against only
  the excerpts that mention it. A single article can be bullish on one entity
  and bearish on another at the same time.
- **Evidence-grounded** — every sentiment record cites a real `source_span`
  quoted from the scraped text. Fallbacks are built from real source mentions,
  never fabricated.
- **Hybrid QA** — deterministic rules plus an LLM pass flag conflicting
  sentiment, low-confidence entities, duplicates, numerical conflicts, and
  ungrounded claims.
- **Cost & metrics** — every LLM call is logged to `llm_calls.jsonl` with
  hash, tokens, and timestamps; `cost_report.json` aggregates estimated USD
  per stage / source / entity.
- **Failure-tolerant** — failed sources are written to `failed_sources.json`
  and the pipeline keeps going.
- **Validator** — `validate.py` checks every artifact post-run and exits
  non-zero if anything is off-spec (drop-in for CI).

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| LLM provider | [Groq Cloud](https://groq.com) |
| Model | `llama-3.3-70b-versatile` |
| HTTP | `requests` |
| HTML parsing | `beautifulsoup4` |
| Secrets | `python-dotenv` |

See [`TECH_STACK.md`](./TECH_STACK.md) for the full breakdown and the rationale
behind every choice (and a list of things deliberately *not* used).

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd Deriv
pip install -r requirements.txt
```

### 2. Configure your Groq API key

Get a free key at <https://console.groq.com/keys>, then:

```bash
# Linux / macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` and paste your key:

```
GROQ_API_KEY=gsk_your_real_key_here
```

`.env` is gitignored, so the key never reaches version control.

### 3. Run the pipeline

```bash
python pipeline.py
```

You'll see stage names print as the pipeline progresses:

```
[STAGE] INIT
[STAGE] SOURCES_LOADED
  loaded 5 source(s)
[STAGE] CONTENT_FETCHED
  [OK] https://finance.yahoo.com/quote/EURUSD%3DX/ -> 6 record(s) in 0.84s (attempts=1)
  ...
```

### 4. Validate the output

```bash
python validate.py
```

A passing run ends with `VALIDATION PASSED: all checks ok`. Non-zero exit on
failure makes this safe to drop into CI.

---

## Pipeline stages

The pipeline has 11 stages, printed as it runs:

| # | Stage | What it does |
|---|---|---|
| 1 | `INIT` | Verifies API key, makes `reports/`, resets logs. |
| 2 | `SOURCES_LOADED` | Reads URLs from `sources.json`. |
| 3 | `CONTENT_FETCHED` | HTTP-GETs each URL with retries + browser headers. |
| 4 | `CONTENT_EXTRACTED` | Parses HTML into title / body / headlines / numerical signals. |
| 5 | `CONTENT_NORMALISED` | Trims whitespace, marks duplicates, writes `extracted_content.json`. |
| 6 | `ENTITIES_EXTRACTED` | One LLM call returns entities + aliases + mentions. |
| 7 | `ENTITIES_RESOLVED` | Persists `entities.json`, flags low-confidence ones. |
| 8 | `ENTITY_SENTIMENT_SCORED` | One LLM call **per entity** → `entity_sentiment.json`. |
| 9 | `QA_AND_CONFLICTS_CHECKED` | Rules + LLM pass → `qa_report.json`. |
| 10 | `REPORTS_GENERATED` | Three Markdown reports under `reports/`. |
| 11 | `RESULTS_FINALISED` | Writes `cost_report.json` + `run_metrics.json`. |

For a function-by-function walkthrough see
[`CODE_WALKTHROUGH.md`](./CODE_WALKTHROUGH.md).

---

## Output files

```
extracted_content.json   # everything scraped, with source_url on each record
entities.json            # entities + aliases + source_mentions + confidence
entity_sentiment.json    # per-entity sentiment with quoted evidence
qa_report.json           # contradictions, duplicates, low-confidence entities
cost_report.json         # token usage + estimated USD per stage / source / entity
run_metrics.json         # latency, error rate, confidence distribution
llm_calls.jsonl          # one line per LLM call (hash, tokens, timestamps, errors)
failed_sources.json      # only created when sources fail to fetch / extract
reports/
  trader_brief.md        # concise, action-oriented
  analyst_report.md      # detailed, cross-source evidence
  executive_summary.md   # high-level macro & risk
```

### Example record shapes

`extracted_content.json`:
```json
{
  "content_id": "c_a1b2c3d4e5_main",
  "source_url": "https://finance.yahoo.com/quote/EURUSD%3DX/",
  "source_name": "finance.yahoo.com",
  "content_type": "market_data",
  "title": "EUR/USD",
  "body": "...",
  "published_at": "2026-05-06T12:00:00Z",
  "extracted_at": "2026-05-06T12:34:56+00:00",
  "numerical_data": [
    { "label": "EUR/USD", "value": 1.0823, "unit": null,
      "source_span": "EUR/USD trades at 1.0823 after the ECB",
      "source_url": "https://finance.yahoo.com/quote/EURUSD%3DX/" }
  ]
}
```

`entity_sentiment.json`:
```json
{
  "entity_id": "e_002_federal_reserve",
  "canonical_name": "Federal Reserve",
  "sentiment": "bearish",
  "sentiment_score": -0.45,
  "confidence": 0.82,
  "evidence": [
    {
      "content_id": "c_a1b2c3d4e5_main",
      "source_url": "https://www.federalreserve.gov/newsevents/pressreleases.htm",
      "source_span": "Committee voted to maintain the target range...",
      "reason": "Hold signals dovish stance vs market expectations"
    }
  ]
}
```

---

## Configuration

### Sources

Edit `sources.json` to change which URLs are ingested:

```json
{
  "sources": [
    "https://finance.yahoo.com/quote/EURUSD%3DX/",
    "https://www.federalreserve.gov/newsevents/pressreleases.htm",
    "https://www.imf.org/en/News"
  ]
}
```

### Network behavior (in `pipeline.py`)

```python
REQUEST_TIMEOUT = 20            # per-request timeout, seconds
REQUEST_DELAY_SECONDS = 2       # pause between sources
REQUEST_MAX_RETRIES = 3         # attempts per URL before giving up
REQUEST_RETRY_BACKOFF = 2       # linear backoff multiplier between retries
```

### LLM / pricing (in `pipeline.py`)

```python
MODEL = "llama-3.3-70b-versatile"
PRICE_INPUT_PER_M = 0.59        # USD per 1M input tokens
PRICE_OUTPUT_PER_M = 0.79       # USD per 1M output tokens
```

These prices are used only for `cost_report.json`; real billing happens on
Groq's side.

---

## Project structure

```
Deriv/
├── pipeline.py              # main pipeline (11 stages)
├── validate.py              # post-run sanity checks
├── sources.json             # input URLs
├── requirements.txt         # pip dependencies
├── .env.example             # API key template (committed)
├── .env                     # your real key (gitignored)
├── .gitignore               # protects .env
├── README.md                # this file
├── CODE_WALKTHROUGH.md      # plain + detailed code explanation
├── TECH_STACK.md            # tech stack rationale
└── reports/                 # generated reports (created on run)
```

---

## Validation checks

`validate.py` enforces:

1. All required artifacts exist on disk.
2. JSON / JSONL files are valid.
3. At least 2 sources processed successfully.
4. Every content record carries `source_url` + `source_name`.
5. Numerical data preserves `source_span`.
6. Every entity has `aliases` and at least one `source_mention`.
7. Entity sentiment is **entity-specific**, not page-level.
8. Every sentiment record has at least one `evidence` item with all four
   fields populated.
9. Low-confidence entities are flagged consistently.
10. `llm_calls.jsonl` covers all required stages.
11. Every LLM-call record has the mandated fields.

A failure prints `[FAIL]` next to the offending check, lists every failure
at the end, and exits non-zero.

---

## Troubleshooting

**`GROQ_API_KEY is not set`**
You haven't created `.env` yet, or the key inside is empty. Copy
`.env.example` to `.env` and paste your key.

**Some sources 403 or fail with connection reset**
Expected — sites like Reuters and the IMF aggressively block bots. The
pipeline retries, then writes the failure to `failed_sources.json` and
continues. As long as ≥2 sources succeed, validation still passes.

**Validator says `sources_succeeded<2`**
All your sources are getting blocked. Try swapping in different URLs
(financial blogs, government press-release pages) in `sources.json`.

**`json_decode_failed` in `llm_calls.jsonl`**
Groq returned non-JSON for one stage. The pipeline tries to recover with a
regex; if recovery also fails, that stage is logged as an error but the run
continues with sensible defaults.

---

## Limitations

- The scraper is generic, so it can occasionally pick up navigation /
  boilerplate text. The QA pass and content-type heuristics mitigate this
  but don't eliminate it.
- The cost numbers are estimates based on the published Groq pricing; the
  actual bill comes from Groq.
- `entities.json` is rebuilt from scratch on every run — there's no
  cross-run deduplication of entities yet.

---

## Disclaimer

This project produces **market intelligence only**. Nothing it generates is
financial advice. All scraped figures may be stale or wrong. Do not trade on
the output of this pipeline without verifying against authoritative sources.

---

## License

Specify a license here (MIT / Apache-2.0 / proprietary) before publishing the
repo. The current code does not include a license file.
