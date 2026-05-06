# Code Walkthrough — Financial Content Intelligence Pipeline

This document explains how `pipeline.py` and `validate.py` work.

- **Part 1** is a plain-English overview. Read this if you just want to know what the code does.
- **Part 2** is a function-by-function deep dive. Read this if you want to modify or extend the code.

---

# Part 1 — Simplified Version

## What is this project?

It's a Python pipeline that:

1. Reads a list of finance website URLs from `sources.json`.
2. Scrapes each page for headlines, articles, prices, and named people / institutions.
3. Asks an LLM (Groq's `llama-3.3-70b-versatile`) to identify financial entities (currencies, central banks, indices, etc.) and merge their aliases (e.g. `the Fed` = `Federal Reserve`).
4. Asks the LLM to score sentiment **per entity** (not per page), with quoted evidence.
5. Runs a QA pass that flags conflicts, low-confidence entities, duplicates, and ungrounded claims.
6. Writes three reports — a **Trader Brief**, an **Analyst Report**, and an **Executive Summary** — into `reports/`.
7. Logs every LLM call, estimates cost, and saves run metrics.

The whole thing runs end-to-end with `python pipeline.py`. After that, `python validate.py` checks every output is valid.

## The 11 stages, one line each

| # | Stage | What it does |
|---|---|---|
| 1 | `INIT` | Checks the API key, makes the `reports/` folder, wipes the LLM log. |
| 2 | `SOURCES_LOADED` | Reads `sources.json` from disk. |
| 3 | `CONTENT_FETCHED` | HTTP-GETs each URL with a real-browser User-Agent. |
| 4 | `CONTENT_EXTRACTED` | Parses HTML into title / body / headlines / numerical data. |
| 5 | `CONTENT_NORMALISED` | Trims whitespace, marks near-duplicates, saves `extracted_content.json`. |
| 6 | `ENTITIES_EXTRACTED` | One LLM call returns all entities with aliases + mentions. |
| 7 | `ENTITIES_RESOLVED` | Persists entities to `entities.json` and flags low-confidence ones. |
| 8 | `ENTITY_SENTIMENT_SCORED` | One LLM call per entity → `entity_sentiment.json`. |
| 9 | `QA_AND_CONFLICTS_CHECKED` | Rule-based + LLM checks for conflicts → `qa_report.json`. |
| 10 | `REPORTS_GENERATED` | Three LLM calls produce three Markdown reports under `reports/`. |
| 11 | `RESULTS_FINALISED` | Writes `cost_report.json` + `run_metrics.json` and prints total cost. |

## Files produced

```
extracted_content.json   # everything we scraped, with source_url on each row
entities.json            # entities + aliases + source_mentions + confidence
entity_sentiment.json    # per-entity sentiment with quoted evidence
qa_report.json           # contradictions, duplicates, low-confidence entities
cost_report.json         # token usage + estimated USD per stage / source / entity
run_metrics.json         # latency, error rate, confidence distribution
llm_calls.jsonl          # one line per LLM call (hash, tokens, timestamps, errors)
reports/trader_brief.md
reports/analyst_report.md
reports/executive_summary.md
```

## Important guarantees

- **No hardcoded content** — only `sources.json` URLs and the LLM model name are static. Everything else comes from the live pages.
- **Source attribution everywhere** — every record carries `source_url` and `source_name`.
- **Entity-specific sentiment** — sentiment is scored once per entity, against only the excerpts that mention it. A single article can be bullish on USD and bearish on EUR/USD at the same time.
- **Evidence required** — every sentiment must cite a `source_span` quoted from the scraped text.
- **Failures are surfaced, not hidden** — fetch errors, JSON-decode errors, and contradictions are all written to disk instead of silently dropped.
- **Disclaimer** — every report opens with "market intelligence only, NOT financial advice".

## How to run it

```
pip install -r requirements.txt
copy .env.example .env        # then edit .env with your GROQ_API_KEY
python pipeline.py
python validate.py
```

---

# Part 2 — Detailed Version

This part walks through the code file-by-file, top-to-bottom.

## File: `pipeline.py`

### Configuration block (top of file)

```python
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
PRICE_INPUT_PER_M = 0.59
PRICE_OUTPUT_PER_M = 0.79
```

- `load_dotenv()` reads `.env` so `GROQ_API_KEY` becomes available via `os.getenv`.
- `client` is the single Groq SDK client used for every LLM call.
- `MODEL` is the only place the model ID is set — change it once and it propagates everywhere.
- `PRICE_INPUT_PER_M` / `PRICE_OUTPUT_PER_M` are Groq's published USD prices per million tokens. They are used **only** for the cost report; actual billing is on Groq's side.

```python
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
LLM_LOG_PATH = ROOT / "llm_calls.jsonl"
```

All paths are absolute and anchored to the script's directory, so the pipeline works no matter where you `cd` from.

```python
REQUEST_HEADERS = { "User-Agent": "...Chrome...", ... }
REQUEST_TIMEOUT = 20
```

A real-looking desktop User-Agent is used because finance sites (Yahoo, Reuters) often block default Python UA strings.

```python
RUN_STATE = {
    "stage_timings": {}, "source_latency": {}, "extraction_errors": [],
    "llm_calls": [], "started_at": None, "finished_at": None,
}
```

A single mutable dict that every stage updates. At the end it feeds the cost and metrics reports.

### Utility helpers

#### `now_iso()`
Returns the current UTC time as an ISO-8601 string. Used everywhere a timestamp is required (run metrics, llm_calls.jsonl, content `extracted_at`).

#### `stage(name)` and `stage_done(name)`
- `stage()` prints `[STAGE] NAME` to stdout and records the start time in `RUN_STATE["stage_timings"]`.
- `stage_done()` records the end time. Together they let `run_metrics.json` show wall-clock duration per stage.

#### `estimate_tokens(text)`
A rough fallback: `len(text) // 4`. Only used when the Groq API doesn't return a real `usage` object on the response.

#### `hash_prompt(text)`
SHA-256 hash of `system + "\n" + user`. Stored in every LLM-call record so identical prompts can be detected (useful for caching or deduping).

#### `slugify_host(url)`
Returns a short readable name like `finance.yahoo.com` for a given URL. Used as `source_name` on every content record.

#### `safe_write_json(path, data)`
Wraps `json.dump` with `indent=2`, `ensure_ascii=False`, and parent-dir creation. Used for all JSON outputs.

#### `log_llm_call(record)`
Appends one record to both `RUN_STATE["llm_calls"]` (for the cost report) and `llm_calls.jsonl` on disk (for auditing).

#### `call_llm(...)` — the LLM wrapper
This is the single function every stage uses to talk to Groq. It:

1. Hashes the prompt.
2. Calls `client.chat.completions.create(...)` with `temperature=0.1` (low randomness for reproducibility) and `response_format={"type": "json_object"}` when `json_mode=True`.
3. Parses the response. If JSON parsing fails it tries to recover with a regex that finds the first `{...}` block.
4. Pulls real token counts from `response.usage` if available; otherwise estimates them from char length.
5. Catches **every** exception (network, auth, rate-limit) and stores the error string in the log record instead of crashing the pipeline.
6. Logs a record to `llm_calls.jsonl` with `stage`, `source_url`, `content_ids`, `timestamp`, `provider`, `model`, `prompt_hash`, `input_artifacts`, `output_artifact`, `estimated_input_tokens`, `estimated_output_tokens`, `duration_seconds`, `error`.
7. Returns `{"raw": str, "parsed": dict|None, "error": str|None}`.

This wrapper is what makes the rest of the pipeline simple — every stage just constructs a system + user prompt and trusts the wrapper to handle errors and logging.

### Stage 1 — `load_sources()`

Reads `sources.json`, asserts it has a non-empty `sources` list, and returns the URLs as a list of strings. Raises if the file is missing or malformed (which fails the run loudly rather than silently).

### Stage 2 — Fetching and extracting content

#### `fetch_url(url)`
A thin wrapper over `requests.get` with the desktop UA and a 20-second timeout. Returns `(html, error)` so the caller can decide whether to keep going.

#### `classify_content(url, title)`
Heuristics that return one of `headline | article | market_data | press_release | other` based on URL substrings (`pressrelease`, `markets`, `news`, `quote`) and title length.

#### `extract_published_at(soup)`
Tries a list of common publication-time meta tags (`article:published_time`, `pubdate`, `itemprop="datePublished"`, plain `<time>` tags) and returns the first non-empty value. `None` when nothing matches.

#### `extract_numerical_data(text, source_url)`
A regex (`PRICE_PATTERN`) finds tokens like `1.0823`, `4.25%`, `25 bps`, `$150`. For each match it captures:
- `label` — up to 60 chars of text immediately before the number (gives context).
- `value` — the number, comma-stripped and parsed as `float`.
- `unit` — `%`, `bps`, `bn`, `usd`, etc., or `None`.
- `source_span` — ~80 chars surrounding the number (used by QA to show where the number came from).
- `source_url` — the page it came from.

It caps at 25 numbers per record to avoid runaway extraction on tables-heavy pages.

#### `extract_from_html(url, html)`
The core scraper:
1. Strips `<script>`, `<style>`, `<noscript>`.
2. Picks a title — `<h1>` if present, else `<title>`.
3. Collects all `<p>` tags whose text is longer than 30 chars, joins up to 80 of them as the body.
4. If no usable paragraphs, falls back to all visible text (capped at 8000 chars).
5. Pulls up to 15 `<h2>` / `<h3>` tags as sub-headlines.
6. Returns one "main" content record plus up to 5 headline records, each with their own `content_id` (a SHA-1 prefix of the URL plus a suffix).

This is intentionally generic — Yahoo, Reuters, the Fed, and the IMF all have different DOMs, so we don't write per-site selectors.

#### `fetch_and_extract(sources)`
Iterates sources, times each fetch (writes to `RUN_STATE["source_latency"]`), logs failures (writes to `RUN_STATE["extraction_errors"]`), and returns a flat list of all content records.

### Stage 3 — `normalise_content(records)`

Two jobs:

1. Collapse repeated whitespace and strip empties from `title` / `body`.
2. Hash `(title + body[:500])` to detect near-duplicates. The first occurrence is the "canonical" one; later duplicates get `duplicate_of = canonical_id`.

Records with empty title and body are dropped. Saved to `extracted_content.json`.

### Stages 4 + 5 — Entity extraction and resolution

#### `ENTITY_EXTRACTION_SYSTEM`
A system prompt that defines the entity types we care about (currency_pair, currency, index, commodity, central_bank, economic_indicator, company, person, country, policy_event), demands strict JSON output, and gives concrete alias examples (`the Fed = Federal Reserve = US central bank`, `greenback = USD`, etc.).

#### `build_entity_extraction_input(records)`
Concatenates all non-duplicate records into a single user prompt, capped at 18 000 chars and per-body capped at 2 500 chars. Caps exist because the model has a finite context window and we want to keep token cost predictable.

#### `extract_entities(records)`
1. Calls `call_llm` once with the prompt above.
2. For each entity returned by the LLM, generates a stable `entity_id` (`e_001_federal_reserve`-style).
3. Decorates each `mention` with its `source_url` (looked up from the content record).
4. Sets `low_confidence_flag = True` when `resolution_confidence < 0.6`.
5. Saved (at the next stage boundary) to `entities.json`.

The "Stages 4 and 5" split is logical — both are filled by a single LLM call. Stage 5 is just the persistence checkpoint.

### Stage 6 — `score_entity_sentiment(entity, records_by_id)`

This is where the **per-entity** rule lives.

For one entity:
1. Collect every `content_id` that mentioned it (deduped).
2. Build excerpts containing **only those** content records — never the whole corpus. This is critical: if we sent every record we'd get page-level sentiment, not entity-specific.
3. Call `call_llm` with a system prompt that explicitly says: "score sentiment FOR THAT ENTITY ONLY. A single article can be bullish on one entity and bearish on another."
4. Demand strict JSON with `sentiment` (bullish/bearish/neutral/mixed), `sentiment_score` (-1.0..1.0), `confidence` (0.0..1.0), and an `evidence[]` list with `content_id`, `source_url`, `source_span`, `reason`.

The pipeline runs this **once per entity** in a Python loop. Each call's record gets logged with `stage="ENTITY_SENTIMENT_SCORED"` so the cost report can show how many sentiment calls happened.

### Stage 7 — `run_qa(records, entities, sentiments)`

A hybrid: deterministic Python rules **plus** a single LLM pass.

Deterministic rules:
1. Every entity with `low_confidence_flag=True` → `unresolved_entity` warning.
2. Every record with `duplicate_of` → `duplicate_content` info.
3. For each entity, if `max(sentiment_score) > 0.3` and `min(sentiment_score) < -0.3` across sources → `conflicting_sentiment` critical.

LLM rules (one call):
- Send a compact JSON with entities, sentiments, and numerical_data.
- Ask for additional issues — particularly `numerical_conflict` (same metric disagrees across sources) and `ungrounded_claim` (sentiment evidence that doesn't match any source span).

Each issue gets a unique `issue_id` (`q_001`, `q_002`, ...). Saved to `qa_report.json`.

### Stage 8 — `generate_reports(records, entities, sentiments, qa_issues)`

Three Markdown reports, three LLM calls:

| File | Tone |
|---|---|
| `reports/trader_brief.md` | Concise, action-oriented. Bullet points on price levels and momentum. |
| `reports/analyst_report.md` | Detailed. Cross-source evidence, per-entity confidence, contradictions. |
| `reports/executive_summary.md` | High-level. Macro trends and risk for senior leadership. |

Each report:
- Uses the same payload (`_report_payload(...)`) — entities, sentiments, qa_issues, numerical signals, headlines — so all three reports are grounded in the same evidence.
- Has a system prompt instructing the LLM to **only use facts present in the supplied JSON** (this is the anti-hallucination guardrail).
- Is written to disk with the disclaimer banner prepended.

### Stages 9–10 — Cost and metrics

#### `build_cost_report(records, entities)`
Aggregates everything in `RUN_STATE["llm_calls"]`:
- Sums `estimated_input_tokens` and `estimated_output_tokens`.
- Multiplies by Groq's per-million prices and converts to USD.
- Bucket-aggregates by stage (`by_stage[stage] = {calls, input_tokens, output_tokens, errors}`).
- Computes per-source and per-entity averages.
- Adds `deduplicated_content_count` (how many records were skipped because they were duplicates — i.e. the LLM-call savings).

#### `build_run_metrics(records, entities, sources)`
- Stage timings (start / end per stage).
- Per-source latency.
- Extraction errors and the resulting error rate.
- Resolution-confidence distribution (`0.0–0.6 / 0.6–0.8 / 0.8–1.0` buckets).

### `main()`

The orchestrator:
1. Records `started_at`, deletes any existing `llm_calls.jsonl` (so each run is clean).
2. Calls `stage("INIT")` then verifies the API key is set; bails with exit code 2 if not.
3. Calls each stage in order, printing `[STAGE] NAME` before each.
4. Wraps the whole thing in `try/except` so a crash prints a traceback and exits 1 instead of leaving the user wondering.

Each stage:
- Prints its name.
- Does its work.
- Saves its main artifact to disk.
- Calls `stage_done(...)` to record the end time.

---

## File: `validate.py`

`validate.py` is independent of `pipeline.py` — it reads only the artifacts on disk.

### `REQUIRED_FILES` and `REQUIRED_LLM_STAGES`
The exact list of files that must exist after a successful run, plus the stage names that must appear in `llm_calls.jsonl` (`ENTITIES_EXTRACTED`, `ENTITY_SENTIMENT_SCORED`, `QA_AND_CONFLICTS_CHECKED`, `REPORTS_GENERATED`).

### Helpers
- `load_json(path)` / `load_jsonl(path)` — small wrappers with friendly errors.
- `check(name, condition, detail)` — prints `[PASS]` or `[FAIL]` and returns the bool.

### The 11 checks (in order)

1. **All required artifacts exist.** If any are missing, deeper validation is aborted.
2. **JSON files parse cleanly.** Catches half-written or empty outputs.
3. **At least 2 sources processed.** Reads `run_metrics.json`'s `sources_succeeded` field.
4. **Every content record has `source_url` + `source_name`.** Enforces the source-attribution rule.
5. **Numerical data preserves `source_span`.** Confirms scraped numbers carry quoted context.
6. **Every entity has `aliases` and at least one `source_mention`.** Enforces the alias-resolution requirement.
7. **Entity sentiment is entity-specific, not page-level.** Heuristic: if multiple entities share a `content_id`, their sentiment scores should not all be identical. (Articles with only one entity per content are exempt.)
8. **Sentiment records include evidence spans.** Every `evidence[i]` must have a `source_span`.
9. **Low-confidence entities flagged.** `low_confidence_flag` must equal `resolution_confidence < 0.6`.
10. **`llm_calls.jsonl` covers required stages.** All four mandatory stages must appear.
11. **Every `llm_calls.jsonl` record has the mandated fields.** Specifically: `stage, source_url, content_ids, timestamp, provider, model, prompt_hash, input_artifacts, output_artifact, estimated_input_tokens, estimated_output_tokens`.

### Exit code
- `0` if every check passes.
- `1` if anything fails (and a list of failed checks is printed).

This makes `validate.py` safe to drop into CI: a non-zero exit means the pipeline produced something off-spec and the build should fail.

---

## How to extend the pipeline

A few entry points designed for modification:

| Want to... | Edit... |
|---|---|
| Add a source | `sources.json` |
| Switch the LLM | `MODEL` constant in `pipeline.py` |
| Change pricing | `PRICE_INPUT_PER_M` / `PRICE_OUTPUT_PER_M` |
| Add an entity type | The list inside `ENTITY_EXTRACTION_SYSTEM` |
| Add a QA rule | `run_qa()` — either as a Python rule or a new field in the QA system prompt |
| Add a report | Append a new spec tuple to the `specs` list inside `generate_reports()` |
| Change scraper logic per-site | Replace `extract_from_html` with a site-specific dispatcher |
| Add a validation check | Append to `validate.py` and bump `REQUIRED_FILES` if new outputs are produced |
