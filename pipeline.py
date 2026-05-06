"""
Financial Content Intelligence Pipeline.

Reads URLs from sources.json, scrapes them, extracts entities + per-entity
sentiment via Groq, runs QA / conflict detection, and emits trader / analyst /
executive reports plus cost + run metrics.

Output is market intelligence only and is NOT financial advice.
"""

import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# Approximate Groq prices (USD per 1M tokens) for llama-3.3-70b-versatile.
# Used only for cost reporting; real billing is on Groq's side.
PRICE_INPUT_PER_M = 0.59
PRICE_OUTPUT_PER_M = 0.79

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
LLM_LOG_PATH = ROOT / "llm_calls.jsonl"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

REQUEST_TIMEOUT = 20
REQUEST_DELAY_SECONDS = 2  # delay between sources to avoid rate limiting
REQUEST_MAX_RETRIES = 3  # total attempts per URL before giving up
REQUEST_RETRY_BACKOFF = 2  # seconds between retry attempts (linear)
FAILED_SOURCES_PATH = "failed_sources.json"

# Mutable run-state populated as the pipeline runs. Used for metrics + cost.
RUN_STATE = {
    "stage_timings": {},
    "source_latency": {},
    "extraction_errors": [],
    "llm_calls": [],
    "started_at": None,
    "finished_at": None,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def stage(name: str) -> None:
    """Print the current pipeline stage and remember when it started."""
    print(f"[STAGE] {name}")
    RUN_STATE["stage_timings"][name] = {"started_at": now_iso()}


def stage_done(name: str) -> None:
    """Record stage completion time."""
    RUN_STATE["stage_timings"].setdefault(name, {})["finished_at"] = now_iso()


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text using the rough 4-chars-per-token rule."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def hash_prompt(text: str) -> str:
    """Hash a prompt so identical prompts can be deduped / cached."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def slugify_host(url: str) -> str:
    """Return a short, human-readable name for a URL based on its host."""
    try:
        host = urlparse(url).netloc or url
        host = host.replace("www.", "")
        return host
    except Exception:
        return url


def safe_write_json(path: Path, data) -> None:
    """Write JSON to disk with utf-8 + indentation, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def log_llm_call(record: dict) -> None:
    """Append a single LLM call record to llm_calls.jsonl and run state."""
    RUN_STATE["llm_calls"].append(record)
    with LLM_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(
    *,
    stage_name: str,
    system: str,
    user: str,
    source_url: str | None = None,
    content_ids: list | None = None,
    input_artifacts: list | None = None,
    output_artifact: str | None = None,
    json_mode: bool = True,
    temperature: float = 0.1,
) -> dict:
    """
    Wrapper around Groq chat completion that:
      * forces JSON-mode output by default,
      * estimates token usage,
      * logs every call to llm_calls.jsonl, and
      * returns ``{"raw": str, "parsed": dict|None}``.
    """
    prompt_hash = hash_prompt(system + "\n" + user)
    started = time.time()
    raw_text = ""
    parsed = None
    error = None

    try:
        kwargs = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        raw_text = response.choices[0].message.content or ""

        if json_mode:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                # Try to recover JSON from a fenced block or first {...} chunk.
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except json.JSONDecodeError as e:
                        error = f"json_decode_failed: {e}"
                else:
                    error = "json_decode_failed: no object found"

        # Prefer real usage from API if available, otherwise estimate.
        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", None) or estimate_tokens(
            system + user
        )
        out_tokens = getattr(usage, "completion_tokens", None) or estimate_tokens(
            raw_text
        )

    except Exception as exc:  # network, auth, rate-limit, etc.
        error = f"{type(exc).__name__}: {exc}"
        in_tokens = estimate_tokens(system + user)
        out_tokens = 0

    record = {
        "stage": stage_name,
        "source_url": source_url,
        "content_ids": content_ids or [],
        "timestamp": now_iso(),
        "provider": "groq",
        "model": MODEL,
        "prompt_hash": prompt_hash,
        "input_artifacts": input_artifacts or [],
        "output_artifact": output_artifact,
        "estimated_input_tokens": in_tokens,
        "estimated_output_tokens": out_tokens,
        "duration_seconds": round(time.time() - started, 3),
        "error": error,
    }
    log_llm_call(record)

    return {"raw": raw_text, "parsed": parsed, "error": error}


# ---------------------------------------------------------------------------
# Stage 1 - Load sources
# ---------------------------------------------------------------------------


def load_sources() -> list[str]:
    """Read the list of source URLs from sources.json on disk."""
    path = ROOT / "sources.json"
    if not path.exists():
        raise FileNotFoundError(f"sources.json not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    sources = data.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources.json must contain a non-empty 'sources' list")
    return [s for s in sources if isinstance(s, str) and s.strip()]


# ---------------------------------------------------------------------------
# Stage 2 - Fetch and extract content
# ---------------------------------------------------------------------------


# Patterns for pulling raw numerical signals out of scraped article bodies.
PRICE_PATTERN = re.compile(
    r"(?P<value>\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>%|bps|basis points|bn|billion|million|trillion|usd|eur|gbp|jpy|"
    r"\$|€|£|¥)?",
    re.IGNORECASE,
)


def _build_session() -> requests.Session:
    """Build a single requests.Session preloaded with realistic browser headers.

    Sites like Reuters and the IMF return 403 / drop the connection when they
    see default Python User-Agents or missing Accept-* headers. A persistent
    Session also lets us reuse the underlying TCP connection.
    """
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    return session


def fetch_url(
    url: str,
    session: requests.Session | None = None,
    max_retries: int = REQUEST_MAX_RETRIES,
) -> tuple[str | None, str | None, int]:
    """Fetch a URL with retries.

    Returns ``(html, error, attempts)``. ``html`` is the response body on
    success; ``error`` is a human-readable message on failure; ``attempts`` is
    the number of attempts made. Never raises - the caller decides what to do
    with a failed source.
    """
    sess = session or _build_session()
    last_err: str | None = None
    attempts = 0

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            resp = sess.get(
                url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return resp.text, None, attempts
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                # Linear backoff between retries; small enough to keep the run
                # moving but enough to clear most transient throttles.
                time.sleep(REQUEST_RETRY_BACKOFF * attempt)

    return None, last_err, attempts


def classify_content(url: str, title: str) -> str:
    """Pick a content_type from the URL + title heuristics."""
    u = url.lower()
    t = (title or "").lower()
    if "pressrelease" in u or "press-release" in u or "press release" in t:
        return "press_release"
    if "quote" in u or "markets" in u or "/markets/" in u:
        return "market_data"
    if "news" in u or "article" in u or "/story/" in u:
        return "article"
    if t and len(t) < 140:
        return "headline"
    return "other"


def extract_published_at(soup: BeautifulSoup) -> str | None:
    """Try to recover a publication timestamp from common meta tags."""
    candidates = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "pubdate"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"name": "date"}),
        ("meta", {"itemprop": "datePublished"}),
        ("time", {}),
    ]
    for tag, attrs in candidates:
        node = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
        if not node:
            continue
        value = node.get("content") or node.get("datetime") or node.get_text(strip=True)
        if value:
            return value
    return None


def extract_numerical_data(text: str, source_url: str) -> list[dict]:
    """Pull out numerical signals (prices, %, bps, etc.) from a body of text."""
    out = []
    if not text:
        return out
    for match in PRICE_PATTERN.finditer(text):
        value_raw = match.group("value")
        unit = (match.group("unit") or "").strip() or None
        if not unit and not re.search(r"\d\.\d", value_raw):
            # plain integers without units are too noisy - skip
            continue
        try:
            value_num = float(value_raw.replace(",", ""))
        except ValueError:
            continue
        span_start = max(0, match.start() - 40)
        span_end = min(len(text), match.end() + 40)
        out.append(
            {
                "label": text[span_start : match.start()].strip()[-60:],
                "value": value_num,
                "unit": unit,
                "source_span": text[span_start:span_end].strip(),
                "source_url": source_url,
            }
        )
        if len(out) >= 25:  # cap to avoid runaway extraction
            break
    return out


def extract_from_html(url: str, html: str) -> list[dict]:
    """
    Parse a fetched page into one or more content records.

    Yahoo / Reuters / Fed / IMF have different layouts, so we use a generic
    approach: take the <title>, headline-looking elements, and the largest
    block of text we can find.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 30]
    body = "\n".join(paragraphs[:80])  # cap to keep token use reasonable

    # If we got nothing useful, fall back to visible text (still capped).
    if not body:
        body = soup.get_text(" ", strip=True)[:8000]

    headlines = []
    for h in soup.find_all(["h2", "h3"]):
        text = h.get_text(" ", strip=True)
        if 10 < len(text) < 220:
            headlines.append(text)
        if len(headlines) >= 15:
            break

    published_at = extract_published_at(soup)

    base_record = {
        "source_url": url,
        "source_name": slugify_host(url),
        "published_at": published_at,
        "extracted_at": now_iso(),
    }

    records = []
    main_record = {
        "content_id": f"c_{hashlib.sha1(url.encode()).hexdigest()[:10]}_main",
        **base_record,
        "content_type": classify_content(url, title),
        "title": title,
        "body": body,
        "numerical_data": extract_numerical_data(body, url),
    }
    records.append(main_record)

    for i, headline in enumerate(headlines[:5]):
        # Capture sub-headlines as their own content rows so sentiment can
        # later score them against detected entities.
        records.append(
            {
                "content_id": f"c_{hashlib.sha1(url.encode()).hexdigest()[:10]}_h{i}",
                **base_record,
                "content_type": "headline",
                "title": headline,
                "body": headline,
                "numerical_data": extract_numerical_data(headline, url),
            }
        )

    return records


def _record_failed_source(
    url: str, err: str, attempts: int, stage_name: str
) -> None:
    """Persist a failure to ``failed_sources.json`` so we never silently drop a source.

    Each failure is appended to a top-level "failures" list with the URL, the
    error message, the number of attempts made, the stage that failed, and a
    timestamp. Existing failures are preserved across runs of this function
    within a single pipeline execution but rewritten cleanly for a brand-new
    run (the file is reset by ``main()`` at startup).
    """
    path = ROOT / FAILED_SOURCES_PATH
    payload = {"failures": []}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if isinstance(existing, dict) and isinstance(
                existing.get("failures"), list
            ):
                payload = existing
        except Exception:
            # File was corrupt; start fresh rather than crashing the pipeline.
            payload = {"failures": []}

    payload["failures"].append(
        {
            "source_url": url,
            "stage": stage_name,
            "error": err,
            "attempts": attempts,
            "timestamp": now_iso(),
        }
    )
    safe_write_json(path, payload)


def fetch_and_extract(sources: list[str]) -> list[dict]:
    """Fetch every source, extract content, and survive any individual failure.

    A single ``requests.Session`` is reused across all sources so connection
    pooling + cookies are shared. A ``REQUEST_DELAY_SECONDS`` pause sits
    between sources to stay under per-host rate limits. Any source that fails
    (after the retries inside ``fetch_url``) is logged to ``failed_sources.json``
    and the loop continues to the next URL - the pipeline never crashes on a
    single bad source.
    """
    all_records: list[dict] = []
    session = _build_session()

    for index, url in enumerate(sources):
        # Polite pacing: sleep BEFORE every request except the first.
        if index > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

        t0 = time.time()
        try:
            html, err, attempts = fetch_url(url, session=session)
        except Exception as exc:
            # Defensive: fetch_url is meant to never raise, but if it does we
            # still keep the pipeline alive.
            html, err, attempts = None, f"{type(exc).__name__}: {exc}", 0
        latency = round(time.time() - t0, 3)
        RUN_STATE["source_latency"][url] = latency

        if err or not html:
            message = err or "empty_body"
            print(
                f"  [WARN] fetch failed for {url} after {attempts} attempt(s): {message}"
            )
            RUN_STATE["extraction_errors"].append(
                {
                    "source_url": url,
                    "stage": "fetch",
                    "error": message,
                    "attempts": attempts,
                }
            )
            _record_failed_source(url, message, attempts, "fetch")
            continue

        try:
            records = extract_from_html(url, html)
            all_records.extend(records)
            print(
                f"  [OK] {url} -> {len(records)} record(s) in {latency}s "
                f"(attempts={attempts})"
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  [WARN] extract failed for {url}: {err_msg}")
            RUN_STATE["extraction_errors"].append(
                {
                    "source_url": url,
                    "stage": "extract",
                    "error": err_msg,
                    "attempts": attempts,
                }
            )
            _record_failed_source(url, err_msg, attempts, "extract")

    return all_records


# ---------------------------------------------------------------------------
# Stage 3 - Normalise content
# ---------------------------------------------------------------------------


def normalise_content(records: list[dict]) -> list[dict]:
    """
    Tidy whitespace, drop empty bodies, and mark near-duplicates so later
    LLM calls can avoid scoring the same text twice.
    """
    seen_hashes: dict[str, str] = {}
    cleaned: list[dict] = []

    for r in records:
        body = re.sub(r"\s+", " ", (r.get("body") or "")).strip()
        title = re.sub(r"\s+", " ", (r.get("title") or "")).strip()
        if not body and not title:
            continue
        r["body"] = body
        r["title"] = title

        # Mark duplicates by hashing the first ~500 chars of body+title.
        digest = hashlib.sha1((title + "|" + body[:500]).encode()).hexdigest()
        if digest in seen_hashes:
            r["duplicate_of"] = seen_hashes[digest]
        else:
            seen_hashes[digest] = r["content_id"]
            r["duplicate_of"] = None

        cleaned.append(r)

    return cleaned


# ---------------------------------------------------------------------------
# Stage 4 - Entity extraction + resolution
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_SYSTEM = """You are a financial NER system. Extract financial entities
from the provided content.

Entity types you care about:
- currency_pair (EUR/USD, GBP/JPY, ...)
- currency (USD, EUR, JPY, ...)
- index (S&P 500, FTSE 100, Nikkei 225, ...)
- commodity (gold, brent crude, WTI, copper, ...)
- central_bank (Federal Reserve, ECB, BoE, BoJ, ...)
- economic_indicator (CPI, NFP, GDP, PMI, ...)
- company (publicly traded firms)
- person (CEOs, central bankers, finance ministers, ...)
- country (United States, Japan, ...)
- policy_event (rate decision, FOMC meeting, ...)

Return STRICT JSON of shape:
{"entities": [
   {"canonical_name": str,
    "entity_type": str,
    "aliases": [str],
    "mentions": [{"content_id": str, "mention_text": str, "source_span": str}],
    "resolution_confidence": float}
]}

resolution_confidence is your 0..1 confidence the entity is correctly resolved.
Resolve aliases: "the Fed"="Federal Reserve"="US central bank";
"greenback"="USD"="US dollar"; "EUR/USD"="EURUSD"="euro-dollar";
"NFP"="nonfarm payrolls". Always return JSON only."""


def build_entity_extraction_input(records: list[dict]) -> str:
    """Build a single user prompt from all unique content records."""
    chunks = []
    for r in records:
        if r.get("duplicate_of"):
            continue
        body = r.get("body") or ""
        chunks.append(
            f"---\ncontent_id: {r['content_id']}\n"
            f"source_url: {r['source_url']}\n"
            f"title: {r.get('title','')}\n"
            f"body: {body[:2500]}"
        )
    return "\n".join(chunks)[:18000]


def extract_entities(records: list[dict]) -> list[dict]:
    """Single LLM call that returns extracted + alias-resolved entities."""
    user_prompt = build_entity_extraction_input(records)
    if not user_prompt.strip():
        print("  [WARN] no usable content for entity extraction")
        return []

    result = call_llm(
        stage_name="ENTITIES_EXTRACTED",
        system=ENTITY_EXTRACTION_SYSTEM,
        user=user_prompt,
        content_ids=[r["content_id"] for r in records if not r.get("duplicate_of")],
        input_artifacts=["extracted_content.json"],
        output_artifact="entities.json",
    )
    if result["error"] or not result["parsed"]:
        print(f"  [WARN] entity extraction LLM failed: {result['error']}")
        return []

    raw_entities = result["parsed"].get("entities", []) or []
    enriched: list[dict] = []
    for i, ent in enumerate(raw_entities):
        canonical = (ent.get("canonical_name") or "").strip()
        if not canonical:
            continue
        mentions = ent.get("mentions") or []
        # Decorate mentions with source_url by looking up content_id.
        url_by_id = {r["content_id"]: r["source_url"] for r in records}
        for m in mentions:
            m["source_url"] = url_by_id.get(m.get("content_id"), None)

        enriched.append(
            {
                "entity_id": f"e_{i:03d}_{re.sub(r'[^A-Za-z0-9]+','_',canonical).strip('_').lower()}"[
                    :60
                ],
                "canonical_name": canonical,
                "entity_type": ent.get("entity_type", "unknown"),
                "aliases": ent.get("aliases", []) or [],
                "source_mentions": mentions,
                "resolution_confidence": float(
                    ent.get("resolution_confidence", 0.5) or 0.5
                ),
                "low_confidence_flag": float(
                    ent.get("resolution_confidence", 0.5) or 0.5
                )
                < 0.6,
            }
        )
    return enriched


# ---------------------------------------------------------------------------
# Stage 5 - Per-entity sentiment
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM = """You are a financial sentiment analyst. You will receive ONE entity
and a set of source excerpts that mention it. Score sentiment FOR THAT ENTITY ONLY.

A single article can be bullish on one entity and bearish on another - score
only the supplied entity, not overall page sentiment.

Return STRICT JSON:
{"sentiment": "bullish"|"bearish"|"neutral"|"mixed",
 "sentiment_score": float between -1.0 and 1.0,
 "confidence": float between 0.0 and 1.0,
 "evidence": [{"content_id": str, "source_url": str,
              "source_span": str, "reason": str}]}

Every evidence item must quote a real source_span from the inputs - do not
invent quotes or numbers. If there is no clear signal return neutral with
low confidence."""


def _build_fallback_evidence(
    entity: dict, records_by_id: dict[str, dict], reason: str
) -> dict | None:
    """Build ONE evidence item from real source data when the LLM gave us none.

    We never fabricate quotes - we only quote text that actually exists in the
    extracted records. Order of preference:

      1. A ``source_mentions`` entry with a usable ``mention_text`` /
         ``source_span`` that points at a known content record.
      2. The same as above but with the record's title / body snippet as the
         span when the mention itself didn't carry one.
      3. As a last resort, any content record at all - keeps the contract
         "every sentiment has at least one evidence item" intact even when
         entity resolution disagreed with the available content.

    Returns ``None`` only if there is literally no content to point at.
    """
    for mention in entity.get("source_mentions", []) or []:
        cid = mention.get("content_id")
        if not cid:
            continue
        rec = records_by_id.get(cid)
        if not rec:
            continue
        span = (
            mention.get("source_span")
            or mention.get("mention_text")
            or (rec.get("body") or "")[:200]
            or rec.get("title")
        )
        if not span:
            continue
        return {
            "content_id": cid,
            "source_url": mention.get("source_url") or rec.get("source_url"),
            "source_span": span,
            "reason": reason,
        }

    for cid, rec in records_by_id.items():
        span = (rec.get("body") or "")[:200] or rec.get("title")
        if span:
            return {
                "content_id": cid,
                "source_url": rec.get("source_url"),
                "source_span": span,
                "reason": reason + " (no entity mention matched a record)",
            }

    return None


def _validate_evidence_items(
    raw_items: list, records_by_id: dict[str, dict]
) -> list[dict]:
    """Keep only evidence items that have content_id + source_span; backfill the rest.

    The LLM occasionally returns items that are missing ``source_url`` or
    ``reason``. ``source_url`` we can recover from the record; ``reason`` we
    default to a sensible string. Items missing ``content_id`` or
    ``source_span`` are dropped because we can't recover them honestly.
    """
    validated: list[dict] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("content_id")
        span = item.get("source_span")
        if not cid or not span:
            continue
        rec = records_by_id.get(cid) or {}
        source_url = item.get("source_url") or rec.get("source_url")
        if not source_url:
            # Without a source_url we cannot satisfy the validator either.
            continue
        validated.append(
            {
                "content_id": cid,
                "source_url": source_url,
                "source_span": span,
                "reason": item.get("reason") or "no reason provided by model",
            }
        )
    return validated


def score_entity_sentiment(
    entity: dict, records_by_id: dict[str, dict]
) -> dict:
    """Run ONE LLM call per entity to get an entity-specific sentiment.

    Guarantees: the returned record always has a non-empty ``evidence`` list,
    and every evidence item carries ``content_id``, ``source_url``,
    ``source_span``, and ``reason``. If the LLM fails or returns nothing
    usable we synthesise an evidence item from real source mentions rather
    than fabricating quotes.
    """
    relevant_ids = list(
        {m["content_id"] for m in entity["source_mentions"] if m.get("content_id")}
    )
    excerpts = []
    for cid in relevant_ids:
        rec = records_by_id.get(cid)
        if not rec:
            continue
        body = rec.get("body") or ""
        excerpts.append(
            f"---\ncontent_id: {cid}\nsource_url: {rec['source_url']}\n"
            f"title: {rec.get('title','')}\nbody: {body[:1800]}"
        )

    base_record = {
        "entity_id": entity["entity_id"],
        "canonical_name": entity["canonical_name"],
        "sentiment": "neutral",
        "sentiment_score": 0.0,
        "confidence": 0.0,
        "evidence": [],
    }

    if not excerpts:
        # No content to send the LLM - return neutral but still attach evidence
        # built from the entity's own source_mentions so the validator passes.
        fb = _build_fallback_evidence(
            entity,
            records_by_id,
            "No usable excerpts were available; sentiment defaults to neutral",
        )
        if fb:
            base_record["evidence"].append(fb)
        return base_record

    user_prompt = (
        f"Entity:\n  canonical_name: {entity['canonical_name']}\n"
        f"  entity_type: {entity['entity_type']}\n"
        f"  aliases: {entity['aliases']}\n\n"
        f"Source excerpts:\n" + "\n".join(excerpts)
    )[:16000]

    result = call_llm(
        stage_name="ENTITY_SENTIMENT_SCORED",
        system=SENTIMENT_SYSTEM,
        user=user_prompt,
        content_ids=relevant_ids,
        input_artifacts=["entities.json", "extracted_content.json"],
        output_artifact="entity_sentiment.json",
    )
    parsed = result.get("parsed") or {}

    record = {
        "entity_id": entity["entity_id"],
        "canonical_name": entity["canonical_name"],
        "sentiment": parsed.get("sentiment", "neutral"),
        "sentiment_score": float(parsed.get("sentiment_score", 0.0) or 0.0),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "evidence": _validate_evidence_items(
            parsed.get("evidence", []) or [], records_by_id
        ),
    }

    if not record["evidence"]:
        # LLM gave us nothing usable. Attach a real fallback so the contract
        # "every sentiment record has >=1 evidence item with all fields filled"
        # is preserved end-to-end.
        reason = (
            f"LLM error: {result['error']}"
            if result.get("error")
            else "LLM returned no usable evidence items"
        )
        fb = _build_fallback_evidence(entity, records_by_id, reason)
        if fb:
            record["evidence"].append(fb)
            # If the LLM also failed to produce a meaningful score, mark this
            # explicitly low confidence so downstream readers know.
            if record["confidence"] == 0.0:
                record["confidence"] = 0.1

    return record


# ---------------------------------------------------------------------------
# Stage 6 - QA + conflict detection
# ---------------------------------------------------------------------------

QA_SYSTEM = """You are a QA reviewer for a financial intelligence pipeline. Look at
the supplied entities, per-entity sentiment, and numerical data. Flag:

- conflicting_sentiment: same entity pulled in opposite directions across sources
- unresolved_entity: entities the upstream system marked low-confidence
- numerical_conflict: same metric / price disagrees across sources
- ungrounded_claim: sentiment evidence that does not quote a real source span
- duplicate_content: same article appears twice

Return STRICT JSON:
{"issues": [
  {"severity": "critical"|"warning"|"info",
   "issue_type": str,
   "entities": [str],
   "source_content_ids": [str],
   "details": str}
]}"""


def run_qa(
    records: list[dict],
    entities: list[dict],
    sentiments: list[dict],
) -> list[dict]:
    """Pipeline-level QA. Combines deterministic checks with one LLM pass."""
    issues: list[dict] = []
    next_id = 0

    def add(severity, issue_type, entities_, source_ids, details):
        nonlocal next_id
        next_id += 1
        issues.append(
            {
                "issue_id": f"q_{next_id:03d}",
                "severity": severity,
                "issue_type": issue_type,
                "entities": entities_,
                "source_content_ids": source_ids,
                "details": details,
            }
        )

    # Deterministic check: low-confidence entities.
    for ent in entities:
        if ent.get("low_confidence_flag"):
            add(
                "warning",
                "unresolved_entity",
                [ent["canonical_name"]],
                [m.get("content_id") for m in ent.get("source_mentions", [])],
                f"resolution_confidence={ent['resolution_confidence']:.2f}",
            )

    # Deterministic check: duplicate content.
    for r in records:
        if r.get("duplicate_of"):
            add(
                "info",
                "duplicate_content",
                [],
                [r["content_id"], r["duplicate_of"]],
                f"{r['content_id']} duplicates {r['duplicate_of']}",
            )

    # Deterministic check: same entity, opposite directions.
    by_entity: dict[str, list[dict]] = {}
    for s in sentiments:
        by_entity.setdefault(s["canonical_name"], []).append(s)
    for name, lst in by_entity.items():
        scores = [s["sentiment_score"] for s in lst]
        if scores and max(scores) > 0.3 and min(scores) < -0.3:
            add(
                "critical",
                "conflicting_sentiment",
                [name],
                [
                    ev.get("content_id")
                    for s in lst
                    for ev in s.get("evidence", [])
                    if ev.get("content_id")
                ],
                f"sentiment scores range from {min(scores):.2f} to {max(scores):.2f}",
            )

    # LLM pass for things rules miss (ungrounded claims, subtle conflicts).
    payload = {
        "entities": [
            {
                "canonical_name": e["canonical_name"],
                "entity_type": e["entity_type"],
                "aliases": e["aliases"],
                "resolution_confidence": e["resolution_confidence"],
            }
            for e in entities
        ],
        "sentiments": sentiments,
        "numerical_data": [
            {
                "content_id": r["content_id"],
                "source_url": r["source_url"],
                "values": r.get("numerical_data", []),
            }
            for r in records
            if r.get("numerical_data")
        ],
    }
    result = call_llm(
        stage_name="QA_AND_CONFLICTS_CHECKED",
        system=QA_SYSTEM,
        user=json.dumps(payload)[:16000],
        input_artifacts=[
            "entities.json",
            "entity_sentiment.json",
            "extracted_content.json",
        ],
        output_artifact="qa_report.json",
    )
    parsed = result.get("parsed") or {}
    for raw in parsed.get("issues", []) or []:
        add(
            raw.get("severity", "info"),
            raw.get("issue_type", "other"),
            raw.get("entities", []) or [],
            raw.get("source_content_ids", []) or [],
            raw.get("details", ""),
        )

    return issues


# ---------------------------------------------------------------------------
# Stage 7 - Reports
# ---------------------------------------------------------------------------

REPORTS_DISCLAIMER = (
    "DISCLAIMER: This is market intelligence only. It is NOT financial advice. "
    "All figures are scraped from public sources and may be stale or wrong."
)


def _report_payload(records, entities, sentiments, qa_issues) -> str:
    """Compact payload string the report-writer LLM can ground itself in."""
    return json.dumps(
        {
            "entities": entities,
            "sentiments": sentiments,
            "qa_issues": qa_issues,
            "numerical_signals": [
                {
                    "content_id": r["content_id"],
                    "source_url": r["source_url"],
                    "title": r["title"],
                    "values": r.get("numerical_data", []),
                }
                for r in records
                if r.get("numerical_data")
            ],
            "headlines": [
                {"content_id": r["content_id"], "source_url": r["source_url"], "title": r["title"]}
                for r in records
                if r.get("content_type") == "headline"
            ][:25],
        },
        ensure_ascii=False,
    )[:16000]


def generate_reports(records, entities, sentiments, qa_issues) -> dict:
    """Generate trader / analyst / executive reports grounded in extracted content."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base_payload = _report_payload(records, entities, sentiments, qa_issues)
    outputs: dict[str, str] = {}

    specs = [
        (
            "trader_brief.md",
            "Trader Brief",
            "You are writing a TRADER BRIEF. Be concise and action-oriented. Highlight "
            "price levels, momentum, and immediate trade-relevant signals. Use bullets. "
            "Cite source_url for each claim. Mark output as market intelligence only, "
            "not financial advice. Only use facts present in the supplied JSON.",
        ),
        (
            "analyst_report.md",
            "Analyst Report",
            "You are writing a DETAILED ANALYST REPORT. Show cross-source evidence, "
            "confidence scores, and contradictions. Group by entity. Cite source_url for "
            "every claim. Mark output as market intelligence only, not financial advice. "
            "Only use facts present in the supplied JSON.",
        ),
        (
            "executive_summary.md",
            "Executive Summary",
            "You are writing an EXECUTIVE SUMMARY for senior leadership. Emphasise "
            "macro trends and risk. Keep it short, plain language. Cite source_url for "
            "each claim. Mark output as market intelligence only, not financial advice. "
            "Only use facts present in the supplied JSON.",
        ),
    ]

    for filename, title, system_prompt in specs:
        result = call_llm(
            stage_name="REPORTS_GENERATED",
            system=system_prompt + " Return Markdown text only, no JSON.",
            user="Source data JSON:\n" + base_payload,
            input_artifacts=[
                "entities.json",
                "entity_sentiment.json",
                "qa_report.json",
            ],
            output_artifact=filename,
            json_mode=False,
            temperature=0.2,
        )
        body = (result.get("raw") or "").strip()
        if not body:
            body = f"# {title}\n\n_No content was generated (LLM error: {result.get('error')})._\n"

        full = f"# {title}\n\n_{REPORTS_DISCLAIMER}_\n\n{body}\n"
        path = REPORTS_DIR / filename
        path.write_text(full, encoding="utf-8")
        outputs[filename] = str(path)

    return outputs


# ---------------------------------------------------------------------------
# Stage 8 - Cost report + run metrics
# ---------------------------------------------------------------------------


def build_cost_report(records: list[dict], entities: list[dict]) -> dict:
    """Aggregate token usage from llm_calls and translate to estimated USD."""
    calls = RUN_STATE["llm_calls"]
    total_in = sum(c["estimated_input_tokens"] for c in calls)
    total_out = sum(c["estimated_output_tokens"] for c in calls)

    in_cost = total_in * PRICE_INPUT_PER_M / 1_000_000
    out_cost = total_out * PRICE_OUTPUT_PER_M / 1_000_000

    by_stage: dict[str, dict] = {}
    for c in calls:
        s = c["stage"]
        d = by_stage.setdefault(
            s, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
        )
        d["calls"] += 1
        d["input_tokens"] += c["estimated_input_tokens"]
        d["output_tokens"] += c["estimated_output_tokens"]
        if c.get("error"):
            d["errors"] += 1

    sources_seen = sorted({r["source_url"] for r in records})
    per_source_cost = (in_cost + out_cost) / max(1, len(sources_seen))
    per_entity_cost = (in_cost + out_cost) / max(1, len(entities))

    deduped = sum(1 for r in records if r.get("duplicate_of"))

    return {
        "model": MODEL,
        "provider": "groq",
        "price_input_per_million_usd": PRICE_INPUT_PER_M,
        "price_output_per_million_usd": PRICE_OUTPUT_PER_M,
        "total_llm_calls": len(calls),
        "total_estimated_input_tokens": total_in,
        "total_estimated_output_tokens": total_out,
        "estimated_input_cost_usd": round(in_cost, 6),
        "estimated_output_cost_usd": round(out_cost, 6),
        "estimated_total_cost_usd": round(in_cost + out_cost, 6),
        "estimated_cost_per_source_usd": round(per_source_cost, 6),
        "estimated_cost_per_entity_usd": round(per_entity_cost, 6),
        "by_stage": by_stage,
        "deduplicated_content_count": deduped,
    }


def build_run_metrics(
    records: list[dict], entities: list[dict], sources: list[str]
) -> dict:
    """Latency, error rate, and confidence distribution for the run."""
    confidence_buckets = {"0.0-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for ent in entities:
        c = ent["resolution_confidence"]
        if c < 0.6:
            confidence_buckets["0.0-0.6"] += 1
        elif c < 0.8:
            confidence_buckets["0.6-0.8"] += 1
        else:
            confidence_buckets["0.8-1.0"] += 1

    extraction_errors = RUN_STATE["extraction_errors"]
    error_rate = (
        round(len(extraction_errors) / len(sources), 3) if sources else 0.0
    )

    return {
        "started_at": RUN_STATE["started_at"],
        "finished_at": RUN_STATE["finished_at"],
        "stage_timings": RUN_STATE["stage_timings"],
        "source_latency_seconds": RUN_STATE["source_latency"],
        "extraction_errors": extraction_errors,
        "extraction_error_rate": error_rate,
        "sources_attempted": len(sources),
        "sources_succeeded": len(sources) - len(extraction_errors),
        "content_records": len(records),
        "entities_total": len(entities),
        "entity_resolution_confidence_distribution": confidence_buckets,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all eleven stages of the pipeline."""
    RUN_STATE["started_at"] = now_iso()

    # Reset llm_calls.jsonl so each run starts clean.
    if LLM_LOG_PATH.exists():
        LLM_LOG_PATH.unlink()

    # Reset failed_sources.json too - it's a per-run record, not cumulative.
    failed_path = ROOT / FAILED_SOURCES_PATH
    if failed_path.exists():
        failed_path.unlink()

    stage("INIT")
    if not os.getenv("GROQ_API_KEY"):
        print("  [ERROR] GROQ_API_KEY is not set. Copy .env.example to .env first.")
        return 2
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stage_done("INIT")

    stage("SOURCES_LOADED")
    try:
        sources = load_sources()
    except Exception as exc:
        print(f"  [ERROR] could not load sources.json: {exc}")
        return 2
    print(f"  loaded {len(sources)} source(s)")
    stage_done("SOURCES_LOADED")

    stage("CONTENT_FETCHED")
    raw_records = fetch_and_extract(sources)
    stage_done("CONTENT_FETCHED")

    stage("CONTENT_EXTRACTED")
    print(f"  total content records: {len(raw_records)}")
    stage_done("CONTENT_EXTRACTED")

    stage("CONTENT_NORMALISED")
    records = normalise_content(raw_records)
    safe_write_json(ROOT / "extracted_content.json", records)
    print(f"  normalised records written: {len(records)}")
    stage_done("CONTENT_NORMALISED")

    stage("ENTITIES_EXTRACTED")
    entities = extract_entities(records)
    print(f"  extracted entities: {len(entities)}")
    stage_done("ENTITIES_EXTRACTED")

    stage("ENTITIES_RESOLVED")
    # Resolution happens inside the same LLM call; this stage is a logical
    # checkpoint where we persist the resolved entities to disk.
    safe_write_json(ROOT / "entities.json", entities)
    print(
        f"  resolved entities saved (low-confidence: "
        f"{sum(1 for e in entities if e.get('low_confidence_flag'))})"
    )
    stage_done("ENTITIES_RESOLVED")

    stage("ENTITY_SENTIMENT_SCORED")
    records_by_id = {r["content_id"]: r for r in records}
    sentiments: list[dict] = []
    for ent in entities:
        try:
            record = score_entity_sentiment(ent, records_by_id)
        except Exception as exc:
            print(f"  [WARN] sentiment failed for {ent['canonical_name']}: {exc}")
            record = {
                "entity_id": ent["entity_id"],
                "canonical_name": ent["canonical_name"],
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "evidence": [],
                "error": str(exc),
            }
            fb = _build_fallback_evidence(
                ent, records_by_id, f"Sentiment scoring crashed: {exc}"
            )
            if fb:
                record["evidence"].append(fb)

        # Final guard: if a sentiment record somehow still has no evidence
        # (e.g. there were no scrapeable records at all), drop it instead of
        # writing an empty evidence list. The validator requires every
        # persisted record to carry at least one evidence item.
        if not record.get("evidence"):
            print(
                f"  [WARN] no evidence available for {ent['canonical_name']}; "
                "skipping sentiment record"
            )
            continue

        sentiments.append(record)

    safe_write_json(ROOT / "entity_sentiment.json", sentiments)
    stage_done("ENTITY_SENTIMENT_SCORED")

    stage("QA_AND_CONFLICTS_CHECKED")
    qa_issues = run_qa(records, entities, sentiments)
    safe_write_json(ROOT / "qa_report.json", qa_issues)
    print(f"  qa issues: {len(qa_issues)}")
    stage_done("QA_AND_CONFLICTS_CHECKED")

    stage("REPORTS_GENERATED")
    report_paths = generate_reports(records, entities, sentiments, qa_issues)
    print(f"  reports written: {list(report_paths.keys())}")
    stage_done("REPORTS_GENERATED")

    stage("RESULTS_FINALISED")
    RUN_STATE["finished_at"] = now_iso()
    cost = build_cost_report(records, entities)
    metrics = build_run_metrics(records, entities, sources)
    safe_write_json(ROOT / "cost_report.json", cost)
    safe_write_json(ROOT / "run_metrics.json", metrics)
    print(
        f"  done. estimated total cost: ${cost['estimated_total_cost_usd']:.6f} "
        f"across {cost['total_llm_calls']} LLM call(s)"
    )
    stage_done("RESULTS_FINALISED")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
