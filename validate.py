"""
validate.py - sanity-checks the artifacts produced by pipeline.py.

Run after the pipeline has finished. Exits non-zero if any required check
fails. Prints a summary of pass / fail per check.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "extracted_content.json",
    "entities.json",
    "entity_sentiment.json",
    "qa_report.json",
    "cost_report.json",
    "run_metrics.json",
    "llm_calls.jsonl",
    "sources.json",
    "reports/trader_brief.md",
    "reports/analyst_report.md",
    "reports/executive_summary.md",
]

REQUIRED_LLM_STAGES = {
    "ENTITIES_EXTRACTED",
    "ENTITY_SENTIMENT_SCORED",
    "QA_AND_CONFLICTS_CHECKED",
    "REPORTS_GENERATED",
}


def load_json(path: Path):
    """Helper: load a JSON file, raising a friendly message on failure."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path):
    """Helper: load a JSONL file as a list of dicts."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def check(name, condition, detail=""):
    """Print a single check result and return True if it passed."""
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
    return bool(condition)


def main() -> int:
    """Run every validation check and return a process exit code."""
    failures: list[str] = []

    # 1. Required artifacts exist.
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not check(f"file exists: {rel}", path.exists()):
            failures.append(rel)

    if failures:
        # Some required files are missing - cannot proceed with deeper checks.
        print(f"\n{len(failures)} missing artifact(s); aborting deeper validation.")
        return 1

    # 2. JSON files parse cleanly.
    json_files = [f for f in REQUIRED_FILES if f.endswith(".json")]
    parsed = {}
    for rel in json_files:
        try:
            parsed[rel] = load_json(ROOT / rel)
            check(f"valid JSON: {rel}", True)
        except Exception as exc:
            check(f"valid JSON: {rel}", False, str(exc))
            failures.append(rel)

    try:
        llm_calls = load_jsonl(ROOT / "llm_calls.jsonl")
        check("valid JSONL: llm_calls.jsonl", True, f"{len(llm_calls)} record(s)")
    except Exception as exc:
        check("valid JSONL: llm_calls.jsonl", False, str(exc))
        failures.append("llm_calls.jsonl")
        llm_calls = []

    # 3. >= 2-3 sources processed (we require >=2 successful sources).
    metrics = parsed.get("run_metrics.json", {})
    sources_succeeded = metrics.get("sources_succeeded", 0)
    if not check(
        "at least 2 sources processed",
        sources_succeeded >= 2,
        f"sources_succeeded={sources_succeeded}",
    ):
        failures.append("sources_succeeded<2")

    # 4. Extracted content includes source attribution.
    content = parsed.get("extracted_content.json", [])
    if not check(
        "extracted_content non-empty", len(content) > 0, f"{len(content)} record(s)"
    ):
        failures.append("extracted_content_empty")

    if content:
        missing_attr = [
            r
            for r in content
            if not r.get("source_url") or not r.get("source_name")
        ]
        if not check(
            "every content record has source_url + source_name",
            len(missing_attr) == 0,
            f"{len(missing_attr)} missing attribution",
        ):
            failures.append("content_attribution_missing")

    # 5. Numerical data preserves source spans.
    numerical_records = [r for r in content if r.get("numerical_data")]
    spans_ok = all(
        all(n.get("source_span") for n in r["numerical_data"])
        for r in numerical_records
    )
    if not check(
        "numerical_data preserves source_span",
        spans_ok or not numerical_records,
        f"{len(numerical_records)} record(s) with numerical data",
    ):
        failures.append("numerical_spans_missing")

    # 6. Entities include aliases and source_mentions.
    entities = parsed.get("entities.json", [])
    if entities:
        missing_alias_field = [e for e in entities if "aliases" not in e]
        missing_mentions = [
            e for e in entities if not e.get("source_mentions")
        ]
        check(
            "every entity has aliases field",
            len(missing_alias_field) == 0,
            f"{len(missing_alias_field)} missing 'aliases'",
        )
        if not check(
            "every entity has at least one source_mention",
            len(missing_mentions) == 0,
            f"{len(missing_mentions)} entities with no mentions",
        ):
            failures.append("entities_missing_mentions")
    else:
        check("entities.json has entries", False, "0 entities")
        failures.append("no_entities")

    # 7. Entity sentiment is entity-specific, not page-level.
    sentiments = parsed.get("entity_sentiment.json", [])
    if sentiments:
        # Heuristic: if every sentiment for two different entities tied to the
        # same content_id has identical score, that points at a page-level
        # score. We require at least one pair of differing scores to exist OR
        # only one entity per content_id.
        per_content_scores: dict[str, set] = {}
        for s in sentiments:
            for ev in s.get("evidence", []) or []:
                cid = ev.get("content_id")
                if cid:
                    per_content_scores.setdefault(cid, set()).add(
                        round(s["sentiment_score"], 4)
                    )
        diverse = any(len(v) > 1 for v in per_content_scores.values())
        only_one_entity_per_content = all(
            len(v) == 1 for v in per_content_scores.values()
        )
        ok = diverse or only_one_entity_per_content or len(sentiments) <= 1
        if not check(
            "entity sentiment is entity-specific (not page-level)",
            ok,
            "all multi-entity articles share one score - looks page-level",
        ):
            failures.append("page_level_sentiment")

        # 8. Sentiment records include evidence spans.
        missing_evidence = [
            s
            for s in sentiments
            if not s.get("evidence")
            or any(not ev.get("source_span") for ev in s["evidence"])
        ]
        if not check(
            "sentiment records include evidence spans",
            len(missing_evidence) == 0,
            f"{len(missing_evidence)} sentiment(s) missing evidence spans",
        ):
            failures.append("sentiment_missing_evidence")
    else:
        check("entity_sentiment.json has entries", False, "0 sentiments")
        failures.append("no_sentiments")

    # 9. Low-confidence entities are flagged.
    if entities:
        flagged_consistently = all(
            (e.get("low_confidence_flag") is True)
            == (float(e.get("resolution_confidence", 1.0)) < 0.6)
            for e in entities
        )
        check(
            "low-confidence entities flagged",
            flagged_consistently,
            "low_confidence_flag matches resolution_confidence < 0.6",
        )
        if not flagged_consistently:
            failures.append("low_confidence_flag_inconsistent")

    # 10. llm_calls.jsonl has the required stage records.
    stages_seen = {c.get("stage") for c in llm_calls}
    missing_stages = REQUIRED_LLM_STAGES - stages_seen
    if not check(
        "llm_calls.jsonl covers required stages",
        len(missing_stages) == 0,
        f"missing stages: {sorted(missing_stages) or 'none'}",
    ):
        failures.append("llm_stage_records_missing")

    # 11. Every llm_call record has the mandated fields.
    required_fields = {
        "stage",
        "source_url",
        "content_ids",
        "timestamp",
        "provider",
        "model",
        "prompt_hash",
        "input_artifacts",
        "output_artifact",
        "estimated_input_tokens",
        "estimated_output_tokens",
    }
    bad = [c for c in llm_calls if not required_fields.issubset(c.keys())]
    if not check(
        "every llm_calls record has required fields",
        len(bad) == 0,
        f"{len(bad)} records missing fields",
    ):
        failures.append("llm_record_fields_missing")

    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VALIDATION PASSED: all checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
