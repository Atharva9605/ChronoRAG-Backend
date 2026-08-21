from itertools import combinations

from .db import pg
from .passes import load_events


def _pairwise_accuracy(events: list[dict], gold: list[str]) -> float | None:
    """
    gold: ordered list of event_names representing ground truth chronology.
    Score = fraction of gold pairs the system also orders correctly.
    """
    if len(gold) < 2:
        return None
    pos = {e["event_name"]: i for i, e in enumerate(events)}
    usable = [g for g in gold if g in pos]
    if len(usable) < 2:
        return None
    total = correct = 0
    for a, b in combinations(usable, 2):
        if gold.index(a) < gold.index(b):
            total += 1
            if pos[a] < pos[b]:
                correct += 1
    return round(correct / total, 4) if total else None


def _kendall_tau(events: list[dict], gold: list[str]) -> float | None:
    """Kendall's tau between system order and gold order (Lapata, 2006)."""
    pos = {e["event_name"]: i for i, e in enumerate(events)}
    usable = [g for g in gold if g in pos]
    n = len(usable)
    if n < 2:
        return None
    conc = disc = 0
    for a, b in combinations(usable, 2):
        gold_dir = usable.index(a) - usable.index(b)
        sys_dir = pos[a] - pos[b]
        if gold_dir * sys_dir > 0:
            conc += 1
        else:
            disc += 1
    return round((conc - disc) / (0.5 * n * (n - 1)), 4)


def summarise(doc_id: str, gold: list[str] | None = None) -> dict:
    events = load_events(doc_id)

    with pg() as cur:
        cur.execute("SELECT count(*) AS c FROM naive_chunks WHERE doc_id = %s", (doc_id,))
        chunks = cur.fetchone()["c"]
        cur.execute("SELECT count(*) AS c FROM observations WHERE doc_id = %s", (doc_id,))
        obs = cur.fetchone()["c"]
        cur.execute(
            """SELECT avg(naive_ms) AS n_ms, avg(kaal_ms) AS k_ms,
                      avg(naive_tokens) AS n_tok, avg(kaal_tokens) AS k_tok,
                      count(*) AS runs
               FROM query_runs WHERE doc_id = %s""",
            (doc_id,),
        )
        q = cur.fetchone()

    total_pages_cited = sum(len(e["source_pages"]) for e in events)
    merges = sum(e["merge_count"] - 1 for e in events)
    raw_records = len(events) + merges

    return {
        "events_total": len(events),
        "events_major": sum(1 for e in events if e["category"] == "major"),
        "events_minor": sum(1 for e in events if e["category"] != "major"),
        "raw_records_before_merge": raw_records,
        # duplicate rate = share of raw records that were folded into an existing event
        "duplicate_rate": round(merges / raw_records, 4) if raw_records else 0.0,
        "page_traceability": round(
            sum(1 for e in events if e["source_pages"]) / max(1, len(events)), 4
        ),
        "avg_pages_per_event": round(total_pages_cited / max(1, len(events)), 2),
        "naive_chunks": chunks,
        "pass1_windows": obs,
        "stage_distribution": {
            s: sum(1 for e in events if e["timeline_anchor"] == s)
            for s in {e["timeline_anchor"] for e in events}
        },
        "query_runs": q["runs"] or 0,
        "avg_naive_ms": round(q["n_ms"] or 0),
        "avg_kaalkram_ms": round(q["k_ms"] or 0),
        "avg_naive_tokens": round(q["n_tok"] or 0),
        "avg_kaalkram_tokens": round(q["k_tok"] or 0),
        "pairwise_accuracy": _pairwise_accuracy(events, gold or []),
        "kendall_tau": _kendall_tau(events, gold or []),
    }
