import networkx as nx

from .db import neo4j

# ------------------------------------------------------------
# Edge construction
# ------------------------------------------------------------
# BEFORE edges come from two independent sources:
#   (a) the deterministic Pass-3 ordering  -> high confidence, consecutive chain
#   (b) causal links inferred by the LLM   -> lower confidence, may contradict (a)
# Contradictions create cycles. We locate them with Tarjan's SCC and drop the
# lowest-confidence edge in each cycle until the graph is a DAG.
CONF_SEQUENTIAL = 0.3
CONF_CAUSAL = 0.9

def _causal_pairs(events: list[dict]) -> list[tuple[str, str, float]]:
    """Use vector similarity + concurrent LLM to infer causal links."""
    from . import llm
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    candidates = []
    for a in events:
        for b in events:
            if a["id"] == b["id"]:
                continue
            
            sim = cosine_sim(a["embedding"], b["embedding"])
            if sim > 0.78:  # Higher threshold to prevent 900+ calls
                candidates.append((a, b))

    pairs = []
    
    def check_causal(a, b):
        prompt = f"""Does Event A directly cause Event B?
Event A: {a['event_name']} - {a['core_event']}
Event B: {b['event_name']} - {b['core_event']}
Output only YES or NO."""
        try:
            resp = llm.chat("You are a causal reasoning engine.", prompt, max_tokens=10).strip().upper()
            if "YES" in resp:
                return (a["id"], b["id"], CONF_CAUSAL)
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(check_causal, a, b) for a, b in candidates]
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                pairs.append(res)
                
    return pairs

def build_edges(events: list[dict]) -> tuple[list[dict], dict, list[str]]:
    """Return (edges, repair_stats, topo_order)."""
    candidates: list[tuple[str, str, float, str]] = []

    ordered = sorted(events, key=lambda e: e["first_page"])
    for a, b in zip(ordered, ordered[1:]):
        candidates.append((a["id"], b["id"], CONF_SEQUENTIAL, "sequence"))

    for src, dst, conf in _causal_pairs(events):
        candidates.append((src, dst, conf, "causal"))

    g = nx.DiGraph()
    g.add_nodes_from(e["id"] for e in events)
    for src, dst, conf, kind in candidates:
        if g.has_edge(src, dst):
            if conf > g[src][dst]["confidence"]:
                g[src][dst].update(confidence=conf, kind=kind)
        else:
            g.add_edge(src, dst, confidence=conf, kind=kind)

    dropped = 0
    guard = 0
    while guard < 1000:
        guard += 1
        sccs = [c for c in nx.strongly_connected_components(g) if len(c) > 1]
        if not sccs:
            break
        for comp in sccs:
            inner = [(u, v, g[u][v]["confidence"])
                     for u in comp for v in g.successors(u) if v in comp]
            if not inner:
                continue
            u, v, _ = min(inner, key=lambda t: t[2])
            g.remove_edge(u, v)
            dropped += 1

    g.remove_edges_from(nx.selfloop_edges(g))
    topo_order = list(nx.topological_sort(g))

    edges = [
        {"src": u, "dst": v, "confidence": d["confidence"], "kind": d["kind"]}
        for u, v, d in g.edges(data=True)
    ]
    return edges, {"cycles_repaired": dropped}, topo_order
def push(doc_id: str, events: list[dict], edges: list[dict]) -> None:
    drv = neo4j()
    with drv.session() as sess:
        sess.run(
            "MATCH (e:Event {doc_id:$doc}) DETACH DELETE e", doc=doc_id
        )
        sess.run(
            "MATCH (c:Character {doc_id:$doc}) DETACH DELETE c", doc=doc_id
        )
        sess.run(
            """UNWIND $rows AS r
               CREATE (e:Event {
                 id: r.id, doc_id: $doc, name: r.event_name, category: r.category,
                 anchor: r.chronological_clue, stage_order: r.topological_order,
                 location: r.location, core: r.core_event,
                 cause: r.antecedent_cause, effect: r.consequent_effect,
                 pages: r.source_pages, first_page: r.first_page
               })""",
            rows=events, doc=doc_id,
        )
        sess.run(
            """UNWIND $rows AS r
               UNWIND r.characters AS cname
               MERGE (c:Character {uid: $doc + '::' + cname})
               SET c.doc_id = $doc, c.name = cname
               WITH c, r
               MATCH (e:Event {id: r.id})
               MERGE (c)-[:APPEARS_IN]->(e)""",
            rows=events, doc=doc_id,
        )
        sess.run(
            """UNWIND $rows AS r
               MATCH (a:Event {id: r.src}), (b:Event {id: r.dst})
               MERGE (a)-[rel:BEFORE]->(b)
               SET rel.confidence = r.confidence, rel.kind = r.kind""",
            rows=edges,
        )


def fetch_graph(doc_id: str) -> dict:
    """Nodes + edges for the frontend graph view."""
    with neo4j().session() as sess:
        nodes = sess.run(
            """MATCH (e:Event {doc_id:$doc})
               RETURN e.id AS id, e.name AS name, e.category AS category,
                      e.anchor AS anchor, e.stage_order AS stage_order,
                      e.first_page AS first_page, e.pages AS pages, e.core AS core
               ORDER BY e.stage_order, e.first_page""",
            doc=doc_id,
        ).data()
        edges = sess.run(
            """MATCH (a:Event {doc_id:$doc})-[r:BEFORE]->(b:Event {doc_id:$doc})
               RETURN a.id AS src, b.id AS dst, r.confidence AS confidence,
                      r.kind AS kind""",
            doc=doc_id,
        ).data()
    return {"nodes": nodes, "edges": edges}


def reachable(src_id: str, dst_id: str) -> dict:
    """Deterministic ordering lookup: is src before dst in the DAG?"""
    with neo4j().session() as sess:
        rec = sess.run(
            """MATCH (a:Event {id:$a}), (b:Event {id:$b})
               OPTIONAL MATCH p = shortestPath((a)-[:BEFORE*1..80]->(b))
               RETURN p IS NOT NULL AS forward,
                      CASE WHEN p IS NULL THEN [] ELSE [n IN nodes(p) | n.id] END AS chain""",
            a=src_id, b=dst_id,
        ).single()
        if rec and rec["forward"]:
            return {"relation": "before", "chain": rec["chain"]}
        rec2 = sess.run(
            """MATCH (a:Event {id:$a}), (b:Event {id:$b})
               OPTIONAL MATCH p = shortestPath((b)-[:BEFORE*1..80]->(a))
               RETURN p IS NOT NULL AS backward,
                      CASE WHEN p IS NULL THEN [] ELSE [n IN nodes(p) | n.id] END AS chain""",
            a=src_id, b=dst_id,
        ).single()
        if rec2 and rec2["backward"]:
            return {"relation": "after", "chain": rec2["chain"]}
    return {"relation": "incomparable", "chain": []}


def neighbours(event_ids: list[str], hops: int = 1) -> list[str]:
    """Expand a seed set along BEFORE edges in both directions."""
    with neo4j().session() as sess:
        rows = sess.run(
            f"""MATCH (e:Event) WHERE e.id IN $ids
                MATCH (n:Event)
                WHERE (n)-[:BEFORE*1..{hops}]->(e) OR (e)-[:BEFORE*1..{hops}]->(n)
                RETURN DISTINCT n.id AS id""",
            ids=event_ids,
        ).data()
    return [r["id"] for r in rows]
