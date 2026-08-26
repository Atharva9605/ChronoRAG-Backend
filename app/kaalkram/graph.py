from collections import defaultdict, deque
import logging
from ..db import neo4j

logger = logging.getLogger(__name__)


def build_timeline_graph(events: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Constructs directional edges based on sequential flow, preceding event references,
    and flashback logic. Computes topological ordering using Kahn's algorithm.
    """
    edges: list[dict] = []
    id_map = {e["id"]: e for e in events}
    name_map = {e["event_name"].lower().strip(): e["id"] for e in events}
    
    # 1. Connect sequential story_progression events
    story_events = [e for e in events if e["classification"] == "story_progression"]
    for i in range(len(story_events) - 1):
        edges.append({
            "src": story_events[i]["id"],
            "dst": story_events[i + 1]["id"],
            "rel": "HAPPENS_BEFORE",
        })

    # 2. Connect explicit preceding references (strict word matching)
    for e in events:
        ref = (e.get("preceding_event_reference") or "").lower().strip()
        if ref and ref not in ("none", "n/a", "null") and len(ref) > 8:
            best_match_id = None
            best_overlap = 0
            ref_words = set(ref.split()) - {"the", "a", "an", "and", "of", "to", "in", "he", "she", "they"}
            
            for name, prior_id in name_map.items():
                if prior_id == e["id"]:
                    continue
                name_words = set(name.split()) - {"the", "a", "an", "and", "of", "to", "in", "he", "she", "they"}
                overlap = len(ref_words & name_words)
                if overlap >= 2 and overlap > best_overlap:
                    best_overlap = overlap
                    best_match_id = prior_id
            
            if best_match_id:
                edges.append({
                    "src": best_match_id,
                    "dst": e["id"],
                    "rel": "CAUSES",
                })

    # 3. Handle Flashbacks: Flashbacks occurred chronologically BEFORE forward narrative
    flashbacks = [e for e in events if e["classification"] in ("flashback", "backstory")]
    if story_events and flashbacks:
        first_story_event = story_events[0]
        for fb in flashbacks:
            edges.append({
                "src": fb["id"],
                "dst": first_story_event["id"],
                "rel": "HAPPENS_BEFORE",
            })

    # Deduplicate edges
    seen_edges = set()
    raw_edges = []
    for edge in edges:
        key = (edge["src"], edge["dst"])
        if key not in seen_edges and edge["src"] != edge["dst"]:
            seen_edges.add(key)
            raw_edges.append(edge)

    # Transitive Reduction: Remove direct edge (u -> v) if there is an alternate path from u to v
    adj_graph = defaultdict(set)
    for edge in raw_edges:
        adj_graph[edge["src"]].add(edge["dst"])

    def has_alternate_path(start: str, target: str) -> bool:
        queue = deque([nbr for nbr in adj_graph[start] if nbr != target])
        visited = set(queue)
        while queue:
            curr = queue.popleft()
            if curr == target:
                return True
            for nxt in adj_graph[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    unique_edges = []
    for edge in raw_edges:
        src, dst = edge["src"], edge["dst"]
        if not has_alternate_path(src, dst):
            unique_edges.append(edge)

    # 4. Topological Sort (Kahn's Algorithm with cycle resolution)
    adj = defaultdict(list)
    in_degree = {e["id"]: 0 for e in events}
    
    for edge in unique_edges:
        src, dst = edge["src"], edge["dst"]
        if src in in_degree and dst in in_degree:
            adj[src].append(dst)
            in_degree[dst] += 1

    queue = deque([node for node, deg in in_degree.items() if deg == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle remaining nodes in case of circular dependencies
    remaining = [node for node in in_degree if node not in set(topo_order)]
    remaining.sort(key=lambda nid: (id_map[nid]["page_start"], id_map[nid]["id"]))
    topo_order.extend(remaining)

    return unique_edges, topo_order


def push_to_neo4j(doc_id: str, events: list[dict], edges: list[dict]) -> None:
    """Pushes nodes and relationships into Neo4j graph database."""
    with neo4j().session() as session:
        # Clear existing nodes for document
        session.run("MATCH (n:Event {doc_id: $d}) DETACH DELETE n", d=doc_id)

        # Batch create Event nodes
        node_payload = [
            {
                "id": e["id"],
                "doc_id": doc_id,
                "name": e["event_name"],
                "summary": e["summary"],
                "classification": e["classification"],
                "page_start": e["page_start"],
                "page_end": e["page_end"],
                "source_pages": e["page_numbers"],
                "characters": e["characters"],
                "temporal_anchor": e["temporal_anchor"],
                "topological_order": e.get("topological_order", 0),
            }
            for e in events
        ]

        session.run(
            """
            UNWIND $nodes AS n
            CREATE (e:Event {
                id: n.id,
                doc_id: n.doc_id,
                name: n.name,
                summary: n.summary,
                classification: n.classification,
                page_start: n.page_start,
                page_end: n.page_end,
                source_pages: n.source_pages,
                characters: n.characters,
                temporal_anchor: n.temporal_anchor,
                topological_order: n.topological_order
            })
            """,
            nodes=node_payload,
        )

        # Batch create directional edges
        if edges:
            session.run(
                """
                UNWIND $edges AS r
                MATCH (a:Event {id: r.src}), (b:Event {id: r.dst})
                MERGE (a)-[rel:HAPPENS_BEFORE {relation_type: r.rel}]->(b)
                """,
                edges=edges,
            )


def fetch_graph(doc_id: str) -> dict:
    """Retrieves full graph representation for UI visualization."""
    with neo4j().session() as session:
        result = session.run(
            """
            MATCH (e:Event {doc_id: $d})
            OPTIONAL MATCH (e)-[r]->(target:Event {doc_id: $d})
            RETURN e, r, target
            ORDER BY e.topological_order, e.page_start
            """,
            d=doc_id,
        )
        
        nodes_dict = {}
        links = []
        for record in result:
            n = record["e"]
            if n["id"] not in nodes_dict:
                nodes_dict[n["id"]] = {
                    "id": n["id"],
                    "name": n.get("name", "Event"),
                    "summary": n.get("summary", ""),
                    "classification": n.get("classification", "story_progression"),
                    "pages": n.get("source_pages", []),
                    "page_start": n.get("page_start", 0),
                    "page_end": n.get("page_end", 0),
                    "topological_order": n.get("topological_order", 0),
                    "temporal_anchor": n.get("temporal_anchor", ""),
                }
            r = record["r"]
            t = record["target"]
            if r and t:
                links.append({
                    "source": n["id"],
                    "target": t["id"],
                    "relation": r.get("relation_type", "HAPPENS_BEFORE"),
                })

        nodes = []
        for n in nodes_dict.values():
            category = "major" if n.get("classification") == "story_progression" else "minor"
            nodes.append({
                "id": n["id"],
                "name": n.get("name", "Event"),
                "category": category,
                "anchor": n.get("temporal_anchor", ""),
                "stage_order": n.get("topological_order", 0),
                "first_page": n.get("page_start", 0),
                "pages": n.get("pages", []),
                "core": n.get("summary", ""),
            })

        edges = []
        for r in links:
            edges.append({
                "src": r["source"],
                "dst": r["target"],
                "confidence": 1.0,
                "kind": "causal" if r.get("relation") == "CAUSES" else "temporal",
            })

        nodes.sort(key=lambda x: x["stage_order"])
        return {"nodes": nodes, "edges": edges, "links": links}


def neighbours(seed_ids: list[str], hops: int = 1) -> list[str]:
    """Expands seed nodes along graph relationships."""
    if not seed_ids:
        return []
    with neo4j().session() as session:
        result = session.run(
            """
            MATCH (s:Event) WHERE s.id IN $seeds
            MATCH path = (s)-[*1..%d]-(n:Event)
            RETURN DISTINCT n.id AS id
            """ % hops,
            seeds=seed_ids,
        )
        return [r["id"] for r in result]
