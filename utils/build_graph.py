import networkx as nx
import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your"
}

GENERIC_ENTITY_STOPWORDS = STOPWORDS | {
    "about", "after", "before", "between", "each", "else", "first", "found", "from",
    "into", "like", "more", "new", "other", "over", "same", "since", "system",
    "their", "there", "these", "those", "using", "used", "very", "where", "while"
}


def _normalize_tokens(text):
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) <= 2 or token in STOPWORDS:
            continue

        normalized = token
        for suffix in ("ing", "ers", "er", "ed", "ies", "es", "s"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
                normalized = normalized[: -len(suffix)]
                break

        tokens.append(normalized)
    return tokens


def _is_meaningful_entity(entity):
    raw_tokens = re.findall(r"[a-z0-9]+", entity.lower())
    if not raw_tokens:
        return False

    if len(raw_tokens) == 1:
        token = raw_tokens[0]
        return len(token) >= 4 and token not in GENERIC_ENTITY_STOPWORDS

    normalized_tokens = [token for token in _normalize_tokens(entity) if token not in GENERIC_ENTITY_STOPWORDS]
    return len(normalized_tokens) >= 2

def build_knowledge_graph(docs):
    G = nx.Graph()
    for doc in docs:
        entities = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', doc.page_content)
        entities = [entity for entity in entities if _is_meaningful_entity(entity)]
        # Ensure meaningful relationships exist
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                G.add_edge(entities[i], entities[i + 1])  # Create edge
    return G


def retrieve_from_graph(query, G, top_k=5):
    query_tokens = set(_normalize_tokens(query))
    if not query_tokens:
        return []

    # Match nodes by meaningful token overlap and rank by overlap strength.
    matched_nodes = []
    for node in G.nodes:
        node_tokens = set(_normalize_tokens(node))
        if not node_tokens:
            continue

        overlap = len(query_tokens.intersection(node_tokens))
        if overlap >= 2 or node.lower() in query.lower():
            matched_nodes.append((overlap, node))

    matched_nodes.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    
    if matched_nodes:
        related_nodes = []
        seen = set()
        for _, node in matched_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    related_nodes.append(neighbor)
        return related_nodes[:top_k]

    return []
