import networkx as nx
import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your"
}


def _normalize_tokens(text):
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in STOPWORDS]


def _is_meaningful_entity(entity):
    tokens = _normalize_tokens(entity)
    return bool(tokens)

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

    # Match nodes by meaningful token overlap (not substring on stopwords).
    matched_nodes = []
    for node in G.nodes:
        node_tokens = set(_normalize_tokens(node))
        if node_tokens and query_tokens.intersection(node_tokens):
            matched_nodes.append(node)
    
    if matched_nodes:
        related_nodes = []
        seen = set()
        for node in matched_nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    related_nodes.append(neighbor)
        return related_nodes[:top_k]

    return []
