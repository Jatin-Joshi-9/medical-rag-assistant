import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from typing import List, Dict
import re


class Retriever:
    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._init_collection()
        self._bm25 = None
        self._bm25_docs = []

    def _init_collection(self):
        """Create a fresh empty collection."""
        self.collection = self.client.get_or_create_collection(
            name="medical_guidelines",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        """
        Drop and recreate the collection.
        Called when a new PDF is uploaded to wipe previous data.
        """
        try:
            self.client.delete_collection("medical_guidelines")
        except Exception:
            pass
        self._init_collection()
        self._bm25 = None
        self._bm25_docs = []

    def is_populated(self) -> bool:
        return self.collection.count() > 0

    def index_chunks(self, chunks: List[Document]) -> None:
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    def _build_bm25(self) -> None:
        if self._bm25 is not None:
            return
        result = self.collection.get(include=["documents", "metadatas"])
        self._bm25_docs = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        tokenized = [self._tokenize(d["content"]) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _expand_query(self, query: str) -> List[str]:
        queries = [query]
        words = query.strip().split()
        if len(words) <= 3:
            queries.append(f"definition and explanation of {query}")
            queries.append(f"guidelines and rules regarding {query}")
            queries.append(f"what are the procedures for {query}")
        return queries

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        self._build_bm25()
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self._bm25_docs[idx]["content"],
                    "score": float(scores[idx]),
                    "metadata": self._bm25_docs[idx]["metadata"],
                    "source": "bm25"
                })
        return results

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "content": doc,
                "score": round(1 - dist, 4),
                "metadata": meta,
                "source": "semantic"
            })
        return hits

    def _mmr_deduplicate(self, candidates: List[Dict], top_k: int, diversity: float = 0.4) -> List[Dict]:
        if not candidates:
            return []
        max_bm25 = max(
            (c["score"] for c in candidates if c.get("source") == "bm25"), default=1
        )
        normalized = []
        for c in candidates:
            score = c["score"] / max_bm25 if c.get("source") == "bm25" else c["score"]
            normalized.append({**c, "norm_score": score})

        selected = []
        remaining = normalized[:]

        while len(selected) < top_k and remaining:
            if not selected:
                best = max(remaining, key=lambda x: x["norm_score"])
            else:
                def mmr_score(candidate):
                    rel = candidate["norm_score"]
                    max_sim = max(
                        self._text_overlap(candidate["content"], s["content"])
                        for s in selected
                    )
                    return (1 - diversity) * rel - diversity * max_sim
                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return selected

    def _text_overlap(self, text_a: str, text_b: str) -> float:
        tokens_a = set(self._tokenize(text_a))
        tokens_b = set(self._tokenize(text_b))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        queries = self._expand_query(query)
        fetch_k = top_k * 3
        all_candidates = []
        seen_contents = set()

        for q in queries:
            for hit in self._semantic_search(q, fetch_k):
                if hit["content"] not in seen_contents:
                    all_candidates.append(hit)
                    seen_contents.add(hit["content"])
            for hit in self._bm25_search(q, fetch_k):
                if hit["content"] not in seen_contents:
                    all_candidates.append(hit)
                    seen_contents.add(hit["content"])

        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        return self._mmr_deduplicate(all_candidates, top_k=top_k)