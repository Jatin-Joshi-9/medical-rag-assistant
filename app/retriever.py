import numpy as np
from typing import List, Dict
from langchain_core.documents import Document


class Retriever:
    def __init__(self, embedder):
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        chunks: List[Document],
        chunk_embeddings: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        # Encode the user query with the same model used for chunks
        query_vector = self.embedder.model.encode(query)

        # Cosine similarity: (A · B) / (||A|| * ||B||)
        dot_product = np.dot(chunk_embeddings, query_vector)
        norm_chunks = np.linalg.norm(chunk_embeddings, axis=1)
        norm_query = np.linalg.norm(query_vector)

        # Small epsilon avoids division by zero
        similarities = dot_product / (norm_chunks * norm_query + 1e-8)

        # argsort is ascending, flip for descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "content": chunks[idx].page_content,
                "score": float(similarities[idx]),
                "metadata": chunks[idx].metadata
            })

        return results