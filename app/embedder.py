from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from typing import List
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks: List[Document]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def print_embedding_details(self, embeddings: np.ndarray) -> None:
        total_vectors = len(embeddings)
        dimension = len(embeddings[0]) if total_vectors > 0 else 0

        print("\nEmbedding details:")
        print(f"  Total Vectors Generated: {total_vectors}")
        print(f"  Embedding Dimension: {dimension}")