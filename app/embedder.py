from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from typing import List


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)


    def generate_embeddings(self, chunks: List[Document]) -> List[list]:
        embeddings = []
        for chunk in chunks:
            vector = self.model.encode(chunk.page_content)
            embeddings.append(vector)
        return embeddings


    def print_embedding_details(self, embeddings: List[list]) -> None:
        total_vectors = len(embeddings)
        if total_vectors > 0:
            dimension = len(embeddings[0])
        else:
            dimension = 0
        print("\nEmbedding details:")
        print(f"Total Vectors Generated: {total_vectors}")
        print(f"Embedding Dimension: {dimension}")
