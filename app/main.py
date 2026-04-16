import os
import pickle
import numpy as np
from dotenv import load_dotenv

from loader import PDFLoader
from chunker import TextChunker
from embedder import Embedder
from retriever import Retriever
from generator import Generator

load_dotenv(dotenv_path="../.env")

CACHE_PATH = "../data/embeddings_cache.pkl"


def load_or_build_index(file_path: str, cache_path: str):
    """
    Load chunks + embeddings from disk if the cache exists,
    otherwise build them from scratch and save for next time.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path} ...")
        with open(cache_path, "rb") as f:
            chunks, embeddings = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from cache.")
        embedder = Embedder()  # still need the model for query encoding
        return chunks, embeddings, embedder

    print("No cache found. Building index from PDF ...")
    pdf_loader = PDFLoader(file_path)
    documents = pdf_loader.load()
    pdf_loader.print_statistics(documents)

    chunker = TextChunker()
    chunks = chunker.split(documents)
    chunker.print_chunk_statistics(chunks)

    embedder = Embedder()
    embeddings = embedder.generate_embeddings(chunks)
    embedder.print_embedding_details(embeddings)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((chunks, embeddings), f)
    print(f"\nCache saved to {cache_path}")

    return chunks, embeddings, embedder


def main():
    # --- Configuration ---
    file_path = "../data/Good-Clinical-Practice-Guideline.pdf"
    groq_api_key = os.getenv("GROQ_API_KEY")
    api_key =groq_api_key
    if not api_key:
        raise EnvironmentError(
            "API Key is not Available"
        )

    # --- Build or load index ---
    chunks, embeddings, embedder = load_or_build_index(file_path, CACHE_PATH)

    retriever = Retriever(embedder)
    generator = Generator(api_key=api_key)

    print("\n Medical RAG System is Live!")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask a medical guideline question: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        # Retrieve relevant chunks
        results = retriever.retrieve(query, chunks, embeddings, top_k=3)

        # Show retrieved sources (optional, helpful for debugging)
        print("\nTop sources retrieved:")
        for i, r in enumerate(results, 1):
            page = r["metadata"].get("page", "?")
            print(f"  [{i}] Page {page} | score: {r['score']:.3f}")

        # Generate answer
        print("\nThinking...\n")
        answer = generator.get_answer(query, results)

        print("=" * 50)
        print(f"ANSWER:\n{answer}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()