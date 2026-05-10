import os
from dotenv import load_dotenv

from loader import PDFLoader
from chunker import TextChunker
from retriever import Retriever
from generator import Generator

load_dotenv(dotenv_path="../.env")

CHROMA_DIR = "../data/chroma_db"


def main():
    file_path = "../data/Good-Clinical-Practice-Guideline.pdf"
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    retriever = Retriever(persist_dir=CHROMA_DIR)

    # Only index if DB is empty (first run)
    if not retriever.is_populated():
        print("ChromaDB empty. Building index from PDF...")
        loader = PDFLoader(file_path)
        docs = loader.load()
        loader.print_statistics(docs)

        chunker = TextChunker()
        chunks = chunker.split(docs)
        chunker.print_chunk_statistics(chunks)

        retriever.index_chunks(chunks)
    else:
        print(f"ChromaDB loaded ({retriever.collection.count()} vectors).")

    generator = Generator(api_key=api_key)

    print("\nMedical RAG System is Live!")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask a medical guideline question: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        results = retriever.retrieve(query, top_k=5)

        print("\nTop sources retrieved:")
        for i, r in enumerate(results, 1):
            page = r["metadata"].get("page", "?")
            source = r.get("source", "?")
            print(f"  [{i}] Page {page} | score: {r['score']:.3f} | via {source}")

        print("\nThinking...\n")
        answer = generator.get_answer(query, results)

        print("=" * 50)
        print(f"ANSWER:\n{answer}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()