from embedder import Embedder
from loader import PDFLoader
from chunker import TextChunker

def main():
    file_path = "../data/Good-Clinical-Practice-Guideline.pdf"
    print("file_Path: ", file_path)

    pdf_loader = PDFLoader(file_path)
    documents = pdf_loader.load()
    pdf_loader.print_statistics(documents)

    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split(documents)
    chunker.print_chunk_statistics(chunks)
    print("First chunk preview:\n")
    print(chunks[0].page_content)

    embedder = Embedder()
    embeddings = embedder.generate_embeddings(chunks)
    embedder.print_embedding_details(embeddings)

if __name__ == "__main__":
    main()