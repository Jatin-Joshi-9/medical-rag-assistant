from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        return chunks

    def print_chunk_statistics(self, chunks: List[Document]) -> None:
        total_chunks = len(chunks)
        total_characters = 0

        for chunk in chunks:
            total_characters += len(chunk.page_content)

        if total_chunks > 0:
            average_length = total_characters // total_chunks
        else:
            average_length = 0

        print("\nChunk details:\n")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Chunk Overlap: {self.chunk_overlap}")
        print(f"Total Chunks Created: {total_chunks}")
        print(f"Average Characters per Chunk: {average_length}")
