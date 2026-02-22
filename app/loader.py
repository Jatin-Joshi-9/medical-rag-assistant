from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
import os
class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self) -> List[Document]:
        loader = PyPDFLoader(self.file_path)
        pdf_documents = loader.load()
        return pdf_documents

    def print_statistics(self, documents: List[Document]) -> None:
        total_pages = len(documents)
        total_characters = 0
        empty_pages = 0

        for doc in documents:
            content_length = len(doc.page_content)
            total_characters += content_length

            if not doc.page_content.strip():
                empty_pages += 1

        print("pdf details:")
        print(f"File: {self.file_path}")
        print(f"Total Pages: {total_pages}")
        print(f"Total Characters: {total_characters}")
        print(f"Empty Pages: {empty_pages}")
