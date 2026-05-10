
# Medical RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed to provide accurate answers from medical guidelines (specifically Good Clinical Practice Guidelines) using local embeddings and the Llama-3.3-70b model via Groq.

##  Features
* **PDF Processing:** Extracts text from complex medical PDF documents.
* **Intelligent Chunking:** Uses `RecursiveCharacterTextSplitter` to maintain context.
* **Local Embeddings:** Uses `all-MiniLM-L6-v2` for fast, local vector generation.
* **Vector Cache:** Pickles embeddings to disk to avoid redundant processing.
* **Advanced Retrieval:** Implements manual Cosine Similarity for top-k document retrieval.
* **LLM Generation:** Leverages Groq's Llama-3.3-70b for high-speed, professional responses.

##  Project Structure
- `app/main.py`: Entry point for the application.
- `app/loader.py`: Handles PDF document loading.
- `app/chunker.py`: Splits documents into manageable segments.
- `app/embedder.py`: Converts text chunks into numerical vectors.
- `app/retriever.py`: Performs semantic search using cosine similarity.
- `app/generator.py`: Constructs prompts and interfaces with Groq Cloud API.

##  Setup & API Configuration

### 1. Get a Groq API Key
1. Go to the [Groq Console](https://console.groq.com/keys).
2. Sign in with your account.
3. Click **"Create API Key"**.
4. Give it a name (e.g., `Medical-RAG`) and copy the key immediately.

### 2. Environment Setup
1. Create a `.env` file in the root directory (one level above `app/`).
2. Add your key to the file:
   ```text
   GROQ_API_KEY=your_gsk_key_here
   ```
3. Ensure your PDF is placed in `data/<your_random>.pdf`.

### 3. Installation
```bash
pip install -r requirements.txt
```

##  Testing & Validation Logic

The system includes built-in validation at every step of the pipeline:

### 1. Data Integrity (Loader)
The `PDFLoader` checks if the file exists and calculates characters-per-page. 
* **Validation:** If the character count is too low, it warns that the PDF might be an image/scanned file, preventing "empty" context errors.

### 2. Retrieval Accuracy (Retriever)
The `Retriever` uses **Cosine Similarity** to calculate the mathematical distance between your question and the document chunks.
* **Logic:** It calculates the dot product divided by the product of norms: 
    $$similarity = \frac{A \cdot B}{\|A\| \|B\|}$$
* **Validation:** It prints a "relevance score" for the top 3 sources. A score closer to 1.0 indicates a high semantic match.

### 3. Generation Grounding (Generator)
The LLM is restricted by a "System Prompt" to prevent hallucinations.
* **Rules Applied:** - MUST use provided context only.
    - MUST cite source numbers.
    - MUST admit if information is missing.
* **Validation:** If the `GROQ_API_KEY` is missing from the environment, the system raises a specific `EnvironmentError` before attempting to run.

##  How to Run
```bash
cd app
python main.py
```
