# 🚀 Rapid-Context-RAG

A high-performance **Retrieval-Augmented Generation (RAG)** pipeline that uses semantic search to ground LLM responses in custom data.

This project implements a complete loop: **Vector Embedding** → **FAISS Retrieval** → **Contextual Generation**.

## 🌟 Key Updates
*   **Full RAG Integration:** Now generates natural language answers instead of just retrieving raw text.
*   **Gemini 2.5 Flash:** Powered by the latest [Google Gen AI SDK](https://google.devgemini-api/docs/libraries) for lightning-fast, context-aware responses.
*   **Optimized Retrieval:** Uses `IndexFlatIP` for cosine-based similarity matching in sub-milliseconds.

## 🛠️ Tech Stack
*   **LLM:** `gemini-2.5-flash` for high-speed response generation.
*   **Embeddings:** `gemini-embedding-001` (3072-dimensional vectors).
*   **Vector DB:** [FAISS](https://github.com) (Facebook AI Similarity Search).
*   **Processing:** [NumPy](https://numpy.org) for L2 normalization and matrix operations.

## 📈 Pipeline Architecture
1.  **Ingestion:** Reads `sentences.txt` and generates embeddings via the [Gemini API](https://google.dev).
2.  **Normalization:** Vectors are L2-normalized using NumPy to ensure accurate similarity scoring.
3.  **Indexing:** Normalized vectors are stored in a FAISS `IndexFlatIP` database.
4.  **Generation:**
    *   The user query is embedded and searched against the index.
    *   Top 5 relevant snippets are retrieved as context.
    *   Gemini 2.5 Flash synthesizes a final answer based **ONLY** on that context.

## 🔧 Installation
1.  Clone the repository:

    ```bash
    git clone https://github.com
    ```
2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your `.env` file:

    ```env
    GEMINI_API_KEY=your_actual_key_here
    ```

##   code Review Observations:
*   `gemini-2.5-flash` is suitable for RAG due to low latency and high reasoning capabilities.
*   Using `float32` for the `embeddings_matrix` balances memory usage and search precision.
*   The prompt instructs the model to use only the provided context to reduce hallucinations.

##   Future improvements
Considering adding a "chunking" strategy to handle larger text files or integrating a web interface using [Streamlit](https://streamlit.io).




