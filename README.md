# Vector Semantic Search Engine

This system retrieves information based on conceptual meaning using AI-powered embeddings, rather than literal keyword matching.

## 🚀 Overview
This project implements a semantic retrieval pipeline. It transforms unstructured text into high-dimensional vectors. It uses **Google Gemini (LLM/Embeddings)** for context-aware representation and **FAISS** for fast similarity searches. This is a core architecture used in modern Retrieval-Augmented Generation (RAG) systems.

## 🛠️ Tech Stack
- **Language:** Python
- **AI Models:** Google Gemini (`gemini-embedding-001`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Data Science:** NumPy (Matrix operations & L2 Normalization)
- **Environment:** Dotenv (Secure API management)

## 🧠 Key Features
- **Semantic Retrieval:** This feature understands user intent and context to find relevant matches.
- **High-Performance Indexing:** FAISS `IndexFlatIP` is used for similarity matching across datasets in sub-milliseconds.
- **Optimized Math:** Vector normalization is implemented via NumPy to ensure accurate cosine similarity scoring.
- **Production-Ready Security:** API credentials are securely managed using environment variables. This prevents sensitive data leaks.

## 📈 How It Works
1.  **Embedding:** Text converts into 768-dimensional vectors using the Gemini API.
2.  **Normalization:** Vectors are normalized to unit length for precise inner-product search.
3.  **Indexing:** FAISS stores the vectors in a high-speed searchable index.
4.  **Querying:** User queries are embedded in real-time to find the "nearest neighbors" (most similar concepts) in the database.

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
