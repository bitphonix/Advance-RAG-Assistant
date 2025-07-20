# 🔬 Advanced RAG Research Assistant

This project is a Streamlit web application that serves as an advanced research assistant powered by a Retrieval-Augmented Generation (RAG) pipeline. It leverages state-of-the-art techniques like hybrid search, re-ranking, and query expansion to provide accurate, detailed, and cited answers from a local document collection.

## ✨ Key Features

- **Hybrid Search**: Combines the strengths of keyword-based (BM25) and semantic (ChromaDB vector search) retrieval for more relevant results.
- **Re-ranking**: Uses a Cross-Encoder model to re-rank the retrieved documents, pushing the most relevant ones to the top.
- **Query Expansion**: Automatically reformulates user questions into more detailed versions to improve retrieval accuracy (optional).
- **LangChain Expression Language (LCEL)**: Built with the modern, composable LCEL for creating powerful, readable chains.
- **Streamlit Interface**: A clean and interactive web interface for asking questions and viewing results.
- **Citable Answers**: The final answer includes citations to the source documents, allowing users to verify the information.

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **LLM & Embeddings**: Google Gemini (via `langchain-google-genai`)
- **Core Orchestration**: LangChain
- **Vector Database**: ChromaDB
- **Search**: BM25 & ChromaDB for hybrid search
- **Re-ranking**: Hugging Face Cross-Encoders

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.10 or higher
- A Google API Key with access to the Gemini models. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>