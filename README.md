# **🔬 Advanced RAG Research Assistant**

This project is a Streamlit web application that serves as an advanced research assistant powered by a Retrieval-Augmented Generation (RAG) pipeline. It leverages state-of-the-art techniques like hybrid search, re-ranking, and query expansion to provide accurate, detailed, and cited answers from a local collection of documents in various formats (PDF, DOCX, TXT, and more).

## **✨ Key Features**

* **Multi-Format Document Support**: Ingests and processes .pdf, .docx, .txt, and many other file types seamlessly using Unstructured.  
* **Hybrid Search**: Combines the strengths of keyword-based (BM25) and semantic (ChromaDB vector search) retrieval for more relevant results.  
* **Re-ranking**: Uses a Cross-Encoder model to re-rank the retrieved documents, pushing the most relevant ones to the top.  
* **Query Expansion**: Automatically reformulates user questions into more detailed versions to improve retrieval accuracy (optional).  
* **LangChain Expression Language (LCEL)**: Built with the modern, composable LCEL for creating powerful, readable chains.  
* **Streamlit Interface**: A clean and interactive web interface for asking questions and viewing results.  
* **Citable Answers**: The final answer includes citations to the source documents, allowing users to verify the information.

## **🛠️ Tech Stack**

* **Framework**: Streamlit  
* **LLM & Embeddings**: Google Gemini (via langchain-google-genai)  
* **Core Orchestration**: LangChain  
* **Document Parsing**: Unstructured  
* **Vector Database**: ChromaDB  
* **Search**: BM25 & ChromaDB for hybrid search  
* **Re-ranking**: Hugging Face Cross-Encoders

## **🚀 Getting Started**

Follow these steps to set up and run the project locally.

### **Prerequisites**

* Python 3.10 or higher  
* A Google API Key with access to the Gemini models. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### **Step 1: Clone the Repository**

git clone https://github.com/bitphonix/Advance-RAG-Assistant.git  
cd Advance-RAG-Assistant

### **Step 2: Install Dependencies**

Create a virtual environment (recommended) and install the required Python packages from requirements.txt. The unstructured\[all-docs\] package will install all necessary libraries for parsing different document types.

\# Create a virtual environment  
python \-m venv venv  
source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

\# Install dependencies  
pip install \-r requirements.txt

### **Step 3: Set Up Your Environment File**

Create a file named .env in the root of the project directory and add your Google API key:

GOOGLE\_API\_KEY="your\_google\_api\_key\_here"

### **Step 4: Add Your Documents**

Place the files you want to query into the documents folder. You can add .pdf, .docx, .txt, and any other format supported by the unstructured library. The project comes with sample .txt files to get you started.

### **Step 5: Run the Streamlit App**

Once the setup is complete, you can run the application with the following command:

streamlit run app.py

The application will automatically build the ChromaDB vectorstore on the first run. You can clear and rebuild the database at any time using the "Clear and Rebuild Database" button in the sidebar.

## **⚙️ How It Works**

The application follows a multi-step RAG pipeline:

1. **Load & Split**: UnstructuredDirectoryLoader loads files from the /documents directory, automatically detecting their type and extracting text. The documents are then split into smaller, manageable chunks.  
2. **Hybrid Retrieval**: When a query is entered, an EnsembleRetriever fetches documents using both BM25 (keyword search) and a Chroma vectorstore (semantic search).  
3. **Re-ranking & Reordering**: The retrieved documents are passed to a CrossEncoderReranker which re-sorts them based on semantic relevance to the query. A LongContextReorder step ensures the most relevant documents are placed at the beginning and end of the context window.  
4. **Generation**: The refined context and the original query are fed into a Google Gemini model via a LangChain Expression Language (LCEL) chain to generate a helpful, cited answer.
