# app.py

import os
import shutil
import time
import streamlit as st
from dotenv import load_dotenv
from functools import wraps
from google.api_core.exceptions import ResourceExhausted


from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, CrossEncoderReranker
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import UnstructuredDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import LongContextReorder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

class Config:
    EMBEDDING_MODEL_NAME = "models/text-embedding-004"
    LLM_MODEL_NAME = "gemini-1.5-flash"
    DB_PERSIST_PATH = f"chroma_db_{EMBEDDING_MODEL_NAME.replace('/', '_')}"
    DOCS_PATH = "./documents"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    HYBRID_SEARCH_K = 10
    ENSEMBLE_WEIGHTS = [0.4, 0.6]
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    RERANK_TOP_N = 5

def retry_with_backoff(retries=5, initial_delay=1, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except ResourceExhausted as e:
                    if i == retries - 1:
                        raise
                    st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

def configure_environment():
    load_dotenv()
    st.set_page_config(page_title="Advanced RAG Research Assistant", layout="wide")
    st.title("🔬 Advanced RAG Research Assistant")

@st.cache_resource
def load_and_split_documents():
    st.info("Step 1: Loading and splitting documents from multiple formats...")
    loader = UnstructuredDirectoryLoader(Config.DOCS_PATH, show_progress=True)
    docs = loader.load()
    if not docs:
        st.error(f"No documents found in the '{Config.DOCS_PATH}' directory.")
        st.stop()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)
    st.success(f"Loaded and split {len(docs)} documents into {len(split_docs)} chunks.")
    return split_docs

@st.cache_resource
def setup_retrievers(_split_docs):
    st.info(f"Step 2: Setting up Hybrid Search. DB persisted to '{Config.DB_PERSIST_PATH}'")
    embedding_model = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        _split_docs, embedding_model, persist_directory=Config.DB_PERSIST_PATH
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": Config.HYBRID_SEARCH_K})
    bm25_retriever = BM25Retriever.from_documents(_split_docs)
    bm25_retriever.k = Config.HYBRID_SEARCH_K
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=Config.ENSEMBLE_WEIGHTS
    )
    st.success("Hybrid retriever created successfully.")
    return ensemble_retriever

@st.cache_resource
def setup_reranking_pipeline(_ensemble_retriever):
    st.info("Step 3: Setting up Re-ranking and Context Reordering...")
    model = HuggingFaceCrossEncoder(model_name=Config.CROSS_ENCODER_MODEL)
    reranker = CrossEncoderReranker(model=model, top_n=Config.RERANK_TOP_N)
    reordering = LongContextReorder()
    pipeline_compressor = DocumentCompressorPipeline(transformers=[reranker, reordering])
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=_ensemble_retriever)
    st.success("Re-ranking pipeline is ready.")
    return compression_retriever

@st.cache_resource
def setup_lcel_qa_chain(_compression_retriever):
    """
    Set up the final Question-Answering chain using the LangChain Expression Language (LCEL).
    This is the modern, standard way to build chains.
    """
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL_NAME, temperature=0, convert_system_message_to_human=True)

    prompt_template = """You are an expert research assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Your answer should be detailed and cover all aspects of the question.
    For each piece of information you use, you MUST cite the source document. The source is available in the metadata of each context passage.
    Format your citation as [Source: file_name].

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}\nContent: {doc.page_content}" for doc in docs)

    rag_chain = (
        {"context": _compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

@retry_with_backoff()
def generate_expanded_queries(query, llm_model):
    st.info("Step 4a: Performing Query Expansion...")
    llm = llm_model
    expansion_prompt = f"""You are an expert at query expansion. Reformulate the following query into 3 different, more detailed versions.
    Provide a variety of perspectives to improve document retrieval. Do not number the queries.
    Original Query: {query}
    Expanded Queries:"""
    response = llm.invoke(expansion_prompt)
    expanded_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    st.write("Expanded queries:", expanded_queries)
    return expanded_queries


@retry_with_backoff()
def invoke_rag_chain(rag_chain, query):
    st.info("Step 4b: Retrieving, Re-ranking, and Generating Answer...")
    return rag_chain.invoke(query)

def main():
    configure_environment()

    st.sidebar.header("Controls")
    if st.sidebar.button("Clear and Rebuild Database"):
        if os.path.exists(Config.DB_PERSIST_PATH):
            shutil.rmtree(Config.DB_PERSIST_PATH)
            st.sidebar.success(f"Database '{Config.DB_PERSIST_PATH}' cleared.")
            st.rerun()

    use_query_expansion = st.sidebar.toggle("Enable Query Expansion", value=False, help="Uses an extra API call.")

    docs = load_and_split_documents()
    ensemble_retriever = setup_retrievers(docs)
    compression_retriever = setup_reranking_pipeline(ensemble_retriever)
    qa_chain = setup_lcel_qa_chain(compression_retriever)

    st.sidebar.header("Ask a Question")
    query = st.sidebar.text_input("Enter your research question:", key="query_input")

    if query:
        retrieved_docs = compression_retriever.get_relevant_documents(query)
        
        with st.spinner("Thinking..."):
            if use_query_expansion:
                temp_llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL_NAME, temperature=0)
                generate_expanded_queries(query, temp_llm)

            answer = invoke_rag_chain(qa_chain, query)

            st.subheader("Answer:")
            st.markdown(answer) 
            st.subheader("Sources Used:")
            for doc in retrieved_docs:
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                st.info(f"- **{source}**")
                with st.expander("Show content"):
                    st.write(doc.page_content)

if __name__ == "__main__":
    if not os.path.exists(Config.DOCS_PATH):
        os.makedirs(Config.DOCS_PATH)
        with open(os.path.join(Config.DOCS_PATH, "paper_on_fusion.txt"), "w") as f:
            f.write("Nuclear fusion...")
        with open(os.path.join(Config.DOCS_PATH, "article_on_llms.txt"), "w") as f:
            f.write("Large Language Models (LLMs)...")
        with open(os.path.join(Config.DOCS_PATH, "manual_for_hybrid_search.txt"), "w") as f:
            f.write("Hybrid search...")

    main()