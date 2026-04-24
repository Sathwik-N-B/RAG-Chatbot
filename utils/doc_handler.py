import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import os
import re

# Custom EnsembleRetriever implementation
class EnsembleRetriever:
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights
    
    def _get_docs(self, retriever, query):
        """Get documents from a retriever, handling both old and new LangChain API"""
        if hasattr(retriever, 'invoke'):
            return retriever.invoke(query)
        elif hasattr(retriever, 'get_relevant_documents'):
            return retriever.get_relevant_documents(query)
        else:
            raise AttributeError(f"Retriever {type(retriever)} has neither 'invoke' nor 'get_relevant_documents' method")
    
    def get_relevant_documents(self, query):
        all_docs = []
        doc_scores = {}
        
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = self._get_docs(retriever, query)
            for doc in docs:
                doc_id = doc.page_content
                if doc_id in doc_scores:
                    doc_scores[doc_id] = (doc_scores[doc_id][0] + weight, doc_scores[doc_id][1])
                else:
                    doc_scores[doc_id] = (weight, doc)
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs]
    
    def invoke(self, query):
        """Alias for get_relevant_documents to match LangChain's newer API"""
        return self.get_relevant_documents(query)


def process_documents(uploaded_files,reranker,embedding_model, base_url):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []
    
    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Process files
    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
                
            documents.extend(loader.load())
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return

    # Text splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    text_contents = [doc.page_content for doc in texts]

    # 🚀 Hybrid Retrieval Setup
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    
    # Vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # BM25 store
    bm25_retriever = BM25Retriever.from_texts(
        text_contents, 
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
    )

    # Ensemble retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 5})
        ],
        weights=[0.4, 0.6]
    )

    # Store in session
    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,  # Now using the global reranker variable
        "texts": text_contents,
        "knowledge_graph": build_knowledge_graph(texts)  # Store Knowledge Graph
    }

    st.session_state.documents_loaded = True
    st.session_state.processing = False

    # ✅ Debugging: Print Knowledge Graph Nodes & Edges
    if "knowledge_graph" in st.session_state.retrieval_pipeline:
        G = st.session_state.retrieval_pipeline["knowledge_graph"]
        st.write(f"🔗 Total Nodes: {len(G.nodes)}")
        st.write(f"🔗 Total Edges: {len(G.edges)}")
        st.write(f"🔗 Sample Nodes: {list(G.nodes)[:10]}")
        st.write(f"🔗 Sample Edges: {list(G.edges)[:10]}")
