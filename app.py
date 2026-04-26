import streamlit as st
import requests
import json
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from sentence_transformers import CrossEncoder
import torch
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import time
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # Fix for torch classes not found error
load_dotenv(find_dotenv())  # Loads .env file contents into the application based on key-value pairs defined therein, making them accessible via 'os' module functions like os.getenv().

st.set_page_config(page_title="DeepGraph RAG-Pro", layout="wide")      # ✅ Streamlit configuration

# 🌐 LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = os.getenv("MODEL", "huihui-ai/Qwen3-1.7B-abliterated")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "z-ai/glm-5.1")

EMBEDDINGS_MODEL = "nomic-embed-text:latest"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Initialize NVIDIA client if API key is available
nvidia_client = None
if NVIDIA_API_KEY:
    nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = None                                                        # 🚀 Initialize Cross-Encoder (Reranker) at the global level 
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)


                                                                                    # Manage Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "NVIDIA" if nvidia_client else "Ollama"


with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files,reranker,EMBEDDINGS_MODEL, OLLAMA_BASE_URL)
            st.success("Documents processed!")
    
    st.markdown("---")
    st.header("⚙️ RAG Settings")
    
    # 🌐 LLM Provider Selection
    if nvidia_client:
        st.session_state.llm_provider = st.radio(
            "🤖 Select LLM Provider",
            ["NVIDIA", "Ollama"],
            help="Choose between NVIDIA (online) or Ollama (local)"
        )
        if st.session_state.llm_provider == "NVIDIA":
            st.info("✅ Using NVIDIA API for responses")
        else:
            st.info("📦 Using Ollama (local) for responses")
    else:
        st.session_state.llm_provider = "Ollama"
        st.warning("⚠️ NVIDIA API key not found. Using Ollama (local) mode.")
    
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # 🚀 Footer (Bottom Right in Sidebar) For some Credits :)
    st.sidebar.markdown("""
        <div style="position: absolute; top: 20px; right: 10px; font-size: 12px; color: gray;">
            <b>Developed by:</b> N Sai Akhil &copy; All Rights Reserved 2025
        </div>
    """, unsafe_allow_html=True)

# 💬 Chat Interface
st.title("🤖 DeepGraph RAG-Pro")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])  # Last 5 messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 🚀 Build context
        context = ""
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                use_nvidia = st.session_state.llm_provider == "NVIDIA" and nvidia_client
                docs = retrieve_documents(
                    prompt, 
                    OLLAMA_API_URL, 
                    NVIDIA_MODEL if use_nvidia else OLLAMA_MODEL, 
                    chat_history,
                    use_nvidia=use_nvidia,
                    nvidia_client=nvidia_client
                )
                context = "\n".join(
                    f"[Source {i+1}]: {doc.page_content}" 
                    for i, doc in enumerate(docs)
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
        
        # Keep responses concise and user-facing (no chain-of-thought style output).
        system_prompt = f"""You are a helpful assistant.
    Use chat history only for context continuity.
    Answer directly and concisely.
    Do not reveal internal reasoning, analysis steps, or chain-of-thought.
    If the context is insufficient, say so briefly and ask one clarifying follow-up.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {prompt}
    Answer:"""
        
        # 🌐 Generate response using selected provider
        if st.session_state.llm_provider == "NVIDIA" and nvidia_client:
            # 🚀 NVIDIA Streaming Response
            try:
                stream = nvidia_client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=st.session_state.temperature,
                    top_p=1,
                    max_tokens=16384,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True}},
                    stream=True
                )
                
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
                        continue
                    delta = chunk.choices[0].delta
                    token = getattr(delta, "content", None)
                    if token is None:
                        continue
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"NVIDIA Generation error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error with NVIDIA."})
        
        else:
            # 🚀 Ollama Streaming Response
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": st.session_state.temperature,
                        "num_ctx": 4096
                    }
                },
                stream=True
            )
            try:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode())
                        token = data.get("response", "")
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                        
                        # Stop if we detect the end token
                        if data.get("done", False):
                            break
                            
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Ollama Generation error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
