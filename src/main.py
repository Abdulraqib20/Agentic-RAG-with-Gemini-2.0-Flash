import os
import sys
from pathlib import Path
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import google.generativeai as genai
import bs4
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from config.appconfig import (
    QDRANT_LOCATION,
    QDRANT_URL,
    QDRANT_API_KEY,
    GROQ_API_KEY,
    GOOGLE_API_KEY,
    EXAAI_API_KEY
)

# Constants
COLLECTION_NAME = "gemini-thinking-agent-agno"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_MODEL = "gemini/gemini-2.0-flash"

# Set page configuration
st.set_page_config(
    page_title="Agentic RAG with Gemini Flash",
    page_icon="🤔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown(""" 
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4B3FFF;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #6C63FF;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4B3FFF;
        margin-top: 1rem;
    }
    .card {
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #4B3FFF;
        color: white;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .chat-message-user {
        background-color: #E0E7FF;
        border-left: 5px solid #4B3FFF;
    }
    .chat-message-assistant {
        background-color: #F0F2F6;
        border-left: 5px solid #6C63FF;
    }
    .source-card {
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        background-color: #f1f3f9;
        border-left: 3px solid #6C63FF;
    }
    .expander-header {
        font-weight: 600;
        color: #4B3FFF;
    }
    .stButton>button {
        background-color: #4B3FFF;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #3730A3;
    }
</style>
""", unsafe_allow_html=True)

class GeminiEmbedder(Embeddings):
    def __init__(self, model_name=GEMINI_EMBEDDING_MODEL):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']

#--------------------------------------
# Streamlit App Initialization
#--------------------------------------
st.markdown("<div class='main-header'>👨‍💻 Agentic RAG with Gemini 2.0 Flash</div>", unsafe_allow_html=True)
st.markdown("""
<div class='card'>
    <p>An intelligent RAG system powered by Google's Gemini 2.0 Flash Thinking, Qdrant vector storage, and Agno agent orchestration.</p>
    <p>Upload documents, process web pages, and get AI-assisted answers with advanced query rewriting and web search capabilities.</p>
</div>
""", unsafe_allow_html=True)


# Session persistence
@st.cache_resource
def get_persistent_state():
    """Return a persistent state object that will retain data across sessions."""
    return {
        "vector_store": None,
        "processed_documents": [],
        "history": [],
        "use_web_search": True,
        "force_web_search": False,
        "similarity_threshold": 0.7
    }

# Initialize persistent state
persistent_state = get_persistent_state()

# # Session State Initialization
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'processed_documents' not in st.session_state:
#     st.session_state.processed_documents = []
# if 'history' not in st.session_state:
#     st.session_state.history = []
# if 'use_web_search' not in st.session_state:
#     st.session_state.use_web_search = True  # Enable by default
# if 'force_web_search' not in st.session_state:
#     st.session_state.force_web_search = False
# if 'similarity_threshold' not in st.session_state:
#     st.session_state.similarity_threshold = 0.7

# Session State Initialization with persistence
if 'initialized' not in st.session_state:
    # First time initialization from persistent state
    st.session_state.vector_store = persistent_state["vector_store"]
    st.session_state.processed_documents = persistent_state["processed_documents"]
    st.session_state.history = persistent_state["history"]
    st.session_state.use_web_search = persistent_state["use_web_search"]
    st.session_state.force_web_search = persistent_state["force_web_search"]
    st.session_state.similarity_threshold = persistent_state["similarity_threshold"]
    st.session_state.initialized = True

def update_persistent_state():
    """Update the persistent state from the current session state."""
    persistent_state["vector_store"] = st.session_state.vector_store
    persistent_state["processed_documents"] = st.session_state.processed_documents
    persistent_state["history"] = st.session_state.history
    persistent_state["use_web_search"] = st.session_state.use_web_search
    persistent_state["force_web_search"] = st.session_state.force_web_search
    persistent_state["similarity_threshold"] = st.session_state.similarity_threshold

# Sidebar Configuration
st.sidebar.markdown("<div class='sidebar-header'>📊 System Dashboard</div>", unsafe_allow_html=True)

# Document counter in sidebar
doc_count = len(st.session_state.processed_documents)
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>{doc_count}</h3>
        <p>Documents</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>{len(st.session_state.history) // 2}</h3>
        <p>Interactions</p>
    </div>
    """, unsafe_allow_html=True)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    update_persistent_state()
    st.rerun()

# Web Search Configuration
st.sidebar.markdown("<div class='sidebar-header'>🌐 Web Search Settings</div>", unsafe_allow_html=True)
st.session_state.use_web_search = st.sidebar.toggle("Enable Web Search Fallback", value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    # Optional domain filtering
    default_domains = [
        "arxiv.org", 
        "wikipedia.org", 
        "github.com", 
        "medium.com", 
        "linkedin.com"
    ]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)", 
        value=",".join(default_domains),
        help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org"
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Search Configuration
st.sidebar.markdown("<div class='sidebar-header'>🎯 Search Configuration</div>", unsafe_allow_html=True)
st.session_state.similarity_threshold = st.sidebar.slider(
    "Document Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    help="Lower values will return more documents but might be less relevant. Higher values are more strict."
)

# Utility Functions
def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    try:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None


# Document Processing Functions
def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


# Vector Store Management
def create_vector_store(client, texts):
    """Create and initialize vector store with documents."""
    try:
        # Create collection if needed
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,  # Gemini embedding-004 dimension
                    distance=Distance.COSINE
                )
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=GeminiEmbedder()
        )
        
        if not st.session_state.vector_store and st.session_state.processed_documents and qdrant_client:
            st.info("🔄 Reconnecting to your document database...")
            # Recreate the vector store connection
            st.session_state.vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=GeminiEmbedder()
            )
            update_persistent_state()
        
        # Add documents
        with st.spinner('📤 Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store
            
    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None


# Add this after the GeminiEmbedder class
def get_query_rewriter_agent() -> Agent:
    """Initialize a query rewriting agent."""
    return Agent(
        name="Query Rewriter",
        model=Gemini(id="gemini-exp-1206"),
        instructions="""You are an expert at reformulating questions to be more precise and detailed. 
        Your task is to:
        1. Analyze the user's question
        2. Rewrite it to be more specific and search-friendly
        3. Expand any acronyms or technical terms
        4. Return ONLY the rewritten query without any additional text or explanations
        
        Example 1:
        User: "What does it say about ML?"
        Output: "What are the key concepts, techniques, and applications of Machine Learning (ML) discussed in the context?"
        
        Example 2:
        User: "Tell me about transformers"
        Output: "Explain the architecture, mechanisms, and applications of Transformer neural networks in natural language processing and deep learning"
        """,
        show_tool_calls=False,
        markdown=True,
    )


def get_web_search_agent() -> Agent:
    """Initialize a web search agent."""
    return Agent(
        name="Web Search Agent",
        model=Gemini(id="gemini-exp-1206"),
        tools=[ExaTools(
            api_key=EXAAI_API_KEY,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""You are a web search expert. Your task is to:
        1. Search the web for relevant information about the query
        2. Compile and summarize the most relevant information
        3. Include sources in your response
        """,
        show_tool_calls=True,
        markdown=True,
    )


def get_rag_agent() -> Agent:
    """Initialize the main RAG agent."""
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions="""You are an Intelligent Agent specializing in providing accurate answers.
        
        When given context from documents:
        - Focus on information from the provided documents
        - Be precise and cite specific details
        
        When given web search results:
        - Clearly indicate that the information comes from web search
        - Synthesize the information clearly
        
        Always maintain high accuracy and clarity in your responses.
        """,
        show_tool_calls=True,
        markdown=True,
    )


def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    """
    Check if documents in vector store are relevant to the query.
    
    Args:
        query: The search query
        vector_store: The vector store to search in
        threshold: Similarity threshold
        
    Returns:
        tuple[bool, List]: (has_relevant_docs, relevant_docs)
    """
    if not vector_store:
        return False, []
        
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs


# Main Application Flow
genai.configure(api_key=GOOGLE_API_KEY)
qdrant_client = init_qdrant()

# Create tabs for a better organization
tab1, tab2 = st.tabs(["💬 Chat", "📁 Data Upload"])

with tab2:
    st.markdown("<div class='sub-header'>📁 Upload Documents</div>", unsafe_allow_html=True)
    
    # File upload section with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📄 PDF Upload")
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🌐 Web Page Import")
        web_url = st.text_input("Enter Website URL", placeholder="https://raqibcodes.netlify.app/")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process documents
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file:
            file_name = uploaded_file.name
            if st.button("Process PDF", key="process_pdf"):
                if file_name not in st.session_state.processed_documents:
                    with st.spinner('Processing PDF...'):
                        texts = process_pdf(uploaded_file)
                        st.write(f"{len(texts)} chunks found in Document")
                        if texts and qdrant_client:
                            if st.session_state.vector_store:
                                st.session_state.vector_store.add_documents(texts)
                            else:
                                st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                            st.session_state.processed_documents.append(file_name)
                            st.success(f"✅ Added PDF: {file_name}")
                            update_persistent_state()

    with col2:
        if web_url:
            if st.button("Process URL", key="process_url"):
                if web_url not in st.session_state.processed_documents:
                    with st.spinner('Processing URL...'):
                        texts = process_web(web_url)
                        st.write(f"{len(texts)} chunks found in URL")
                        # st.write(f"Debug: Found {len(texts)} chunks from URL")
                        if texts and qdrant_client:
                            if st.session_state.vector_store:
                                st.session_state.vector_store.add_documents(texts)
                            else:
                                st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                            st.session_state.processed_documents.append(web_url)
                            st.success(f"✅ Added URL: {web_url}")
                            update_persistent_state()

    # Display sources in data tab
    if st.session_state.processed_documents:
        st.markdown("<div class='sub-header'>📚 Processed Sources</div>", unsafe_allow_html=True)
        
        # Create a grid of document cards
        cols = st.columns(3)
        for i, source in enumerate(st.session_state.processed_documents):
            col_idx = i % 3
            with cols[col_idx]:
                if source.endswith('.pdf'):
                    st.markdown(f"""
                    <div class='source-card'>
                        <strong>📄 PDF Document</strong><br/>
                        {source}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='source-card'>
                        <strong>🌐 Web Page</strong><br/>
                        {source}
                    </div>
                    """, unsafe_allow_html=True)

with tab1:
    # # Display chat history
    # st.markdown("<div class='sub-header'>💬 Chat History</div>", unsafe_allow_html=True)
    
    
    # # Create a container for the chat history
    # chat_container = st.container()
    
    # # Create a container for the input that will always be at the bottom
    # input_container = st.container()
    
    # # Display chat history with proper styling in the chat container
    # with chat_container:
    #     for message in st.session_state.history:
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])
    
    # # Chat Interface in the input container (always at bottom)
    # with input_container:
    #     # Create two columns for chat input and search toggle
    #     chat_col, toggle_col = st.columns([0.9, 0.1])
        
    #     with chat_col:
    #         prompt = st.chat_input("Ask about your documents...")
        
    #     with toggle_col:
    #         st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    #         st.session_state.force_web_search = st.toggle('🌐', help="Force web search")
    
    # # Display chat history with proper styling
    # for message in st.session_state.history:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])
    
    
    # Create a layout with two main sections
    chat_area = st.container()
    input_area = st.container()
    
    # First put the input area at the bottom
    with input_area:
        # Create two columns for chat input and search toggle
        chat_col, toggle_col = st.columns([0.9, 0.1])
        
        with chat_col:
            prompt = st.chat_input("Ask about your documents...")
        
        with toggle_col:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            st.session_state.force_web_search = st.toggle('🌐', help="Force web search")
    
    # Then render the chat history above
    with chat_area:
        st.markdown("<div class='sub-header'>💬 Chat History</div>", unsafe_allow_html=True)
        
        # Display chat history with proper styling
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            

    if prompt:
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Step 1: Rewrite the query for better retrieval
        with st.spinner("🤔 Reformulating query..."):
            try:
                query_rewriter = get_query_rewriter_agent()
                rewritten_query = query_rewriter.run(prompt).content
                
                with st.expander("🔄 See rewritten query"):
                    st.markdown(f"""
                    <div class='source-card'>
                        <strong>Original:</strong> {prompt}<br/>
                        <strong>Rewritten:</strong> {rewritten_query}
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                
                error_message = "Apologies, I encountered an issue processing your request. Please try rephrasing or ask about another topic."
                st.session_state.history.append({
                    "role": "assistant",
                    "content": error_message
                })
                with st.chat_message("assistant"):
                    st.write(error_message)
                
                st.error(f"❌ Error rewriting query: {str(e)}")
                rewritten_query = prompt

        # Step 2: Choose search strategy based on force_web_search toggle
        context = ""
        docs = []
        if not st.session_state.force_web_search and st.session_state.vector_store:
            # Try document search first
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5, 
                    "score_threshold": st.session_state.similarity_threshold
                }
            )
            docs = retriever.invoke(rewritten_query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
            elif st.session_state.use_web_search:
                st.info("🔄 No relevant documents found in database, falling back to web search...")

        # Step 3: Use web search if:
        # 1. Web search is forced ON via toggle, or
        # 2. No relevant documents found AND web search is enabled in settings
        if (st.session_state.force_web_search or not context) and st.session_state.use_web_search:
            with st.spinner("🔍 Searching the web..."):
                try:
                    web_search_agent = get_web_search_agent()
                    web_results = web_search_agent.run(rewritten_query).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ Using web search as requested via toggle.")
                        else:
                            st.info("ℹ️ Using web search as fallback since no relevant documents were found.")
                except Exception as e:
                    st.error(f"❌ Web search error: {str(e)}")

        # Step 4: Generate response using the RAG agent
        with st.spinner("🤖 Thinking..."):
            try:
                rag_agent = get_rag_agent()
                
                if context:
                    full_prompt = f"""Context: {context}

                        Original Question: {prompt}
                        Rewritten Question: {rewritten_query}

                        Please provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = f"Original Question: {prompt}\nRewritten Question: {rewritten_query}"
                    st.info("ℹ️ No relevant information found in documents or web search.")

                response = rag_agent.run(full_prompt)
                
                # Add assistant response to history
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response.content
                })
                update_persistent_state()
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response.content)
                    
                    # Show sources if available
                    if not st.session_state.force_web_search and 'docs' in locals() and docs:
                        with st.expander("🔍 See document sources"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                                st.markdown(f"""
                                <div class='source-card'>
                                    <strong>{source_icon} Source {i} from {source_name}:</strong><br/>
                                    {doc.page_content[:200]}...
                                </div>
                                """, unsafe_allow_html=True)

            except Exception as e:
                print(f"Error generating response: {str(e)}")
                
                # User-friendly error message
                friendly_message = "I'm sorry, I couldn't process your request. Please try again or rephrase your question."
                
                # Add the friendly message to chat history
                st.session_state.history.append({
                    "role": "assistant",
                    "content": friendly_message
                })
                update_persistent_state()
                
                # Display the friendly message
                with st.chat_message("assistant"):
                    st.write(friendly_message)


                # st.error(f"❌ Error generating response: {str(e)}")
                
