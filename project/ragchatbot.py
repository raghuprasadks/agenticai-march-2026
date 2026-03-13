import streamlit as st
import chromadb
import cohere
import PyPDF2
import os
from io import BytesIO
import uuid
from datetime import datetime

# Configuration
COHERE_API_KEY = "3u0H4lmf6YYOFyMTosFyRipuj1zj6VzHr34EeB0w"
CHROMA_DB_DIR = "./chroma"
COLLECTION_NAME = "rag_pdf_documents"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file-like object"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None


def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, collection


def upload_pdf_to_chromadb(pdf_file, collection, cohere_client):
    """Upload PDF to ChromaDB with embeddings"""
    try:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        if not pdf_text:
            return False, "Failed to extract text from PDF"

        # Generate filename and unique ID
        file_name = pdf_file.name
        doc_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Generate embedding using Cohere
        embedding = cohere_client.embed(texts=[pdf_text]).embeddings[0]

        # Store in ChromaDB
        collection.add(
            ids=[doc_id],
            documents=[pdf_text],
            embeddings=[embedding],
            metadatas=[{
                "filename": file_name,
                "timestamp": timestamp,
                "length": len(pdf_text)
            }]
        )

        return True, f"Successfully uploaded: {file_name}"
    except Exception as e:
        return False, f"Error uploading PDF: {str(e)}"


def search_chromadb(query, collection, cohere_client, top_k=3):
    """Search ChromaDB for relevant documents"""
    try:
        # Generate embedding for query
        query_emb = cohere_client.embed(texts=[query]).embeddings[0]

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )

        # Format results
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                search_results.append({
                    'document': doc,
                    'source': metadata.get('filename', 'Unknown'),
                    'relevance_score': round(1 - distance, 3)  # Convert distance to similarity
                })

        return search_results
    except Exception as e:
        st.error(f"Error searching ChromaDB: {str(e)}")
        return []


def generate_answer(cohere_client, query, docs):
    """Generate answer using Cohere based on retrieved documents"""
    try:
        context = "\n\n".join([doc['document'][:500] for doc in docs])  # Limit context length
        
        message = f"""Based on the following documents, answer the question:

Context:
{context}

Question: {query}

Answer:"""

        response = cohere_client.chat(
            message=message,
            max_tokens=512,
            temperature=0.7
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def get_collection_stats(collection):
    """Get statistics about the collection"""
    try:
        data = collection.get()
        return {
            'total_docs': len(data['ids']),
            'ids': data['ids'],
            'metadatas': data['metadatas']
        }
    except Exception as e:
        st.error(f"Error getting collection stats: {str(e)}")
        return {'total_docs': 0}


def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("📚 RAG Chatbot - PDF Search & Chat")
    
    # Initialize clients
    cohere_client = cohere.Client(COHERE_API_KEY)
    _, collection = initialize_chromadb()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # PDF Upload Section
        st.subheader("Upload PDF Documents")
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            accept_multiple_files=False
        )
        
        if uploaded_pdf is not None:
            if st.button("📤 Upload to ChromaDB"):
                with st.spinner("Processing and uploading PDF..."):
                    success, message = upload_pdf_to_chromadb(
                        uploaded_pdf,
                        collection,
                        cohere_client
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Collection Statistics
        st.subheader("📊 Collection Statistics")
        stats = get_collection_stats(collection)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", stats['total_docs'])
        with col2:
            st.metric("Collection", COLLECTION_NAME)
        
        # List uploaded documents
        if stats['total_docs'] > 0:
            st.subheader("📋 Uploaded Documents")
            for i, (doc_id, metadata) in enumerate(zip(stats['ids'], stats['metadatas']), 1):
                with st.expander(f"{i}. {metadata.get('filename', 'Unknown')}"):
                    st.write(f"**ID:** {doc_id[:8]}...")
                    st.write(f"**Size:** {metadata.get('length', 0)} characters")
                    st.write(f"**Uploaded:** {metadata.get('timestamp', 'Unknown')}")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "💬 Chat", "ℹ️ Info"])
    
    with tab1:
        st.header("Search Documents")
        st.write("Search through your uploaded PDF documents")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input("Enter your search query:", placeholder="e.g., What is artificial intelligence?")
        with col2:
            search_clicked = st.button("🔎 Search", key="search_btn")
        
        if search_clicked and search_query:
            with st.spinner("Searching documents..."):
                results = search_chromadb(search_query, collection, cohere_client, top_k=5)
                st.session_state.search_results = results
        
        # Display search results
        if st.session_state.search_results:
            st.subheader("Search Results")
            for i, result in enumerate(st.session_state.search_results, 1):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.metric("Relevance", f"{result['relevance_score']:.3f}")
                    with col2:
                        st.write(f"**Source:** {result['source']}")
                    
                    with st.expander(f"Result {i}: {result['document'][:100]}..."):
                        st.write(result['document'][:1000])
                        if len(result['document']) > 1000:
                            st.info(f"Full document is {len(result['document'])} characters long")
                    st.divider()
    
    with tab2:
        st.header("RAG Chatbot")
        st.write("Ask questions about your uploaded documents")
        
        # Check if collection has documents
        stats = get_collection_stats(collection)
        if stats['total_docs'] == 0:
            st.warning("⚠️ No documents uploaded yet. Please upload PDFs in the sidebar first.")
        else:
            # Chat interface
            col1, col2 = st.columns([4, 1])
            with col1:
                chat_query = st.text_input(
                    "You:",
                    placeholder="Ask a question about your documents...",
                    key="chat_input"
                )
            with col2:
                chat_clicked = st.button("Ask", key="chat_btn")
            
            if chat_clicked and chat_query:
                with st.spinner("Searching and generating answer..."):
                    # Search for relevant documents
                    docs = search_chromadb(chat_query, collection, cohere_client, top_k=3)
                    
                    if docs:
                        # Generate answer
                        answer = generate_answer(cohere_client, chat_query, docs)
                        
                        # Add to chat history
                        st.session_state.chat_history.append(("You", chat_query))
                        st.session_state.chat_history.append(("Bot", answer))
                    else:
                        st.warning("No relevant documents found for your query.")
                        st.session_state.chat_history.append(("You", chat_query))
                        st.session_state.chat_history.append(("Bot", "No relevant documents found."))
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for speaker, text in st.session_state.chat_history:
                    if speaker == "You":
                        st.markdown(f"**🧑 {speaker}:**\n\n{text}")
                    else:
                        st.markdown(f"**🤖 {speaker}:**\n\n{text}")
                    st.divider()
    
    with tab3:
        st.header("About This Application")
        st.markdown("""
        ### RAG Chatbot Features
        
        This application combines **Retrieval-Augmented Generation (RAG)** with ChromaDB and Cohere API to:
        
        1. **📤 Upload PDFs** - Upload PDF documents to the system
        2. **🔍 Search** - Search through document content with semantic search
        3. **💬 Chat** - Ask questions and get answers based on your documents
        4. **📊 Manage** - View statistics about your uploaded documents
        
        ### How It Works
        
        1. **PDF Processing**: PDFs are uploaded and text is extracted
        2. **Embedding Generation**: Cohere API generates semantic embeddings
        3. **Storage**: Documents and embeddings are stored in ChromaDB
        4. **Search**: New queries generate embeddings and find similar documents
        5. **Generation**: Cohere Chat API generates answers based on retrieved context
        
        ### Technologies Used
        
        - **Streamlit**: Web interface
        - **ChromaDB**: Vector database for document storage
        - **Cohere API**: Embeddings and text generation
        - **PyPDF2**: PDF text extraction
        
        ### Tips
        
        - Upload multiple PDFs to create a knowledge base
        - Use natural language questions for best results
        - The search feature shows relevance scores (0-1)
        - Chat history is maintained in the session
        """)
        
        st.divider()
        st.info("Created for RAG (Retrieval-Augmented Generation) applications")


if __name__ == "__main__":
    main()
