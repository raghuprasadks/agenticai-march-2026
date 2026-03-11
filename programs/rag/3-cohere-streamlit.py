import chromadb
import cohere
import streamlit as st
import os

# Cohere API key (hardcoded for demo; use secrets in production)
api_key = "yGlhFfYaems7Qn25x5DYVa2eS4NiLz6Bzuh5aXyc"

# Initialize clients
cohere_client = cohere.Client(api_key)
client = chromadb.PersistentClient()
collection = client.get_or_create_collection(name="rag_collection_pdfs")

def get_relevant_docs(collection, cohere_client, query, top_k=3):
    # Embed the query
    query_emb = cohere_client.embed(texts=[query]).embeddings[0]
    # Query ChromaDB for similar documents
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    docs = results['documents'][0] if results['documents'] else []
    return docs

def generate_answer(cohere_client, query, docs):
    context = "\n".join(docs)
    message = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = cohere_client.chat(
        message=message,
        max_tokens=256,
        temperature=0.5
    )
    return response.text.strip()

# Streamlit app
st.title("RAG Chatbot with Cohere and ChromaDB")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        docs = get_relevant_docs(collection, cohere_client, query)
        answer = generate_answer(cohere_client, query, docs)
        
        st.subheader("Relevant Documents:")
        for i, doc in enumerate(docs, 1):
            st.write(f"{i}. {doc}")
        
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")