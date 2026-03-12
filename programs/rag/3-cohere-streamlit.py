import streamlit as st
import chromadb
import cohere
import os
#from dotenv import load_dotenv
#load_dotenv(dotenv_path="../.env")  # Adjust path if needed

#api_key = os.getenv("cohere_api_key")
api_key ="3u0H4lmf6YYOFyMTosFyRipuj1zj6VzHr34EeB0w"

def get_relevant_docs(collection, cohere_client, query, top_k=3):
    query_emb = cohere_client.embed(texts=[query]).embeddings[0]
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

def main():
    st.title("RAG Chatbot with ChromaDB & Cohere")
    cohere_api_key = api_key
    cohere_client = cohere.Client(cohere_api_key)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(name="rag_collection_pdfs")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    user_query = st.text_input("You:", value=st.session_state.user_query, key="user_query")
    ask_clicked = st.button("Ask")

    if ask_clicked and user_query:
        docs = get_relevant_docs(collection, cohere_client, user_query)
        answer = generate_answer(cohere_client, user_query, docs)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", answer))


    st.header("Chat History")
    for speaker, text in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {text}")

if __name__ == "__main__":
    main()