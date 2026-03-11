import streamlit as st
import cohere
"""
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="../.env")  # Adjust path if needed
load_dotenv()
api_key = os.getenv("cohere_api_key")
"""
api_key = "yGlhFfYaems7Qn25x5DYVa2eS4NiLz6Bzuh5aXyc"
if not api_key:
    st.error("API Key not found. Check your .env file and path.")
    st.stop()

co = cohere.ClientV2(api_key=api_key)

st.title("AIT Chikkamagalore GENAI SDP Chatbot using Cohere")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_input
            }
        ]
    })

    response = co.chat(
        messages=st.session_state.chat_history,
        temperature=0.3,
        model="command-a-03-2025",
    )
    assistant_reply = response.message.content[0].text
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": assistant_reply
            }
        ]
    })

# Display chat history

for msg in st.session_state.chat_history:
    #print("msg##",msg['message']['content'][0]['text'])
    role = msg["role"].capitalize()
    text = msg["content"][0]["text"]
    #llm_output['message']['content'][0]['text']
    st.markdown(f"**{role}:** {text}")