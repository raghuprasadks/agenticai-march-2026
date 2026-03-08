import cohere
#from dotenv import load_dotenv
#import os

#load_dotenv(dotenv_path="../.env")  # Adjust path if needed

#api_key = os.getenv("cohere_api_key")
api_key = "yGlhFfYaems7Qn25x5DYVa2eS4NiLz6Bzuh5aXyc"
if not api_key:
    print("API Key not found. Check your .env file and path.")
    exit(1)

co = cohere.ClientV2(api_key=api_key)

print("Welcome to the Cohere Chatbot! Type 'exit' or 'quit' to stop.")

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    chat_history.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_input
            }
        ]
    })

    response = co.chat(
        messages=chat_history,
        temperature=0.3,
        model="command-a-03-2025",
    )
    #print(f"Response Object: {response}")
    assistant_reply = response.message.content[0].text
    print(f"Assistant: {assistant_reply}")

    chat_history.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": assistant_reply
            }
        ]
    })
    print("*****")
    print(f"Chat History: {chat_history}")