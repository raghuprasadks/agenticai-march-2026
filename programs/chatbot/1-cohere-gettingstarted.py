#pip install cohere python-dotenv
import cohere
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="../.env")  # Adjust path if needed
api_key = os.getenv("cohere_api_key")
#api_key = "yGlhFfYaems7Qn25x5DYVa2eS4NiLz6Bzuh5aXyc"
print(f"API Key: {api_key}")

co = cohere.ClientV2(api_key=api_key)

response = co.chat(


     messages=[
        {
            "role": "system",
            "content": "I am an helpful assistant. What can i do for you"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "who are you"
                }
            ]
        }
    ],
    temperature=0.3,
    model="command-a-03-2025",
)

#print(response)

#print(type(response))
#print(dir(response))
assistant_reply = response.message.content[0].text
print("-----Assistant Reply-----")
print(assistant_reply)
