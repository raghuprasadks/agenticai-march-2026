# pip install langchain langchain-community cohere
# pip install python-dotenv

import os
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Cohere
from dotenv import load_dotenv


load_dotenv(dotenv_path="../.env")  # Adjust path if needed

api_key = os.getenv("cohere_api_key")


# Set your Cohere API key
os.environ["COHERE_API_KEY"] = api_key


# Path to your local file
file_path = "sample.txt"  # Change this to the correct local path if needed

# Read the file content
try:
    with open(file_path, "r", encoding="utf-8") as file:
        document_text = file.read()

    print("📄 File loaded successfully!\n")
    print(document_text[:500])  # Show first 500 characters

except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    exit(1)
except Exception as e:
    print(f"⚠️ An error occurred: {e}")
    exit(1)

# Load Cohere model
llm = Cohere(model="command", cohere_api_key=os.environ["COHERE_API_KEY"])

# Define prompt template
prompt = PromptTemplate(
    input_variables=["input_text"],
    template="Summarize this text:\n\n{input_text}"
)

# Format and generate response
formatted_prompt = prompt.format(input_text=document_text)
response = llm.invoke(formatted_prompt)

print("\n📝 Summary:\n", response)
