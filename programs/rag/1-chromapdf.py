import chromadb
import cohere
import PyPDF2
import os
#import dotenv
#from dotenv import load_dotenv
#load_dotenv(dotenv_path="../.env")  # Adjust path if needed

#load_dotenv()
#api_key = os.getenv("cohere_api_key")
api_key = "yGlhFfYaems7Qn25x5DYVa2eS4NiLz6Bzuh5aXyc"

def extract_text_from_pdf(pdf_file):
    # Extract text from a PDF file-like object
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Step 1: Extract text from PDF
pdf_path = "aboutsvit.pdf"
with open(pdf_path, "rb") as f:
    pdf_text = extract_text_from_pdf(f)

# Step 2: Generate embedding using Cohere
cohere_api_key = api_key
co = cohere.Client(cohere_api_key)
embedding = co.embed(texts=[pdf_text]).embeddings[0]

# Step 3: Store in ChromaDB
client = chromadb.PersistentClient()
collection = client.get_or_create_collection(name="rag_collection_pdfs")

collection.add(
    ids=["pdf1"],  # Unique ID for your PDF
    documents=[pdf_text],
    embeddings=[embedding]
)
print("PDF text and embedding added to ChromaDB collection 'rag_collection_pdfs'.")