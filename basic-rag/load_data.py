import os
from dotenv import load_dotenv

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Azure AI Inference configuration
AZURE_AI_INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_MODEL_TOKEN = os.environ["EMBEDDING_MODEL_GITHUB_TOKEN"]  # Use your actual GitHub Marketplace token

# Predefined Path
DATA_DIR = "./weather_data"
PERSISTED_DIR = "./chroma_store"
COLLECTION_NAME="WEATHER"

# Initialize Azure Inference client
client = EmbeddingsClient(
    endpoint=AZURE_AI_INFERENCE_ENDPOINT,
    credential=AzureKeyCredential(EMBEDDING_MODEL_TOKEN)
)


# Custom wrapper for LangChain embedding interface
class AzureInferenceOpenAIEmbeddings(Embeddings):
    def embed_documents(self, texts):
        response = client.embed(input=texts, model=EMBEDDING_MODEL_NAME)
        return [item["embedding"] for item in response["data"]]

    def embed_query(self, text):
        response = client.embed(input=[text], model=EMBEDDING_MODEL_NAME)
        return response["data"][0]["embedding"] # type: ignore[attr-defined]

# Load all .txt files from a folder
def load_text_files(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

# Split text into manageable chunks
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return split_docs

def load_data_to_vector_store():

    # Load and split docs
    raw_docs = load_text_files(DATA_DIR)
    split_docs = split_documents(raw_docs)

    # Initialize embedding model
    embedding_model = AzureInferenceOpenAIEmbeddings()

    # Store in Chroma
    vector_store : Chroma = Chroma.from_documents(
        collection_name=COLLECTION_NAME,
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=PERSISTED_DIR
    )

    print(f"✅ Vector store created and saved to: {PERSISTED_DIR}")


def search_data_in_vector_store(query):

    # Initialize embedding model
    embedding_model = AzureInferenceOpenAIEmbeddings()

    # Reinitialize for search
    vector_store = Chroma(
        persist_directory=PERSISTED_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )

    results = vector_store.similarity_search(query, k=1)
    for i, r in enumerate(results, 1):
        print(f"Result {i}: {r.page_content} [source: {r.metadata.get('source')}]")

if __name__ == "__main__":
   
   load_data_to_vector_store()

   query = "What's the maximum temperature logged in the month of june,2022?"
   search_data_in_vector_store(query=query)