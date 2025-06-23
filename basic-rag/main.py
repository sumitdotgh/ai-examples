import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# GPT 4.1 model configuration
GITHUB_INFERENCE_ENDPOINT = "https://models.github.ai/inference"
GPT_4_1_MODEL_NAME = "openai/gpt-4.1"
GPT_4_1_MODEL_TOKEN = os.environ["GPT_4_1_MODEL_GITHUB_TOKEN"]

# Azure AI Inference configuration
AZURE_AI_INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_MODEL_TOKEN = os.environ["EMBEDDING_MODEL_GITHUB_TOKEN"]  # Use your actual GitHub Marketplace token

# Predefined Path
DATA_DIR = "./weather_data"
PERSISTED_DIR = "./chroma_store"
COLLECTION_NAME="WEATHER"

#Step 1: Set up LLM (ChatOpenAI)
llm = ChatOpenAI(base_url=GITHUB_INFERENCE_ENDPOINT, api_key=GPT_4_1_MODEL_TOKEN, model=GPT_4_1_MODEL_NAME) # type: ignore

# Optional: Custom Prompt
prompt_template = """
You are a helpful assistant answering questions based on weather records.

Context:
{context}

Question: {question}
Answer in 1-2 sentences based only on the above data.
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Embedding client wrapper
client = EmbeddingsClient(endpoint=AZURE_AI_INFERENCE_ENDPOINT, credential=AzureKeyCredential(EMBEDDING_MODEL_TOKEN))

class AzureInferenceOpenAIEmbeddings(Embeddings):
    def embed_documents(self, texts):
        response = client.embed(input=texts, model=EMBEDDING_MODEL_NAME)
        return [item["embedding"] for item in response["data"]]

    def embed_query(self, text):
        response = client.embed(input=[text], model=EMBEDDING_MODEL_NAME)
        return response["data"][0]["embedding"]


# Step 1: Load Chroma vector store
embedding_model = AzureInferenceOpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory=PERSISTED_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
)

# Step 3: Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": prompt}
)

# Step 4: Ask questions
def ask(question: str):
    print(f"🔍 Question: {question}")
    response = qa_chain.invoke({"query": question})
    print(f"💡 Answer: {response}")

if __name__ == "__main__":
    ask("What was the maximum temperature in June 2022?")    