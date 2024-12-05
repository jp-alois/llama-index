from llama_index.core import VectorStoreIndex, StorageContext,SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
import os
import chromadb


# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise ValueError("OpenAI API key not found. Make sure it's set in the .env file.")


def main():
    # Create Chroma Client
    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.create_collection("new-5")
    vector_store = ChromaVectorStore(chroma_collection= chroma_collection)
    storage_context =  StorageContext.from_defaults(vector_store=vector_store)

    # Load docs and Store with 
    docs = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context
    )

    engine = index.as_query_engine(similarity_top_k=5)
    response = engine.query(" what is hiest degree of Him? What is relatations of Raman and Jignesh, and")
    print(response)



if __name__ == "__main__":
    main()
