from llama_index.core import VectorStoreIndex, StorageContext,SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding




def main():
    # Load docs and Store with 
    docs = SimpleDirectoryReader("data").load_data()
    Settings.llm = Ollama(model="llama3.2:1b", request_timeout=360.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    index = VectorStoreIndex.from_documents( 
        docs 
    )
    engine = index.as_query_engine()
    response = engine.query("am I married?")
    print(response)



if __name__ == "__main__":
    main()
