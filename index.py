from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os

# Load environment variables from .env file
load_dotenv()


def main():
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    print(index)


if __name__ == "__main__":
    main()