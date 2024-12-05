from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise ValueError("OpenAI API key not found. Make sure it's set in the .env file.")


def main():
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    answer  = query_engine.query("What is my higest degree?")
    print(answer)


if __name__ == "__main__":
    main()

