import getpass
import os
import json
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai 
from dotenv import load_dotenv
import shutil

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if "LANGCHAIN_API_KEY" not in os.environ:
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("langchain api key here")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("open ai api key here")

llm = ChatOpenAI(
    model = "gpt-3.5-turbo-0125", 
    temperature = 0.1,
    max_tokens = 256
)

CHROMA_PATH = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    json_paths = [
        r"path to guesses.json",
        r"path to Wordle_solution.json"
    ]
    documents = []
    for path in json_paths:
        documents.extend(load_json_documents(path))
    return documents

def load_json_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        documents = []
        for item in data:
            guesses = {key: value for key, value in item.items() if key.startswith("guess") and value}
            content = "\n".join([f"{key}: {value}" for key, value in guesses.items()])
            documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
    else:
        document = chunks[0] if chunks else None

    if document:
        print(document.page_content)
        print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
