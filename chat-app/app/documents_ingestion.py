import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def get_vector_db(source_docs_folder = "chat-app/docs"):
    documents = []
    # Create a List of Documents from all of our files in the ./docs folder
    for file in os.listdir(source_docs_folder):
        if file.endswith(".pdf"):
            pdf_path = source_docs_folder + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = source_docs_folder + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = source_docs_folder + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    # Split the documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)

    # Convert the document chunks to embedding and save them to the vector store
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
    vectordb.persist()

    return vectordb