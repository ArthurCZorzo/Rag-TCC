from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import re

load_dotenv()

PASTA_BASE = "base"

def criar_db():
    documentos = carregar_documentos()
    chunks = criar_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()
    return documentos

def criar_chunks(documentos):
    # chunk_size=500,
    # chunk_overlap=100,
    separador = RecursiveCharacterTextSplitter(
        length_function=len,
        add_start_index=True
    )
    chunks = separador.split_documents(documentos)
    return chunks

def vetorizar_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="database"
    )
    print("Banco de dados criado!")

criar_db()