import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
#from langchain_chroma.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

PASTA_BASE = "docs"
ARQUIVO_CHUNKS = "chunks_gerados.txt"

def criar_db():
    documentos = carregar_documentos()
    chunks = criar_chunks(documentos)
    salvar_chunks_em_arquivo(chunks, ARQUIVO_CHUNKS)
    vetorizar_chunks(chunks)

def salvar_chunks_em_arquivo(chunks, caminho_arquivo):
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        f.write(f"TOTAL DE CHUNKS GERADOS: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "Desconhecido")
            tamanho = len(chunk.page_content)
            
            f.write(f"=== CHUNK {i} / {len(chunks)} ===\n")
            f.write(f"Fonte: {source}\n")
            f.write(f"Tamanho: {tamanho} caracteres\n")
            f.write(f"Metadados: {chunk.metadata}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{chunk.page_content}\n")
            f.write("=" * 80 + "\n\n")
            
    print(f"Chunks salvos com sucesso em '{caminho_arquivo}'!")

def verificar_markdown(arquivo):
    return arquivo.endswith("md")

""" def carregar_documentos():
    carregador = DirectoryLoader(
        PASTA_BASE, 
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
        )
    documentos = carregador.load()
    return documentos """

def carregar_documentos():

    documentos = []

    for root, _, files in os.walk(PASTA_BASE):
        for filename in files:
            caminho_arquivo = os.path.join(root, filename)

            if os.path.isfile(caminho_arquivo) and verificar_markdown(filename):
                try:
                    carregador = DoclingLoader(
                            file_path=caminho_arquivo,
                            export_type=ExportType.MARKDOWN
                    )
                    documento = carregador.load()
                    documento[0].metadata["source"] = filename
                    documentos.append(documento)
                except Exception as ex:
                    print(f"Erro ao carregar documento {filename}, erro: {ex}")

    return documentos

def criar_chunks(documentos):
    chunks_documentos = []
    separador = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        add_start_index=True
    )
    
    for documento in documentos:
        chunks = separador.split_documents(documento)

        for chunk in chunks:
            chunk.page_content = "Conteúdo do documento: " + documento[0].metadata["source"] + ": " + chunk.page_content

        chunks_documentos.extend(chunks)

    return chunks_documentos

""" def criar_chunks(documentos):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    separador_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    chunks_finais = []

    for doc in documentos:
        chunks_markdown = separador_md.split_text(doc.page_content)
        
        for chunk in chunks_markdown:
            chunk.metadata.update(doc.metadata)

        separador_recursivo = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        sub_chunks = separador_recursivo.split_documents(chunks_markdown)
        chunks_finais.extend(sub_chunks)

    return chunks_finais """

def vetorizar_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    db = Chroma.from_documents(
        chunks, 
        embedding=embeddings, 
        persist_directory="database"
    )
    db.persist()
    print("Banco de dados criado!")

criar_db()