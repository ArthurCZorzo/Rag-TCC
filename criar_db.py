import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
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

import re

def normalizar_headers_markdown(texto: str) -> str:
    """ 
    Transforma itens numerados de seções (ex: '- 23.Objetivos do curso:' ou '23. Objetivos do curso:') 
    em cabeçalhos de Markdown (## 23. Objetivos do curso:) para que sejam reconhecidos pelo Splitter.
    """
    # Transforma linhas como '- 23.Objetivos do curso:' em '## 23. Objetivos do curso:'
    texto = re.sub(r'^(?:-\s*)?(\d{1,2}\.\s*[A-ZÀ-Ú].*?:)', r'## \1', texto, flags=re.MULTILINE)
    return texto

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
                    docs_carregados = carregador.load()
                    for doc in docs_carregados:
                        doc.metadata["source"] = filename
                        doc.page_content = normalizar_headers_markdown(doc.page_content)
                        documentos.append(doc)
                except Exception as ex:
                    print(f"Erro ao carregar documento {filename}, erro: {ex}")

    return documentos

def extrair_nome_disciplina(texto_chunk: str) -> str:
    """ Procura no conteúdo do chunk se existe uma disciplina associada (ex: Unidade Curricular / Disciplina: Planejamento Estratégico) """
    match = re.search(r'Unidade\s*Curricular\s*/\s*Disciplina:\s*([^\n|]+)', texto_chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def criar_chunks(documentos):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    separador_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    separador_recursivo = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )

    chunks_finais = []

    for doc in documentos:
        # Fatiar primeiramente pela estrutura do Markdown (headers)
        chunks_markdown = separador_md.split_text(doc.page_content)
        
        for chunk_md in chunks_markdown:
            chunk_md.metadata.update(doc.metadata)

        # Aplica o splitter recursivo caso as seções sejam grandes
        sub_chunks = separador_recursivo.split_documents(chunks_markdown)

        for chunk in sub_chunks:
            source = chunk.metadata.get("source", "")
            
            # Recupera os títulos das seções
            h1 = chunk.metadata.get("Header 1", "")
            h2 = chunk.metadata.get("Header 2", "")
            h3 = chunk.metadata.get("Header 3", "")

            # Tenta extrair disciplina se for uma ementa de disciplina
            disciplina = extrair_nome_disciplina(chunk.page_content)
            
            headers_validos = [h for h in [h1, h2, h3] if h]
            contexto_secao = " > ".join(headers_validos) if headers_validos else "Geral"

            cabecalho_contexto = f"Documento: {source} | Seção: {contexto_secao}"
            if disciplina:
                cabecalho_contexto += f" | Disciplina: {disciplina}"

            # Adiciona o prefixo exigido pelo modelo intfloat/multilingual-e5-large + contexto enriquecido
            chunk.page_content = f"passage: {cabecalho_contexto}\n{chunk.page_content}"
            chunks_finais.append(chunk)

    return chunks_finais


def vetorizar_chunks(chunks):
    pasta_db = "database"
    # Garante que a base anterior seja removida para evitar mistura de chunks antigos
    if os.path.exists(pasta_db):
        shutil.rmtree(pasta_db)
        print("Banco de dados anterior removido.")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    db = Chroma.from_documents(
        chunks, 
        embedding=embeddings, 
        persist_directory=pasta_db
    )
    db.persist()
    print("Novo banco de dados criado com sucesso!")

criar_db()
