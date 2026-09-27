from langchain_community.document_loaders import PyPDFDirectoryLoader
from docling.document_converter import DocumentConverter
import os

PASTA_BASE = "base"

def criar_pasta_docs():
    caminho = "docs"

    if os.path.isdir(caminho) == False:
        os.makedirs(caminho)

def criar_markdown():
    caminho_pasta = "base/"
    caminho_documentos = []

    for raiz, diretorios, arquivos in os.walk(caminho_pasta):
        for arquivo in arquivos:
            caminho_completo = os.path.join(raiz, arquivo)
            caminho_documentos.append(caminho_completo)

    for c in caminho_documentos:
        nome = c
        converter = DocumentConverter()
        result = converter.convert(nome)
        
        doc = result.document

        markdown_output = doc.export_to_markdown()

        print(f"NOME: {nome}")
        
        with open(nome.replace("base/", "docs/").replace(".pdf", ".md"), "w", encoding="utf-8") as f:
            f.write(markdown_output)
        
            print("Conversão concluída. e salva em:", markdown_output)

criar_pasta_docs()
criar_markdown()