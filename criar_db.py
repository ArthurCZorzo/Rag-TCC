import os
import shutil
import re

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

def reestruturar_tabela_matriz_curricular(texto: str) -> str:
    """
    Divide a grande tabela de Componentes Curriculares Obrigatórios por Fase/Semestre,
    repetindo o cabeçalho completo em cada bloco e adicionando cabeçalho H3 explícito.
    Isso garante que cada fase gere chunks fechados e com semântica completa.
    """
    padrao = r'(## Componentes curriculares obrigatórios\s*\n\n\|[^\n]+\|\n\|[-| ]+\|\n)([\s\S]*?)(?=\n## |\Z)'
    match = re.search(padrao, texto)
    if not match:
        return texto
    
    cabecalho_base = match.group(1).strip()
    corpo_tabela = match.group(2).strip()
    
    linhas_header = cabecalho_base.split('\n')
    header_colunas = linhas_header[1]
    header_separador = linhas_header[2]
    
    fases = {}
    fases_ordem = []
    
    for line in corpo_tabela.split('\n'):
        line_clean = line.strip()
        if not line_clean.startswith('|'):
            continue
        parts = [p.strip() for p in line_clean.split('|')]
        if len(parts) >= 3 and parts[2].isdigit():
            fase_num = int(parts[2])
            if fase_num not in fases:
                fases[fase_num] = []
                fases_ordem.append(fase_num)
            fases[fase_num].append(line_clean)
        else:
            if 999 not in fases:
                fases[999] = []
                fases_ordem.append(999)
            fases[999].append(line_clean)
            
    nova_secao = "## Componentes curriculares obrigatórios por Fase (Matriz Curricular)\n\n"
    for f in fases_ordem:
        if f != 999:
            nova_secao += f"### Matriz Curricular - {f}ª Fase ({f}º Semestre) - Componentes Curriculares Obrigatórios\n\n"
            nova_secao += f"{header_colunas}\n{header_separador}\n"
            for row in fases[f]:
                nova_secao += f"{row}\n"
            nova_secao += "\n"
        else:
            nova_secao += "### Resumo da Carga Horária dos Componentes Curriculares Obrigatórios\n\n"
            nova_secao += f"{header_colunas}\n{header_separador}\n"
            for row in fases[f]:
                nova_secao += f"{row}\n"
            nova_secao += "\n"
            
    texto_atualizado = texto[:match.start()] + nova_secao + texto[match.end():]
    return texto_atualizado

def normalizar_headers_markdown(texto: str) -> str:
    """ 
    1. Reestrutura a matriz curricular geral por Fase/Semestre preservando o cabeçalho da tabela.
    2. Rebaixa as subseções de ementa (Objetivos, Conteúdos, Metodologia, Bibliografia) para nível ### 
       para que herdem o contexto de ## Disciplina: NOME no MarkdownHeaderTextSplitter.
    3. Transforma itens numerados de seções (ex: '- 23.Objetivos do curso:') em cabeçalhos (## 23. Objetivos do curso:)
    4. Transforma a linha de início de tabela de disciplina em um cabeçalho oficial do Markdown (## Disciplina: NOME)
    """
    # 1. Reestrutura tabela geral de disciplinas da matriz curricular
    texto = reestruturar_tabela_matriz_curricular(texto)

    # 2. Transforma subseções de ementa de disciplinas em nível 3 (###)
    subsecoes = [
        'Objetivos:', 'Conteúdos:', 'Metodologia de Abordagem:', 
        'Metodologia de abordagem:', 'Bibliografia Básica:', 
        'Bibliografia Complementar:', 'Bibliografia básica:', 
        'Bibliografia complementar:', 'Observações:'
    ]
    escaped_subsecoes = '|'.join([re.escape(s) for s in subsecoes])
    padrao_subsecoes = r'^(?:-\s*)?##?\s*(' + escaped_subsecoes + r')'
    texto = re.sub(padrao_subsecoes, r'### \1', texto, flags=re.MULTILINE)

    # 3. Transforma linhas como '- 23.Objetivos do curso:' em '## 23. Objetivos do curso:'
    texto = re.sub(r'^(?:-\s*)?(\d{1,2}\.\s*[A-ZÀ-Ú].*?:)', r'## \1', texto, flags=re.MULTILINE)
    
    # 4. Transforma '| Unidade Curricular / Disciplina: Algoritmos e Programação I | ... |'
    # em '## Disciplina: Algoritmos e Programação I\n| Unidade Curricular / Disciplina: ...'
    def converter_tabela_disciplina(match):
        nome_disciplina = match.group(1).strip()
        return f"\n\n## Disciplina: {nome_disciplina}\n{match.group(0)}"
        
    texto = re.sub(r'^\|\s*Unidade\s*Curricular\s*/\s*Disciplina:\s*([^\n|]+).*$', converter_tabela_disciplina, texto, flags=re.MULTILINE)
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

def extrair_nome_disciplina(texto_chunk: str, h1: str, h2: str, h3: str) -> str:
    """ Procura o nome da disciplina tanto no conteúdo do chunk quanto nos Headers do Markdown """
    match = re.search(r'Unidade\s*Curricular\s*/\s*Disciplina:\s*([^\n|]+)', texto_chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    for h in [h2, h3, h1]:
        if "Disciplina:" in h:
            return h.replace("Disciplina:", "").strip()
            
    return ""

def eh_tabela_disciplina(texto_chunk: str) -> bool:
    """ Verifica se o chunk contém a estrutura de tabela de ementa / carga horária / professor """
    tem_tabela = "|" in texto_chunk and ("CH Total:" in texto_chunk or "CH EAD:" in texto_chunk or "Unidade Curricular" in texto_chunk)
    tem_prof = "Professor Responsável:" in texto_chunk or "CH Extensão:" in texto_chunk
    return tem_tabela or tem_prof

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

            # Tenta extrair o nome da disciplina associada
            disciplina = extrair_nome_disciplina(chunk.page_content, h1, h2, h3)
            e_tabela = eh_tabela_disciplina(chunk.page_content)
            
            headers_validos = [h for h in [h1, h2, h3] if h]
            contexto_secao = " > ".join(headers_validos) if headers_validos else "Geral"

            cabecalho_contexto = f"Documento: {source} | Seção: {contexto_secao}"
            if disciplina:
                cabecalho_contexto += f" | Disciplina: {disciplina}"
            if e_tabela:
                cabecalho_contexto += " | [Tabela/Ementa de Disciplina]"

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

if __name__ == "__main__":
    criar_db()
