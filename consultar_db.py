from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import openai
import sqlite3
import os
import time
from resposta import Resposta

load_dotenv()

conn = sqlite3.connect("benchmark.db")
cursor = conn.cursor()

def criar_banco_benchmark():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS respostas(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelo TEXT NOT NULL,
            pergunta_usuario TEXT NOT NULL,
            resposta_modelo TEXT NOT NULL,
            token_input INTEGER,
            token_output INTEGER,
            total_tokens INTEGER,
            tempo_inicial REAL,
            tempo_final REAL,
            tempo_total REAL,
            chunks TEXT NOT NULL
        )
    """)

def criar_client(modelo):
    if modelo == "sabia-4":
        return openai.OpenAI(
            api_key=os.getenv("MARITACA_AI_API_KEY"),
            base_url="https://chat.maritaca.ai/api",
        )
    else:
        return openai.OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

def escolher_modelo():
    print("1 - gpt-5.6-luna")
    print("2 - gemini-3.7-flash")
    print("3 - claude-sonnet-5")
    print("4 - deepseek-v4.1-flash")
    print("5 - MaritacaAI Sabia 4")
    
    opcao = input("\nEscolha um modelo: ")
    
    if opcao == "1":
        return "openai/gpt-5.6-luna" #1.05 M contexto, $0.20/M input tokens, $1.20/M output tokens
    elif opcao == "2":
        return "google/gemini-3.7-flash" #1.05 M contexto, $0.75/M input tokens, $3.75/M output tokens
    elif opcao == "3":
        return "anthropic/claude-sonnet-5" #1 M contexto, $2/M input tokens, $10/M output tokens
    elif opcao == "4":
        return "deepseek/deepseek-v4.1-flash" #1.05 M contexto, $0.0198/M input tokens, #0.396/M output tokens
    elif opcao == "5":
        return "sabia-4"
    else:
        print("\nOpção inválida.")
        return "openai/gpt-5.6-luna"
    

def consultar_modelo(contexto, pergunta, client, modelo):
  prompt = f"""
    Prompt do sistema:
    Você é um assistente acadêmico institucional. Sua função é responder às dúvidas dos estudantes com clareza e precisão.

    Instruções:
    1. Priorize as informações do [CONTEXTO] fornecido para responder.
    2. Você pode complementar a resposta com seu conhecimento geral apenas se for para esclarecer ou enriquecer o contexto, sem contradizê-lo.
    3. Se a informação não estiver disponível no [CONTEXTO] e você não tiver certeza absoluta, NÃO invente nada. Responda claramente que não possui essa informação e oriente o estudante a procurar a coordenação do curso.
    4. Mantenha um tom cortês, objetivo e institucional.

    [CONTEXTO]
    {contexto}

    [PERGUNTA DO ESTUDANTE]
    {pergunta}
    """.strip()

  if modelo == "sabia-4":   
        return client.responses.create(
            model=modelo,
            input=prompt,
        )
  else:
        return client.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
def inicializar_sistema_busca(diretorio_banco: str = "database"):
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    
    db = Chroma(
        persist_directory=diretorio_banco,
        embedding_function=embeddings
    )
    
    return db

def buscar_top_chunks(banco_dados: Chroma, pergunta: str, k: int = 5):
    # O modelo intfloat/multilingual-e5-large exige o prefixo 'query: ' nas buscas
    pergunta_formatada = f"query: {pergunta}"
    resultados = banco_dados.similarity_search(pergunta_formatada, k=k)
    return resultados

def tratar_resposta_sabia(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, chunks):
    resposta = Resposta(
        resposta_modelo.model,
        pergunta,
        resposta_modelo.output[0].content[0].text,
        resposta_modelo.usage.input_tokens,
        resposta_modelo.usage.output_tokens,
        resposta_modelo.usage.total_tokens,
        tempo_inicial,
        tempo_final,
        tempo_total
    )

    custo = (resposta.token_input * 5.00 / 1000000) + (resposta.token_output * 20 / 1000000)

    print(resposta.resposta_modelo)
    salvar_resposta_banco(resposta, chunks)
    return resposta

def tratar_resposta_openrouter(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, chunks):
    resposta = Resposta(
        resposta_modelo.model,
        pergunta,
        resposta_modelo.choices[0].message.content,
        resposta_modelo.usage.prompt_tokens,
        resposta_modelo.usage.completion_tokens,
        resposta_modelo.usage.total_tokens,
        tempo_inicial,
        tempo_final,
        tempo_total
    )

    print(resposta.resposta_modelo)
    salvar_resposta_banco(resposta, chunks)
    return resposta

def salvar_resposta_banco(resposta, chunks):
    with conn:
        cursor.execute("INSERT INTO respostas (modelo, pergunta_usuario, resposta_modelo, token_input, token_output, total_tokens, tempo_inicial, tempo_final, tempo_total, chunks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ", 
            (resposta.modelo, resposta.pergunta_usuario, resposta.resposta_modelo, resposta.token_input, resposta.token_output, 
             resposta.total_tokens, resposta.tempo_inicial, resposta.tempo_final, resposta.tempo_total, chunks))

if __name__ == "__main__":
    criar_banco_benchmark()
    modelo = escolher_modelo()
    client = criar_client(modelo)

    db_conectado = inicializar_sistema_busca(diretorio_banco="database")

    rodar = True
    while rodar:
        pergunta = input("Digite sua pergunta: ")

        if pergunta == "Sair":
            print("Saindo...")
            rodar = False
        else:
            chunks_encontrados = buscar_top_chunks(db_conectado, pergunta, k=5)

            textos_limpos = []
            for doc in chunks_encontrados:
                texto_sem_quebras = doc.page_content.replace("\n", " ")
                texto_ajustado = " ".join(texto_sem_quebras.split())
                textos_limpos.append(texto_ajustado)

            texto_consolidado = "\n\n".join(textos_limpos)

            tempo_inicial = time.perf_counter()
            resposta_modelo = consultar_modelo(texto_consolidado, pergunta, client, modelo)
            tempo_final = time.perf_counter()

            tempo_total = tempo_final - tempo_inicial

            #print(f"Resposta do Modelo: {resposta_modelo}")

            if modelo == "sabia-4":
                tratar_resposta_sabia(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, texto_consolidado)
            else:
                tratar_resposta_openrouter(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, texto_consolidado)
