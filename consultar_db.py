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
            versao INTEGER
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
    print("1 - GPT-4.1 Mini")
    print("2 - Gemini 2.5 Flash")
    print("3 - Claude Sonnet 4")
    print("4 - DeepSeek Chat")
    print("5 - MaritacaAI Sabia 4")
    
    opcao = input("\nEscolha um modelo: ")
    
    if opcao == "1":
        return "openai/gpt-4.1-mini"
    elif opcao == "2":
        return "google/gemini-2.5-flash"
    elif opcao == "3":
        return "anthropic/claude-sonnet-4"
    elif opcao == "4":
        return "deepseek/deepseek-chat"
    elif opcao == "5":
        return "sabia-4"
    else:
        print("\nOpção inválida.")
        return "openai/gpt-4.1-mini"
    

def consultar_modelo(contexto, pergunta, client, modelo):
  prompt = f"""
    Prompt do sistema:
    Você é um assistente especializado em responder perguntas com base em documentos do Instituto Federal de Santa Catarina Câmpus Caçador. Sua tarefa é utilizar o contexto e seu conhecimento para auxiliar o usuário solucionar sua dúvida. Por favor, responda listando todas as informações relevantes encontradas no contexto de forma sucinta e profissional. Se não souber, diga "Não sei".
    
    Prompt do usuário:
    Contexto:
    {contexto}
    Pergunta:
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
                    "content": pergunta
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

def tratar_resposta_sabia(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total):
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

    print("*"*20)
    print(f"Pergunta: {resposta.pergunta_usuario}")
    print("*"*20)
    print("Reposta do modelo: ")
    print(resposta.resposta_modelo)
    print("*"*20)
    print(f"Modelo: {resposta.modelo}")
    print("*"*20)
    print("Tokens:")
    print(f"Input tokens: {resposta.token_input}")
    print(f"Output tokens: {resposta.token_output}")
    print(f"Total tokens: {resposta.total_tokens}")
    print("*"*20)
    print("Tempo: ")
    print(f"Tempo Inicial: {resposta.tempo_inicial}")
    print(f"Tempo Final: {resposta.tempo_final}")
    print(f"Tempo Total: {resposta.tempo_total}")
    print("*"*20)
    print(f"Custo: R$ {custo}")
    print("*"*20)
    #salvar_resposta_banco(resposta)
    return resposta

def tratar_resposta_openrouter(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total):
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
    print("*"*20)
    print(f"Pergunta: {resposta.pergunta_usuario}")
    print("*"*20)
    print("Reposta do modelo: ")
    print(resposta.resposta_modelo)
    print("*"*20)
    print(f"Modelo: {resposta.modelo}")
    print("*"*20)
    print("Tokens:")
    print(f"Input tokens: {resposta.token_input}")
    print(f"Output tokens: {resposta.token_output}")
    print(f"Total tokens: {resposta.total_tokens}")
    print("*"*20)
    print("Tempo: ")
    print(f"Tempo Inicial: {resposta.tempo_inicial}")
    print(f"Tempo Final: {resposta.tempo_final}")
    print(f"Tempo Total: {resposta.tempo_total}")
    print("*"*20)
    print(f"Custo: US$ {resposta_modelo.usage.cost}") # como eu vou converter de dolares para reais?
    print("*"*20)
    #salvar_resposta_banco(resposta)
    return resposta

def salvar_resposta_banco(resposta):
    with conn:
        cursor.execute("INSERT INTO respostas (modelo, pergunta_usuario, resposta_modelo, token_input, token_output, total_tokens, tempo_inicial, tempo_final, tempo_total, versao) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ", 
            (resposta.modelo, resposta.pergunta_usuario, resposta.resposta_modelo, resposta.token_input, resposta.token_output, 
             resposta.total_tokens, resposta.tempo_inicial, resposta.tempo_final, resposta.tempo_total, 1))

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

            for doc in chunks_encontrados:
                print("*"*20)
                print("Chunk: ")
                print(doc.page_content)
                print("*"*20)

            """ textos_limpos = []
            for doc, score in chunks_encontrados:
                texto_sem_quebras = doc.page_content.replace("\n", " ")
                texto_ajustado = " ".join(texto_sem_quebras.split())
                textos_limpos.append(texto_ajustado)

            texto_consolidado = "\n\n".join(textos_limpos)

            tempo_inicial = time.perf_counter()
            resposta_modelo = consultar_modelo(texto_consolidado, pergunta, client, modelo)
            tempo_final = time.perf_counter()

            tempo_total = tempo_final - tempo_inicial

            if modelo == "sabia-4":
                tratar_resposta_sabia(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total)
            else:
                tratar_resposta_openrouter(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total) """
