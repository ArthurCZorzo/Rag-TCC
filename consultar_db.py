from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
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
            chunks TEXT NOT NULL,
            prompt TEXT NOT NULL
        )
    """)

def criar_client(modelo):
    if modelo == "sabia-4":
        return ChatOpenAI(
            model=modelo,
            api_key=os.getenv("MARITACA_AI_API_KEY"),
            base_url="https://chat.maritaca.ai/api",
            temperature=0.2
        )
    else:
        return ChatOpenAI(
            model=modelo,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2
        )

def escolher_modelo():
    print("1 - gpt-5.6-luna")
    print("2 - gemini-3.7-flash")
    print("3 - claude-sonnet-5")
    print("4 - deepseek-v4.1-flash")
    print("5 - MaritacaAI Sabia 4")
    
    opcao = input("\nEscolha um modelo: ")
    
    if opcao == "1":
        return "openai/gpt-6.1-sol" #1.1M contexto, $2/M input tokens, $10/M output tokens
        #return "openai/gpt-5.6-luna" #1.05 M contexto, $0.20/M input tokens, $1.20/M output tokens
    elif opcao == "2":
        return "google/gemini-3.8-flash" #1.0 M contexto, $0.75/M input tokens, $3.75/M output tokens
        #return "google/gemini-3.7-flash" #1.05 M contexto, $0.75/M input tokens, $3.75/M output tokens
    elif opcao == "3":
        return "anthropic/claude-sonnet-5.5" #1 M contexto, $2/M input tokens, $10/M output tokens
        #return "anthropic/claude-sonnet-5" #1 M contexto, $2/M input tokens, $10/M output tokens
    elif opcao == "4":
        return "deepseek/deepseek-v4.1-flash" #1.05 M contexto, $0.0198/M input tokens, #0.396/M output tokens
    elif opcao == "5":
        return "sabia-4"
    else:
        print("\nOpção inválida.")
        return "sabia-4"

def reformular_pergunta():
    prompt_reformulacao = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é um especialista em busca acadêmica para o curso de Bacharelado em Sistemas de Informação do IFSC Câmpus Caçador.\n"
            "Sua tarefa é reescrever a pergunta do estudante para torná-la formal, específica, clara e perfeitamente alinhada à terminologia oficial dos documentos institucionais.\n\n"
            "--- GLOSSÁRIO INSTITUCIONAL DO IFSC CAÇADOR ---\n"
            "Traduza a linguagem coloquial do estudante para os termos oficiais correspondentes nos documentos:\n"
            "• 'rematrícula' / 'rematricula' / 'renovar matrícula' / 'fazer matrícula' / 'escolher matérias' -> 'renovação de matrícula ou ajuste de matrícula online no SIGAA / Registro Acadêmico'\n"
            "• 'horas complementares' / 'horas extras' / 'validar certificados' / 'atividades extras' -> 'validação de atividades complementares / requerimento de atividades complementares'\n"
            "• 'matérias' / 'disciplinas' / 'grade' -> 'unidades curriculares ou componentes curriculares da matriz curricular do PPC'\n"
            "• 'TCC' / 'monografia' / 'defesa de tcc' -> 'Trabalho de Conclusão de Curso (TCC I e TCC II) / Regulamento de TCC'\n"
            "• 'trancar o curso' / 'trancamento' / 'cancelar curso' -> 'trancamento de matrícula ou cancelamento de matrícula conforme o RDP'\n"
            "• 'faltas' / 'quantas faltas posso ter' / 'reprovação por falta' / 'presença' -> 'frequência mínima de 75% e assiduidade conforme o RDP (Regulamento Didático-Pedagógico)'\n"
            "• 'bolsas' / 'auxílio financeiro' / 'almoço/refeição' -> 'programa de assistência estudantil (PAE) / auxílio permanência'\n"
            "• 'livros' / 'biblioteca' / 'multa de livro' -> 'empréstimo, renovação e devolução no sistema de bibliotecas do IFSC'\n\n"
            "--- REGRAS DE REFORMULAÇÃO ---\n"
            "1. Resolução de Histórico: se a pergunta fizer referência à conversa anterior (pronomes ou elipses como 'e a carga horária?', 'onde ele dá aula?', 'qual a sala?'), recupere o sujeito do histórico (disciplina, professor, edital, fase) e torne a pergunta autocontida.\n"
            "2. Normalização e Enriquecimento: substitua termos informais pelos termos oficiais do glossário para maximizar a recuperação em banco vetorial de documentos.\n"
            "3. Correção gramatical: corrija erros ortográficos e de pontuação da pergunta original.\n"
            "4. NÃO responda à pergunta. Retorne APENAS o texto da pergunta reformulada."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    return prompt_reformulacao

def consultar_modelo(contexto, pergunta, historico_chat, client):
    historico_texto = ""
    if historico_chat:
        historico_texto = "\n[HISTÓRICO DA CONVERSA]\n"
        for msg in historico_chat[-4:]:
            tipo = "Estudante" if isinstance(msg, HumanMessage) else "Assistente"
            historico_texto += f"{tipo}: {msg.content}\n"

    prompt = f"""
    Prompt do sistema:
    Você é um assistente acadêmico institucional do IFSC Câmpus Caçador. Sua função é responder às dúvidas dos estudantes com clareza e precisão.

    Instruções:
    1. Priorize as informações do [CONTEXTO] fornecido para responder.
    2. Se a pergunta fizer referência a assuntos citados no [HISTÓRICO DA CONVERSA], mantenha a coerência com o assunto anterior.
    3. Se a informação não estiver disponível no [CONTEXTO] e você não tiver certeza absoluta, NÃO invente nada. Responda claramente que não possui essa informação e oriente o estudante a procurar a coordenação do curso, faça uma resposta curta.
    4. Mantenha um tom cortês, objetivo e institucional.

    [CONTEXTO]
    {contexto}
    {historico_texto}
    [PERGUNTA ATUAL DO ESTUDANTE]
    {pergunta}
    """.strip()
  
    return client.invoke(prompt), prompt
        
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

def tratar_resposta_modelo(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, chunks, prompt):
    usage = getattr(resposta_modelo, "usage_metadata", None) or {}
    if not usage:
        usage = resposta_modelo.response_metadata.get("token_usage", {})
    token_input = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    token_output = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    total_tokens = usage.get("total_tokens") or (token_input + token_output)

    resposta = Resposta(
            resposta_modelo.response_metadata.get("model_name"),
            pergunta,
            resposta_modelo.content,
            token_input,
            token_output,
            total_tokens,
            tempo_inicial,
            tempo_final,
            tempo_total
        )

    #salvar_resposta_banco(resposta, chunks, prompt)

def salvar_resposta_banco(resposta, chunks, prompt):
    with conn:
        cursor.execute("INSERT INTO respostas (modelo, pergunta_usuario, resposta_modelo, token_input, token_output, total_tokens, tempo_inicial, tempo_final, tempo_total, chunks, prompt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ", 
            (resposta.modelo, resposta.pergunta_usuario, resposta.resposta_modelo, resposta.token_input, resposta.token_output, 
             resposta.total_tokens, resposta.tempo_inicial, resposta.tempo_final, resposta.tempo_total, chunks, prompt))

if __name__ == "__main__":
    criar_banco_benchmark()
    modelo = escolher_modelo()
    client = criar_client(modelo)

    prompt_reformulacao = reformular_pergunta()

    db_conectado = inicializar_sistema_busca(diretorio_banco="database")
    recuperador = db_conectado.as_retriever(search_kwargs={"k": 5})

    historico = create_history_aware_retriever(
        llm=client,
        retriever=recuperador,
        prompt=prompt_reformulacao 
    )

    historico_chat = []

    rodar = True
    while rodar:
        pergunta = input("Digite sua pergunta: ")

        if pergunta.lower() == "sair":
            print("Saindo...")
            rodar = False
        else:
           #chunks_encontrados = buscar_top_chunks(db_conectado, pergunta, k=5)

            chunks_encontrados = historico.invoke({
                "input": pergunta,
                "chat_history": historico_chat
            })

            textos_limpos = []
            for doc in chunks_encontrados:
                texto_sem_quebras = doc.page_content.replace("\n", " ")
                texto_ajustado = " ".join(texto_sem_quebras.split())
                textos_limpos.append(texto_ajustado)

            texto_consolidado = "\n\n".join(textos_limpos)

            tempo_inicial = time.perf_counter()
            resposta_modelo, prompt_completo = consultar_modelo(texto_consolidado, pergunta, historico_chat, client)
            tempo_final = time.perf_counter()

            tempo_total = tempo_final - tempo_inicial

            print(resposta_modelo)

            historico_chat.append(HumanMessage(content=pergunta))
            historico_chat.append(AIMessage(content=resposta_modelo.content))

            tratar_resposta_modelo(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, texto_consolidado, prompt_completo)
            #if modelo == "sabia-4":
                #tratar_resposta_sabia(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, texto_consolidado)
            #else:
                #tratar_resposta_openrouter(pergunta, resposta_modelo, tempo_inicial, tempo_final, tempo_total, texto_consolidado)
