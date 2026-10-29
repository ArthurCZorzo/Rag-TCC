import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from consultar_db import (
    criar_banco_benchmark,
    escolher_modelo,
    criar_client,
    reformular_pergunta,
    inicializar_sistema_busca,
    create_history_aware_retriever,
    consultar_modelo,
    tratar_resposta_modelo
)

class Rag:
    def __init__(self):
        self.modelo = "anthropic/claude-sonnet-5.5"
        self.client = None
        self.retriever = None

    def criar_cliente(self):
        self.client = criar_client(self.modelo)

    def criar_prompt_reformulacao(self):
        return reformular_pergunta()

    def inicializar_busca(self):
        return inicializar_sistema_busca(diretorio_banco="database")

    def configurar(self, prompt_reformulacao, db_conectado):
        print("\nInicializando banco vetorial e retriever...")
        recuperador_base = db_conectado.as_retriever(search_kwargs={"k": 8})

        self.retriever = create_history_aware_retriever(
            llm=self.client,
            retriever=recuperador_base,
            prompt=prompt_reformulacao
        )

    def executar(self, pergunta, historico_chat):
        # 1. Recuperação dos chunks considerando o histórico
        chunks_encontrados = self.retriever.invoke({
            "input": pergunta,
            "chat_history": historico_chat
        })
        
        textos_limpos = []
        textos_vistos = set()
        for doc in chunks_encontrados:
            texto_ajustado = " ".join(doc.page_content.replace("\n", " ").split())
            if texto_ajustado not in textos_vistos:
                textos_vistos.add(texto_ajustado)
                textos_limpos.append(texto_ajustado)
        texto_consolidado = "\n\n".join(textos_limpos)

        #print(f"CONTEXTO: {texto_consolidado}") 

        # 2. Resposta do modelo considerando o contexto e o histórico
        resposta_modelo, prompt_completo = consultar_modelo(
            texto_consolidado, pergunta, historico_chat, self.client
        )
        print(f"PROMPT COMPLETO: {prompt_completo}")
        print(resposta_modelo.content)
        return resposta_modelo.content

@st.cache_resource
def inicializar_rag():
    print("Inicializando...")
    rag = Rag()
    rag.criar_cliente()
    prompt_reformulacao = rag.criar_prompt_reformulacao()
    banco = rag.inicializar_busca()
    rag.configurar(prompt_reformulacao=prompt_reformulacao, db_conectado=banco)
    print("Fim da inicialização...")
    return rag

if __name__ == "__main__":
    rag = inicializar_rag()

    st.header("RAG Acadêmico - IFSC Caçador", divider="green")

    # 1. Inicializa os históricos e estados na sessão caso não existam
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Para o LangChain (HumanMessage, AIMessage)

    if "mensagens_tela" not in st.session_state:
        st.session_state.mensagens_tela = []  # Para renderizar na tela (role, content)

    if "gerando_resposta" not in st.session_state:
        st.session_state.gerando_resposta = False

    if "pergunta_pendente" not in st.session_state:
        st.session_state.pergunta_pendente = None

    # 2. Container e renderização das mensagens anteriores para não sumirem
    messages = st.container(height=500, autoscroll=True)
    for msg in st.session_state.mensagens_tela:
        messages.chat_message(msg["role"]).write(msg["content"])

    # 3. Entrada do usuário (bloqueada enquanto processa)
    placeholder_texto = "Aguarde a resposta..." if st.session_state.gerando_resposta else "Digite sua dúvida sobre o curso..."
    user_input = st.chat_input(placeholder_texto, disabled=st.session_state.gerando_resposta)

    if user_input:
        # Registra a mensagem na tela e define como pendente para processar com o input travado
        st.session_state.mensagens_tela.append({"role": "user", "content": user_input})
        st.session_state.pergunta_pendente = user_input
        st.session_state.gerando_resposta = True
        st.rerun()

    # 4. Processamento da pergunta pendente (com o campo de entrada travado)
    if st.session_state.gerando_resposta and st.session_state.pergunta_pendente:
        pergunta = st.session_state.pergunta_pendente

        with messages.chat_message("assistant"):
            with st.spinner("Buscando nos documentos sua resposta..."):
                resposta = rag.executar(pergunta, st.session_state.chat_history)
            st.write(resposta)

        # Atualiza a interface e o histórico do LangChain
        st.session_state.mensagens_tela.append({"role": "assistant", "content": resposta})
        st.session_state.chat_history.append(HumanMessage(content=pergunta))
        st.session_state.chat_history.append(AIMessage(content=resposta))

        # Libera o campo de entrada e atualiza a tela
        st.session_state.pergunta_pendente = None
        st.session_state.gerando_resposta = False
        st.rerun()
