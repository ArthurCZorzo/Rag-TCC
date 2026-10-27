import time
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

class BenchmarkRAG:
    def __init__(self, perguntas=None, repeticoes=3):
        self.perguntas = perguntas or [
            "Trabalho como programador, posso validar a matéria de algoritmos?",
            "Estou com dúvida com relação aos processos acadêmicos do curso, com quem entro em contato?",
            "Desejo participar de uma bolsa estudantil, com quem entro em contato?",
            "Aonde eu encontro o meu histórico escolar no sigaa?",
            "Qual é a carga horária de Algoritmos 2?",
            "Como eu me torno um representante dos alunos no CEPE?",
            "Eu tenho um certificado de estágio não obrigatório, posso utilizar nas horas complementares?",
            "Quais matérias eu devo ter feito para fazer o TCC?",
            "Quais são meus direitos como estudantes do ifsc?",
            "Aonde eu posso encontrar os TCCs produzidos pelos alunos do curso?",
            "Me diga uma receita de brownie."
        ]
        self.repeticoes = repeticoes
        self.modelo = None
        self.client = None
        self.retriever = None

    def configurar(self):
        criar_banco_benchmark()
        self.modelo = escolher_modelo()
        self.client = criar_client(self.modelo)

        print("\nInicializando banco vetorial e retriever...")
        prompt_reformulacao = reformular_pergunta()
        db_conectado = inicializar_sistema_busca(diretorio_banco="database")
        recuperador_base = db_conectado.as_retriever(search_kwargs={"k": 5})

        self.retriever = create_history_aware_retriever(
            llm=self.client,
            retriever=recuperador_base,
            prompt=prompt_reformulacao
        )

    def executar(self):
        if not self.client or not self.retriever:
            self.configurar()

        total_testes = len(self.perguntas) * self.repeticoes
        execucao_atual = 0

        print(f"\n{'=' * 60}")
        print(f"Iniciando Benchmark: {len(self.perguntas)} perguntas x {self.repeticoes} repetições = {total_testes} testes")
        print(f"Modelo selecionado: {self.modelo}")
        print(f"{'=' * 60}\n")

        for i, pergunta in enumerate(self.perguntas, start=1):
            for rodada in range(1, self.repeticoes + 1):
                execucao_atual += 1
                print(f"[{execucao_atual}/{total_testes}] Pergunta {i} | Rodada {rodada}/{self.repeticoes}")
                print(f"Pergunta: '{pergunta}'")

                # Cada rodada de pergunta de benchmark inicia com o histórico limpo
                # para garantir a independência da medição
                historico_chat = []

                # 1. Recuperação dos chunks
                chunks_encontrados = self.retriever.invoke({
                    "input": pergunta,
                    "chat_history": historico_chat
                })

                textos_limpos = [
                    " ".join(doc.page_content.replace("\n", " ").split())
                    for doc in chunks_encontrados
                ]
                texto_consolidado = "\n\n".join(textos_limpos)

                # 2. Consulta ao LLM com cronometragem
                tempo_inicial = time.perf_counter()
                resposta_modelo, prompt_completo = consultar_modelo(
                    texto_consolidado, pergunta, historico_chat, self.client
                )
                tempo_final = time.perf_counter()
                tempo_total = tempo_final - tempo_inicial

                print(f"Tempo de resposta: {tempo_total:.2f}s")
                resumo_resposta = resposta_modelo.content.replace('\n', ' ')[:120]
                print(f"Resposta prévia: {resumo_resposta}...\n")

                # 3. Salva no banco de benchmark
                tratar_resposta_modelo(
                    pergunta=pergunta,
                    resposta_modelo=resposta_modelo,
                    tempo_inicial=tempo_inicial,
                    tempo_final=tempo_final,
                    tempo_total=tempo_total,
                    chunks=texto_consolidado,
                    prompt=prompt_completo
                )

                # Pausa breve de 1s para evitar limites de requisições por minuto da API
                time.sleep(2)

        print("\nBenchmark finalizado com sucesso! Todos os dados foram gravados no banco.")

if __name__ == "__main__":
    benchmark = BenchmarkRAG(repeticoes=3)
    benchmark.executar()
