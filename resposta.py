class Resposta:
    #def __init__(self):
        #self.modelo = ""
        #self.pergunta_usuario = ""
        #self.resposta_modelo = ""
        #self.token_input = 0
        #self.token_output = 0
        #self.total_tokens = 0
        #self.tempo_inicial = 0.0
        #self.tempo_final = 0.0
        #self.tempo_total = 0.0

    def __init__(self, modelo, pergunta_usuario, resposta_modelo, token_input, token_output, total_tokens, tempo_inicial, tempo_final, tempo_total):
        self.modelo = modelo
        self.pergunta_usuario = pergunta_usuario
        self.resposta_modelo = resposta_modelo
        self.token_input = token_input
        self.token_output = token_output
        self.total_tokens = total_tokens
        self.tempo_inicial = tempo_inicial
        self.tempo_final = tempo_final
        self.tempo_total = tempo_total
