import sqlite3

conn = sqlite3.connect("benchmark.db")
cursor = conn.cursor()

with conn:
    cursor.execute("SELECT * FROM respostas")

    todas_respostas = cursor.fetchall()

    print("*"*20)
    print("Respostas: ")
    for resposta in todas_respostas:
        i = 0
        while i < 10:
            print(f"Resposta[{i}]")
            print(resposta[i])
            i = i + 1
    print("*"*20)