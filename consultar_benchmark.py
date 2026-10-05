import sqlite3
import csv

def exportar_respostas_para_csv(arquivo_db="benchmark.db", arquivo_csv="respostas_modelo.csv"):
    conn = sqlite3.connect(arquivo_db)
    cursor = conn.cursor()

    # Buscar todos os registros
    cursor.execute("SELECT * FROM respostas ORDER BY id;")
    rows = cursor.fetchall()

    # Obter os nomes das colunas da tabela
    colunas = [description[0] for description in cursor.description]

    # Salvar no arquivo CSV separado por ';'
    with open(arquivo_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        
        # Escreve o cabeçalho (nomes das colunas)
        writer.writerow(colunas)
        
        # Escreve os dados
        for row in rows:
            # Limpa quebras de linha internas para manter 1 linha por registro no CSV
            row_limpa = [
                str(val).replace("\r\n", " ").replace("\n", " ") if isinstance(val, str) else val
                for val in row
            ]
            writer.writerow(row_limpa)

    conn.close()
    print(f"Exportação concluída! {len(rows)} registros salvos em '{arquivo_csv}'.")

if __name__ == "__main__":
    exportar_respostas_para_csv()