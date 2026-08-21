import json
import sys
from poke_env.data import GenData

print("🔍 Iniciando Auditoria da Tabela de Tipos...")

try:
    # 1. Carrega os dados da Geração 9 (igual ao seu bot)
    gen_data = GenData.from_gen(9)
    type_chart = gen_data.type_chart

    # 2. Define o nome do arquivo de saída
    output_file = "tabela_tipos_dump.json"

    # 3. Salva a estrutura EXATA num arquivo JSON formatado
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(type_chart, f, indent=4, sort_keys=True)

    print(f"✅ Sucesso! Os dados foram salvos em: {output_file}")
    print("👉 Abra esse arquivo no VS Code e procure por 'Fire' ou 'FIRE' para ver como está escrito.")

    # 4. Teste Rápido de Acesso (Sem printar a tabela toda)
    print("\n--- TESTE DE CHAVES (Amostragem) ---")
    keys = list(type_chart.keys())
    print(f"Exemplo de Chaves (Defensores): {keys[:5]}")
    
    # Verifica FOGO especificamente para tirar a dúvida
    if 'Fire' in type_chart:
        print("⚠️  A chave 'Fire' existe (Title Case).")
    if 'FIRE' in type_chart:
        print("⚠️  A chave 'FIRE' existe (UPPER CASE).")
    if 'fire' in type_chart:
        print("⚠️  A chave 'fire' existe (lower case).")

except Exception as e:
    print(f"❌ Erro fatal: {e}")