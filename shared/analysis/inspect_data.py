import sys
import os
import json
from poke_env.data import GenData

print("=== INSPEÇÃO DE DADOS BRUTOS (GEN 9) ===")

try:
    gen_data = GenData.from_gen(9)
    type_chart = gen_data.type_chart
    
    # Vamos olhar para o tipo FOGO (Fire)
    # Queremos ver como ele toma dano de Água (Water) e Solo (Ground)
    print("\n--- DADOS DE 'FIRE' (Defensor) ---")
    if 'Fire' in type_chart:
        print(json.dumps(type_chart['Fire'], indent=2))
    elif 'FIRE' in type_chart:
        print(json.dumps(type_chart['FIRE'], indent=2))
    else:
        print("Chave 'Fire' não encontrada. Chaves disponíveis:", list(type_chart.keys())[:5])

    # Vamos olhar para ÁGUA (Water)
    # Queremos ver como ele toma dano de Normal
    print("\n--- DADOS DE 'WATER' (Defensor) ---")
    if 'Water' in type_chart:
        print(json.dumps(type_chart['Water'], indent=2))
    
except Exception as e:
    print(f"Erro fatal: {e}")