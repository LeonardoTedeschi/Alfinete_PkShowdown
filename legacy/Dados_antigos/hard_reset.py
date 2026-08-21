import pickle
import os

def hard_reset_brain():
    print("=== HARD RESET: ESTOURANDO A BOLHA ===")
    filename = "red_brain.pkl"
    
    if not os.path.exists(filename):
        print("Erro: Arquivo red_brain.pkl não encontrado.")
        return

    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    q_table = data.get("q_table", {})
    
    # 1. DEFLAÇÃO BRUTAL (Divisão por 10)
    # Traz o Topo de 17.000 para 1.700 (Realista)
    new_q_table = {k: v / 10.0 for k, v in q_table.items()}
    
    # 2. INJEÇÃO DE CURIOSIDADE (Resetar Epsilon no Arquivo)
    # Vamos forçar o arquivo a aceitar que ele precisa explorar
    # Definimos 30% de exploração inicial para o próximo treino
    data["q_table"] = new_q_table
    data["epsilon"] = 0.3 
    
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Sucesso!")
    print(f"- Valores divididos por 10.")
    print(f"- Epsilon resetado para 0.3 (30% de exploração).")
    print("O bot está pronto para aprender coisas novas.")

if __name__ == "__main__":
    hard_reset_brain()