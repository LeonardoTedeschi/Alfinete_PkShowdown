import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# --- CONFIGURAÇÃO ---
LOG_DIR = "logs_tese"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def decode_state_v5(state_tuple):
    """Traduz a tupla de 11 variáveis do Brain V5 para texto."""
    try:
        # Desempacota as 11 variáveis
        threat, off_pot, hp_my, hp_opp, st_my, st_opp, faster, bst_my, bst_opp, score, weather = state_tuple
        
        # Dicionários de Tradução
        threat_map = {2: "EXTREMA", 1: "Alta", 0: "Neutra", -1: "Baixa"}
        off_map = {2: "Ótimo", 1: "Bom", 0: "Neutro", -1: "Ruim"}
        hp_map = {3: "Cheio", 2: "Médio", 1: "Baixo", 0: "Crítico"}
        w_map = {0: "Limpo", 1: "Sol/Chuva", 2: "Areia/Neve", 3: "TrickRoom"}
        score_map = {1: "Vencendo", 0: "Empate", -1: "Perdendo"}
        
        # Montagem da String (Resumida para caber na tabela)
        context = (
            f"Ameaça: {threat_map.get(threat, threat)} | "
            f"Dano: {off_map.get(off_pot, off_pot)}\n"
            f"HP: {hp_map.get(hp_my)} vs {hp_map.get(hp_opp)} | "
            f"Clima: {w_map.get(weather)}\n"
            f"Speed: {'Rápido' if faster else 'Lento'} | "
            f"Placar: {score_map.get(score)}"
        )
        return context
    except Exception as e:
        return f"Erro Decode: {str(e)}"

def generate_table_image(df_top10, filename):
    """Gera uma imagem PNG bonita da tabela."""
    fig, ax = plt.subplots(figsize=(14, 8)) # Tamanho da imagem
    ax.axis('off')
    
    # Cria a tabela
    table_data = []
    for _, row in df_top10.iterrows():
        table_data.append([
            f"{row['Q-Value']:.1f}",
            row['Ação'],
            row['Cenário']
        ])
    
    col_labels = ["Recompensa (Q)", "Melhor Ação", "Contexto do Estado"]
    
    # Desenha a tabela
    the_table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
        colColours=["#40466e"] * 3
    )
    
    # Estilização
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 2.5) # Aumenta altura das linhas
    
    # Cores alternadas e cabeçalho
    for (i, j), cell in the_table.get_celld().items():
        if i == 0: # Cabeçalho
            cell.set_text_props(color='white', weight='bold')
        else: # Linhas de dados
            if "ATACAR" in table_data[i-1][1]:
                cell.set_facecolor("#ffe6e6") # Vermelho claro para Ataque
            else:
                cell.set_facecolor("#e6f2ff") # Azul claro para Troca

    plt.title(f"TOP 10 ESTADOS APRENDIDOS - {datetime.now().strftime('%d/%m %H:%M')}", 
              fontsize=16, weight='bold', color='#333333')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[IMAGEM] Tabela salva em: {filename}")

def analyze_brain():
    print("=== GERADOR DE DADOS PARA TESE (V5) ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists("red_brain.pkl"):
        print("ERRO: red_brain.pkl não encontrado.")
        return

    try:
        with open("red_brain.pkl", "rb") as f:
            data = pickle.load(f)
        
        q_table = data.get("q_table", {})
        epsilon = data.get("epsilon", -1)

        # Processamento dos Dados
        # Converte formato esparso {(state, action): val} para denso {state: [vals]}
        states_dense = defaultdict(lambda: np.zeros(10))
        for (state, action), val in q_table.items():
            states_dense[state][action] = val

        total_states = len(states_dense)
        positive_states = 0
        negative_states = 0
        ranked_states = []

        for state, values in states_dense.items():
            max_val = np.max(values)
            best_action = np.argmax(values)
            
            if max_val > 0: positive_states += 1
            else: negative_states += 1
            
            ranked_states.append((max_val, state, best_action))

        # Ordenar Top 10
        ranked_states.sort(key=lambda x: x[0], reverse=True)
        top_10 = ranked_states[:10]

        # Preparar dados para exportação
        export_data = []
        for val, state, action in top_10:
            context = decode_state_v5(state)
            act_name = "ATACAR" if action < 4 else "TROCAR"
            export_data.append({
                "Q-Value": val,
                "Ação": f"{act_name} (idx {action})",
                "Cenário": context
            })
            
        df_top10 = pd.DataFrame(export_data)

        # --- GERAÇÃO DE ARQUIVOS ---
        base_name = f"{LOG_DIR}/analise_{timestamp}"
        
        # 1. Imagem da Tabela
        generate_table_image(df_top10, f"{base_name}_tabela.png")
        
        # 2. Relatório de Texto
        with open(f"{base_name}_report.txt", "w", encoding='utf-8') as f:
            f.write(f"RELATÓRIO DE TREINAMENTO - {timestamp}\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total de Estados Únicos: {total_states}\n")
            f.write(f"Taxa de Exploração (Epsilon): {epsilon:.5f}\n")
            f.write(f"Estados Otimistas: {positive_states}\n")
            f.write(f"Estados Pessimistas: {negative_states}\n")
            ratio = (positive_states/total_states)*100 if total_states > 0 else 0
            f.write(f"Taxa de Confiança: {ratio:.2f}%\n\n")
            f.write("TOP 10 ESTADOS:\n")
            f.write(df_top10.to_string())
            
        print(f"[RELATÓRIO] Texto salvo em: {base_name}_report.txt")
        print("\nPronto! Verifique a pasta 'logs_tese'.")

    except Exception as e:
        print(f"ERRO FATAL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_brain()