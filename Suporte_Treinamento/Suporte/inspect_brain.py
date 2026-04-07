import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

# --- CONFIGURAÇÃO DE CAMINHOS ---
BRAIN_DIR = r"C:\Projetos Robotica Computacional\Projeto Showdown IA Pokemon\Bot-QV-Pokemon\Instinto"
BRAIN_FILE = os.path.join(BRAIN_DIR, "blue_brain.pkl")

SUPPORT_DIR = r"C:\Projetos Robotica Computacional\Projeto Showdown IA Pokemon\Bot-QV-Pokemon\Suporte_Treinamento\Suporte"
LOG_DIR = os.path.join(SUPPORT_DIR, "logs_tese")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Ações exatas mapeadas no BlueBrain
BLUE_ACTIONS = [
    "ATTACK", "SWITCH", "BUFF", "DEBUFF", "STATUS", "PROTECT", 
    "HAZARD", "HEAL", "HEAL_50", "TEAM_CURE", "CLEAN", "STAT_CLEAN"
]

def decode_state_blue_exact(state_tuple):
    """Mapeia exatamente as 13 variáveis do novo conjunto de tuplas"""
    try:
        s = list(state_tuple)
        while len(s) < 13:
            s.append(None)
            
        my_role, opp_role, matchup, my_hp, opp_hp, weather, speed_tier, \
        status_my, status_opp, boost_my, boost_opp, hazard_my, hazard_opp = s[:13]
        
        # Colocando os status lado a lado (1 linha)
        return [
            f"{my_role} vs {opp_role}",
            str(matchup),
            f"{my_hp} vs {opp_hp}",
            str(speed_tier),
            f"{status_my} vs {status_opp}",
            f"{boost_my} vs {boost_opp}",
            f"{hazard_my} vs {hazard_opp}",
            str(weather)
        ]
    except Exception as e:
        return ["Erro de Leitura"] * 8

def generate_dashboard(df, action_counts, filename):
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.1)
    fig.patch.set_facecolor('#f4f4f9')

    plt.suptitle(f"ANÁLISE DO MODELO BLUE (13 DIMENSÕES) - {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                 fontsize=18, weight='bold', color='#333333', y=0.96)

    # 1. Pizza (Centralizada no topo)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Ataque', 'Troca', 'Suporte']
    sizes = [action_counts['attack'], action_counts['switch'], action_counts['support']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    filtered_sizes = [s for s in sizes if s > 0]
    filtered_labels = [l for s, l in zip(sizes, labels) if s > 0]
    filtered_colors = [c for s, c in zip(sizes, colors) if s > 0]
    
    if sum(sizes) > 0:
        ax1.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=90, 
                wedgeprops={'edgecolor': 'black'}, textprops={'weight': 'bold', 'fontsize': 10})
        ax1.set_title("Proporção de Melhores Ações", fontsize=12, weight='bold', pad=10)
        ax1.set_aspect('equal')
    else:
        ax1.text(0.5, 0.5, "Dados insuficientes", ha='center', va='center')
        ax1.axis('off')

    # 2. Tabela de Dados (Top 20)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = df.values.tolist()
    col_labels = df.columns.tolist()
    
    # Distribuição ajustada (1.0 = 100%)
    col_widths = [0.05, 0.09, 0.16, 0.10, 0.12, 0.08, 0.11, 0.11, 0.09, 0.09]
    
    the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', 
                          cellLoc='center', colWidths=col_widths, bbox=[0, 0, 1, 1])
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8) # Fonte reduzida em 2 pontos
    the_table.scale(1, 1.8)   # Ajuste vertical proporcional à redução de linhas
    
    for (i, j), cell in the_table.get_celld().items():
        if i == 0: 
            cell.set_text_props(color='white', weight='bold')
            cell.set_facecolor("#2c3e50")
        else:
            action_text = table_data[i-1][1]
            if "ATTACK" in action_text:
                cell.set_facecolor("#ffe6e6")
            elif "SWITCH" in action_text:
                cell.set_facecolor("#e6f2ff")
            else:
                cell.set_facecolor("#e6ffe6")

    ax2.set_title("Top 20 Estados Consolidados", fontsize=14, weight='bold', pad=15)
    
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGEM] Dashboard salvo em: {filename}")

def analyze_brain():
    print("=== EXTRATOR DE DADOS DO BRAIN ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(BRAIN_FILE):
        print(f"ERRO: Arquivo não encontrado em {BRAIN_FILE}")
        return

    try:
        with open(BRAIN_FILE, "rb") as f:
            data = pickle.load(f)
        
        q_table = data.get("q_table", {})
        epsilon = data.get("epsilon", -1)

        states_dense = {}
        for state, q_values_array in q_table.items():
            if isinstance(state, tuple) and isinstance(q_values_array, (list, np.ndarray)):
                states_dense[state] = np.array(q_values_array)

        total_states = len(states_dense)
        if total_states == 0:
            print("AVISO: A Q-Table extraída está vazia.")
            return

        positive_states = 0
        negative_states = 0
        ranked_states = []
        action_counts = {'attack': 0, 'switch': 0, 'support': 0}

        for state, values in states_dense.items():
            max_val = np.max(values)
            best_action = np.argmax(values)
            
            if best_action == 0:
                action_counts['attack'] += 1
            elif best_action == 1:
                action_counts['switch'] += 1
            else:
                action_counts['support'] += 1
            
            if max_val > 0: positive_states += 1
            else: negative_states += 1
            
            ranked_states.append((max_val, state, best_action))

        ranked_states.sort(key=lambda x: x[0], reverse=True)
        top_20 = ranked_states[:20]

        export_data = []
        for val, state, action in top_20:
            state_cols = decode_state_blue_exact(state)
            act_name = BLUE_ACTIONS[action] if action < len(BLUE_ACTIONS) else f"UNKNOWN ({action})"
            
            export_data.append({
                "Q-Value": f"{val:.2f}",
                "Ação": act_name,
                "Roles (My vs Opp)": state_cols[0],
                "Matchup": state_cols[1],
                "HP (My vs Opp)": state_cols[2],
                "Speed": state_cols[3],
                "Status (My vs Opp)": state_cols[4],
                "Boosts (My vs Opp)": state_cols[5],
                "Hazards (My vs Opp)": state_cols[6],
                "Clima": state_cols[7]
            })
            
        df_top20 = pd.DataFrame(export_data)
        base_name = os.path.join(LOG_DIR, f"analise_blue_{timestamp}")
        
        generate_dashboard(df_top20, action_counts, f"{base_name}_dashboard.png")
        
        with open(f"{base_name}_report.txt", "w", encoding='utf-8') as f:
            f.write(f"RELATÓRIO DE TREINAMENTO (MODELO BLUE 13D) - {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total de Estados Únicos: {total_states}\n")
            f.write(f"Taxa de Exploração (Epsilon): {epsilon:.5f}\n")
            f.write(f"Estados Otimistas (Q > 0): {positive_states}\n")
            f.write(f"Estados Pessimistas (Q <= 0): {negative_states}\n\n")
            f.write("TOP 20 ESTADOS:\n")
            f.write(df_top20.to_string())
            
        print(f"[RELATÓRIO] Texto salvo em: {base_name}_report.txt")
        print("Processo finalizado.")

    except Exception as e:
        print(f"ERRO FATAL DURANTE EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_brain()