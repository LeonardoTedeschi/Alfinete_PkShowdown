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

# Ações V_6 (18 ações ramificadas)
BLUE_ACTIONS = [
    "ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH", 
    "ATTACK_MEC", "BUFF", "STATUS", "HEAL", "CLEAN_HAZARD", 
    "PROTECT", "DEBUFF", "STAT_CLEAN", "HEAL_STATUS", "PHAZE", 
    "FIELD_CONTROL", "HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"
]

def decode_state_blue_exact(state_tuple):
    """Extração à prova de falhas: resolve tuplas aninhadas e formata strings longas"""
    try:
        def flatten(t):
            if isinstance(t, (tuple, list, np.ndarray)):
                for item in t:
                    yield from flatten(item)
            else:
                yield t
        
        s = list(flatten(state_tuple))
        
        while len(s) < 14:
            s.append("?")
            
        # Simplificação de strings para economizar espaço
        my_role = str(s[0]).replace('SPEED_SWEEPER', 'SWEEPER').replace('TANK_BULK', 'TANK')
        opp_role = str(s[1]).replace('SPEED_SWEEPER', 'SWEEPER').replace('TANK_BULK', 'TANK')
        matchup = str(s[2]).replace('OFFENSIVE_', 'OFF_').replace('DEFENSIVE_', 'DEF_')
        my_hp = str(s[3])
        opp_hp = str(s[4])
        weather = str(s[5])
        speed_tier = str(s[6])
        status_my = str(s[7])
        status_opp = str(s[8])
        boost_my = str(s[9])
        boost_opp = str(s[10])
        hazard_my = str(s[11])
        hazard_opp = str(s[12])
        mechanic = str(s[13])
        
        return [
            f"{my_role} v {opp_role}",
            matchup,
            f"{my_hp} v {opp_hp}",
            speed_tier,
            f"{status_my} v {status_opp}",
            f"{boost_my} v {boost_opp}",
            f"{hazard_my} v {hazard_opp}",
            weather,
            mechanic
        ]
    except Exception as e:
        return [f"ERR: {str(e)[:10]}"] * 9

def generate_dashboard(df, action_counts, filename):
    fig = plt.figure(figsize=(26, 14)) 
    gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.1)
    fig.patch.set_facecolor('#f4f4f9')

    plt.suptitle(f"ANÁLISE DO MODELO BLUE (V6 - 18 AÇÕES) - {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                 fontsize=20, weight='bold', color='#333333', y=0.96)

    # 1. Pizza
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Ataque', 'Troca', 'Suporte']
    sizes = [action_counts['attack'], action_counts['switch'], action_counts['support']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    filtered_sizes = [s for s in sizes if s > 0]
    filtered_labels = [l for s, l in zip(sizes, labels) if s > 0]
    filtered_colors = [c for s, c in zip(sizes, colors) if s > 0]
    
    if sum(sizes) > 0:
        ax1.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=90, 
                wedgeprops={'edgecolor': 'black'}, textprops={'weight': 'bold', 'fontsize': 11})
        ax1.set_title("Proporção de Melhores Ações Otimizadas", fontsize=14, weight='bold', pad=10)
        ax1.set_aspect('equal')
    else:
        ax1.text(0.5, 0.5, "Q-Table Vazia", ha='center', va='center')
        ax1.axis('off')

    # 2. Tabela de Dados (Top 20)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = df.values.tolist()
    col_labels = df.columns.tolist()
    
    # Balanceamento das 11 colunas
    col_widths = [0.05, 0.10, 0.13, 0.10, 0.10, 0.07, 0.11, 0.11, 0.10, 0.07, 0.06]
    
    the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', 
                          cellLoc='center', colWidths=col_widths, bbox=[0, 0, 1, 1])
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9) 
    the_table.scale(1, 1.8)
    
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

    ax2.set_title("Top 20 Estados Aprendidos (Raw Mapping)", fontsize=16, weight='bold', pad=15)
    
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.02, right=0.98)
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
            if isinstance(q_values_array, (list, np.ndarray)):
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
            
            # Ajuste correto pelos novos índices de BLUE_ACTIONS
            if best_action in [0, 1, 2, 3, 4]: # Variações de ATTACK
                action_counts['attack'] += 1
            elif best_action in [16, 17]: # Variações de SWITCH
                action_counts['switch'] += 1
            else: # Ações de Suporte (índices 5 ao 15)
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
                "Roles (My v Opp)": state_cols[0],
                "Matchup": state_cols[1],
                "HP (My v Opp)": state_cols[2],
                "Speed": state_cols[3],
                "Status (My v Opp)": state_cols[4],
                "Boosts (My v Opp)": state_cols[5],
                "Hazards (My v Opp)": state_cols[6],
                "Clima": state_cols[7],
                "Mecânica": state_cols[8]
            })
            
        df_top20 = pd.DataFrame(export_data)
        base_name = os.path.join(LOG_DIR, f"analise_blue_{timestamp}")
        
        generate_dashboard(df_top20, action_counts, f"{base_name}_dashboard.png")
        
        with open(f"{base_name}_report.txt", "w", encoding='utf-8') as f:
            f.write(f"RELATÓRIO DE TREINAMENTO (MODELO BLUE V6) - {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total de Estados Únicos: {total_states}\n")
            f.write(f"Taxa de Exploração (Epsilon): {epsilon:.5f}\n")
            f.write(f"Estados Otimistas (Q > 0): {positive_states}\n")
            f.write(f"Estados Pessimistas (Q <= 0): {negative_states}\n\n")
            f.write("TOP 20 ESTADOS:\n")
            f.write(df_top20.to_string())
            
        print(f"[RELATÓRIO] Dashboard e Log salvos em: logs_tese")
        print("Processo finalizado com sucesso.")

    except Exception as e:
        print(f"ERRO FATAL DURANTE EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_brain()