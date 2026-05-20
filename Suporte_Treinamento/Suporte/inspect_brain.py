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

# Define o diretório de logs dentro da própria pasta Instinto
LOG_DIR = os.path.join(BRAIN_DIR, "logs_analise")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

BASE_ACTIONS = [
    "ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH", 
    "BUFF", "STATUS", "HEAL", "CLEAN_HAZARD", 
    "PROTECT", "DEBUFF", "DISRUPTION", "STAT_CLEAN", "HEAL_STATUS", "PHAZE",
    "FIELD_CONTROL", "HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
    "BARRIER"
]

BLUE_ACTIONS = []
for act in BASE_ACTIONS:
    BLUE_ACTIONS.append(act)
    if "SWITCH" not in act:
        BLUE_ACTIONS.append(f"{act}_MEC")

def decode_state_blue_exact(state_tuple):
    try:
        def flatten(t):
            if isinstance(t, (tuple, list, np.ndarray)):
                for item in t:
                    yield from flatten(item)
            else:
                yield t
        s = list(flatten(state_tuple))
        while len(s) < 15: s.append("?") # Atualizado para 16 dimensões exatas do InstinctCore
            
        my_role = str(s[0]).replace('SPEED_SWEEPER', 'SWEEPER').replace('TANK_BULK', 'TANK')
        opp_role = str(s[1]).replace('SPEED_SWEEPER', 'SWEEPER').replace('TANK_BULK', 'TANK')
        matchup = str(s[2]).replace('OFFENSIVE_', 'OFF_').replace('DEFENSIVE_', 'DEF_')
        
        return [
            f"{my_role} v {opp_role}", matchup, f"{s[3]} v {s[4]}", 
            str(s[5]), str(s[6]), f"{s[7]} v {s[8]}", f"{s[9]} v {s[10]}", 
            f"{s[11]} v {s[12]}", str(s[13]), str(s[14])
        ]
    except Exception as e:
        return [f"ERR"] * 10

def generate_dashboard(df, action_counts, visit_stats, filename):
    fig = plt.figure(figsize=(26, 16)) 
    # Adicionamos uma coluna para o histograma de visitas
    gs = GridSpec(2, 2, height_ratios=[1.2, 4], width_ratios=[1, 1.2], hspace=0.15)
    fig.patch.set_facecolor('#f4f4f9')

    plt.suptitle(f"ANÁLISE DO MODELO BLUE - {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                 fontsize=22, weight='bold', color='#333333', y=0.96)

    # 1. Pizza de Ações (Esquerda Superior)
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
        ax1.set_title("Proporção de Melhores Ações", fontsize=14, weight='bold', pad=10)
        ax1.set_aspect('equal')

    # 2. Histograma de Visitas (Direita Superior)
    ax3 = fig.add_subplot(gs[0, 1])
    bars = ['1 Visita', '2 a 4 Visitas\n(Mestre-Aluno)', '5 a 19 Visitas\n(Exploração)', '20+ Visitas\n(Maduro)']
    counts = [visit_stats['1'], visit_stats['2_4'], visit_stats['5_19'], visit_stats['20+']]
    bar_colors = ['#ff6666', '#ffcc66', '#99ccff', '#66cc66']
    
    x_pos = np.arange(len(bars))
    ax3.bar(x_pos, counts, color=bar_colors, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bars, fontsize=10, weight='bold')
    ax3.set_title("Profundidade de Aprendizado (Frequência de Visitas)", fontsize=14, weight='bold', pad=10)
    ax3.set_ylabel("Quantidade de Estados", fontsize=12, weight='bold')
    
    for i, v in enumerate(counts):
        pct = (v / max(1, sum(counts))) * 100
        ax3.text(i, v + (max(counts)*0.02), f"{v}\n({pct:.1f}%)", ha='center', va='bottom', weight='bold')

    # 3. Tabela de Dados (Embaixo, ocupando tudo)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = df.values.tolist()
    col_labels = df.columns.tolist()
    col_widths = [0.05, 0.06, 0.10, 0.11, 0.08, 0.08, 0.07, 0.09, 0.09, 0.08, 0.08, 0.06, 0.06]
    
    the_table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colWidths=col_widths, bbox=[0, 0, 1, 1])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9) 
    the_table.scale(1, 1.8)
    
    for (i, j), cell in the_table.get_celld().items():
        if i == 0: 
            cell.set_text_props(color='white', weight='bold')
            cell.set_facecolor("#2c3e50")
        else:
            action_text = table_data[i-1][2]
            if "ATTACK" in action_text: cell.set_facecolor("#ffe6e6")
            elif "SWITCH" in action_text: cell.set_facecolor("#e6f2ff")
            else: cell.set_facecolor("#e6ffe6")

    ax2.set_title("Top 20 Estados Mais Otimizados", fontsize=16, weight='bold', pad=15)
    
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

    # Nova verificação: Impede a leitura se o arquivo estiver temporariamente zerado
    tamanho_arquivo = os.path.getsize(BRAIN_FILE)
    if tamanho_arquivo == 0:
        print("ERRO DE LEITURA: O arquivo blue_brain.pkl está com 0 bytes no momento.")
        print("Ação: O bot provavelmente está sobrescrevendo o arquivo agora. Aguarde alguns segundos e execute novamente.")
        return

    try:
        with open(BRAIN_FILE, "rb") as f:
            data = pickle.load(f)
        
        q_table = data.get("q_table", {})
        visit_counts = data.get("visit_counts", {})
        epsilon = data.get("epsilon", -1)

        total_states = len(q_table)
        if total_states == 0:
            print("AVISO: A Q-Table extraída está vazia.")
            return

        # Estatísticas de Visita
        visit_stats = {'1': 0, '2_4': 0, '5_19': 0, '20+': 0}
        for v in visit_counts.values():
            if v == 1: visit_stats['1'] += 1
            elif 2 <= v <= 4: visit_stats['2_4'] += 1
            elif 5 <= v <= 19: visit_stats['5_19'] += 1
            else: visit_stats['20+'] += 1

        action_counts = {'attack': 0, 'switch': 0, 'support': 0}
        ranked_states = []

        for state, values_list in q_table.items():
            values = np.array(values_list)
            max_val = np.max(values)
            best_action = np.argmax(values)
            
            act_name = BLUE_ACTIONS[best_action] if best_action < len(BLUE_ACTIONS) else "UNKNOWN"
            
            # Conta com base no NOME da ação, independentemente do índice matemático
            if "ATTACK" in act_name: 
                action_counts['attack'] += 1
            elif "SWITCH" in act_name: 
                action_counts['switch'] += 1
            else: 
                action_counts['support'] += 1
            
            visits = visit_counts.get(state, 0)
            ranked_states.append((max_val, visits, state, best_action))

        # Ordena pelo maior Q-Value para mostrar na tabela
        ranked_states.sort(key=lambda x: x[0], reverse=True)
        top_20 = ranked_states[:20]

        export_data = []
        for val, visits, state, action in top_20:
            state_cols = decode_state_blue_exact(state)
            act_name = BLUE_ACTIONS[action] if action < len(BLUE_ACTIONS) else f"UNKNOWN"
            
            export_data.append({
                "Visitas": visits,
                "Q-Value": f"{val:.2f}",
                "Ação": act_name,
                "Roles": state_cols[0],
                "Matchup": state_cols[1],
                "HP": state_cols[2],
                "Clima": state_cols[3],    # Antes estava perdido, agora está no índice 5 real
                "Speed": state_cols[4],
                "Status": state_cols[5],
                "Boosts": state_cols[6],
                "Hazards": state_cols[7],
                "Mecânica": state_cols[8],
                "Contexto": state_cols[9],
            })
            
        df_top20 = pd.DataFrame(export_data)
        base_name = os.path.join(LOG_DIR, f"analise_blue_{timestamp}")
        
        generate_dashboard(df_top20, action_counts, visit_stats, f"{base_name}_dashboard.png")
        print(f"[RELATÓRIO] Concluído! Verifique a pasta logs_tese.")

    except Exception as e:
        print(f"ERRO: {e}")

if __name__ == "__main__":
    analyze_brain()