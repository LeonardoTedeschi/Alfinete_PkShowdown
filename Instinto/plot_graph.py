import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import glob
import re

def setup_training_files(base_folder="historico_treinos"):
    """
    Organiza os arquivos em uma ÚNICA pasta, incrementando apenas o sufixo.
    Ex: log_1.csv, log_2.csv, etc.
    """
    os.makedirs(base_folder, exist_ok=True)
    
    # Procura arquivos existentes para definir o próximo ID
    existing_logs = glob.glob(os.path.join(base_folder, "log_*.csv"))
    
    next_id = 1
    if existing_logs:
        # Extrai os números dos arquivos existentes para achar o maior
        ids = []
        for path in existing_logs:
            # Regex para pegar o número entre 'log_' e '.csv'
            match = re.search(r'log_(\d+).csv', path)
            if match:
                ids.append(int(match.group(1)))
        
        if ids:
            next_id = max(ids) + 1
            
    print(f"[Gerente] Configurando sessão ID: {next_id}")
    
    paths = {
        "id": next_id,
        "folder": base_folder,
        # Arquivos da Sessão (Numerados)
        "csv": os.path.join(base_folder, f"log_{next_id}.csv"),
        "graph": os.path.join(base_folder, f"grafico_{next_id}.png"),
        # Arquivo Compartilhado (Sempre o mesmo na raiz)
        "model": "red_brain.pkl"
    }
    
    print(f"[Gerente] Log atual: {paths['csv']}")
    return paths

def generate_graph(csv_path, img_output_path, title_suffix=""):
    if not os.path.exists(csv_path): return

    battles = []
    win_rates = []
    rewards = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row: continue
                battles.append(int(row[0]))
                win_rates.append(float(row[1]))
                if len(row) > 3:
                    rewards.append(float(row[3]))
    except Exception: return

    if not battles: return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Batalhas')
    ax1.set_ylabel('Win Rate (%)', color=color, fontweight='bold')
    ax1.plot(battles, win_rates, color=color, marker='o', markersize=3, label='Win Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    if rewards:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Avg Reward', color=color, fontweight='bold')
        ax2.plot(battles, rewards, color=color, linestyle='--', alpha=0.5, label='Reward')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Treino ID: {title_suffix}')
    plt.tight_layout()
    plt.savefig(img_output_path)
    plt.close()
    print(f"[Gerente] Gráfico salvo: {img_output_path}")