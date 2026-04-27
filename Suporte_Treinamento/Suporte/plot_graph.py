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
        ids = []
        for path in existing_logs:
            match = re.search(r'log_(\d+).csv', path)
            if match:
                ids.append(int(match.group(1)))

        if ids:
            next_id = max(ids) + 1

    print(f"[Gerente] Configurando sessão ID: {next_id}")

    paths = {
        "id": next_id,
        "folder": base_folder,
        "csv": os.path.join(base_folder, f"log_{next_id}.csv"),
        "graph": os.path.join(base_folder, f"grafico_{next_id}.png"),
        "model": "blue_brain.pkl"
    }

    print(f"[Gerente] Log atual: {paths['csv']}")
    return paths

def generate_graph(csv_path, img_output_path, title_suffix="", opponent="Desconhecido", 
                   phase="N/A", total_battles=0, final_win_rate=0.0, final_states=0):
    """
    Gera um gráfico de treinamento com Win Rate, Recompensa Acumulada e Epsilon.

    Args:
        csv_path: Caminho do CSV de log
        img_output_path: Caminho para salvar a imagem
        title_suffix: Sufixo do título (geralmente o ID da sessão)
        opponent: Nome do oponente treinado
        phase: Fase do circuito (maxdamage, instinct, selfplay)
        total_battles: Total de batalhas completadas
        final_win_rate: Win rate final da sessão (%)
        final_states: Quantidade de estados na Q-table ao final
    """
    if not os.path.exists(csv_path):
        print(f"[Gerente] AVISO: CSV não encontrado: {csv_path}")
        return

    battles = []
    win_rates = []
    rewards = []
    epsilons = []

    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                try:
                    battles.append(int(row[0]))
                    win_rates.append(float(row[1]))
                    epsilons.append(float(row[2]))
                    if len(row) > 3:
                        rewards.append(float(row[3]))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"[Gerente] Erro ao ler CSV: {e}")
        return

    if not battles:
        print("[Gerente] Nenhum dado válido para plotar.")
        return

    # Configuração de estilo
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.facecolor': '#f8f9fa',
        'axes.facecolor': '#ffffff',
        'axes.edgecolor': '#dee2e6',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#adb5bd',
    })

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Eixo Esquerdo: Win Rate ---
    color_wr = '#2563eb'
    ax1.set_xlabel('Batalhas Completadas', fontweight='bold', color='#495057')
    ax1.set_ylabel('Win Rate do Bloco (%)', color=color_wr, fontweight='bold')

    line_wr = ax1.plot(battles, win_rates, color=color_wr, linewidth=2.5, 
                       marker='o', markersize=4, label='Win Rate (Bloco)', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color_wr)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, max(battles) * 1.05)

    # Linha de referência em 50%
    ax1.axhline(y=50, color='#6c757d', linestyle='--', alpha=0.4, linewidth=1, label='Linha de Base (50%)')

    # --- Eixo Direito: Reward e Epsilon ---
    ax2 = ax1.twinx()

    lines = line_wr

    if rewards:
        color_rwd = '#dc2626'
        ax2.set_ylabel('Recompensa Total', color=color_rwd, fontweight='bold')
        line_rwd = ax2.plot(battles, rewards, color=color_rwd, linewidth=2.0, 
                            linestyle='--', alpha=0.7, label='Recompensa Total', zorder=2)
        ax2.tick_params(axis='y', labelcolor=color_rwd)
        lines += line_rwd

    if epsilons:
        color_eps = '#059669'
        # Epsilon usa escala logarítmica ou linear secundária? Vamos usar o mesmo eixo direito
        # mas com valores pequenos, então normalizamos visualmente ou deixamos sobreposto
        # Como epsilon vai de 0-0.5 e reward vai de milhões, precisamos de um terceiro eixo
        # ou plotar epsilon no eixo esquerdo com escala secundária.
        # Solução: Criar um terceiro eixo invisível para Epsilon.
        pass

    # --- Terceiro Eixo para Epsilon (escala independente) ---
    if epsilons:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Desloca o eixo para a direita
        color_eps = '#059669'
        line_eps = ax3.plot(battles, epsilons, color=color_eps, linewidth=2.0, 
                            linestyle='-.', alpha=0.8, marker='s', markersize=3,
                            label='Epsilon (Exploração)', zorder=2)
        ax3.set_ylabel('Epsilon', color=color_eps, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor=color_eps)
        ax3.set_ylim(0, max(epsilons) * 1.2 if epsilons else 1.0)
        lines += line_eps

    # --- Título e Metadados ---
    title = f"Sessão {title_suffix} — Fase: {phase.upper()} | Oponente: {opponent.upper()}"
    ax1.set_title(title, fontweight='bold', color='#212529', pad=20)

    # Caixa de texto com estatísticas finais
    stats_text = (
        f"Batalhas: {total_battles:,}\n"
        f"Win Rate Final: {final_win_rate:.1f}%\n"
        f"Estados Q-Table: {final_states:,}\n"
        f"Epsilon Final: {epsilons[-1]:.3f}" if epsilons else ""
    ).strip()

    if stats_text:
        props = dict(boxstyle='round,pad=0.5', facecolor='#e9ecef', alpha=0.9, edgecolor='#adb5bd')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props, family='monospace', color='#343a40')

    # --- Legenda combinada ---
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', framealpha=0.9, edgecolor='#dee2e6')

    plt.tight_layout()
    plt.savefig(img_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Gerente] Gráfico salvo: {img_output_path}")