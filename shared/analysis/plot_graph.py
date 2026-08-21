"""
shared/analysis/plot_graph.py — grafico de evolucao de uma sessao de treino.

Plota Win Rate (por bloco), Recompensa e Epsilon num grafico de tres eixos.
Serve Blue e Green. Chamado ao fim de cada treino, ou avulso sobre um CSV.

LEITURA POR NOME DE COLUNA (correcao importante): a versao anterior lia as colunas
por POSICAO (row[2] como epsilon, row[3] como reward). Quando o CSV passou a ter
Vitorias e Derrotas nas posicoes 2 e 3, o grafico passaria a plotar dados errados
SEM dar erro. Agora as colunas sao localizadas pelo cabecalho, o que torna o modulo
imune a mudancas de ordem e a colunas novas.

Colunas esperadas (por nome; extras sao ignoradas, ausentes ficam vazias):
    Batalhas, WinRate_Bloco, Epsilon, Reward, Estados_Q, Latencia_ms,
    Margem_Media, Duracao_Media, Auto_Ties, Tempo_s

Uso avulso, da raiz do projeto:
    python -m shared.analysis.plot_graph --csv artefatos/logs/blue_treino_01.csv \
        --out artefatos/logs/blue_grafico_01.png --agent Blue
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Nomes aceites para cada serie (o primeiro que existir no cabecalho e usado).
_ALIASES = {
    "batalhas": ["Batalhas", "battles"],
    "win_rate": ["WinRate_Bloco", "win_rate", "WinRate"],
    "epsilon": ["Epsilon", "epsilon"],
    "reward": ["Reward", "reward"],
    "estados": ["Estados_Q", "states", "Estados"],
}


def _indices(cabecalho):
    """Mapeia cada serie ao indice da sua coluna, pelo nome no cabecalho."""
    mapa = {}
    if not cabecalho:
        return mapa
    normalizado = [c.strip() for c in cabecalho]
    for chave, nomes in _ALIASES.items():
        for nome in nomes:
            if nome in normalizado:
                mapa[chave] = normalizado.index(nome)
                break
    return mapa


def ler_csv(csv_path):
    """Le o CSV por nome de coluna. Devolve dict de listas (series encontradas)."""
    series = {k: [] for k in _ALIASES}
    with open(csv_path, "r", newline="") as f:
        leitor = csv.reader(f)
        cabecalho = next(leitor, None)
        idx = _indices(cabecalho)
        if "batalhas" not in idx or "win_rate" not in idx:
            raise ValueError(
                "CSV sem as colunas minimas 'Batalhas' e 'WinRate_Bloco'. "
                f"Cabecalho encontrado: {cabecalho}")
        for linha in leitor:
            if not linha:
                continue
            try:
                b = int(float(linha[idx["batalhas"]]))
            except (ValueError, IndexError):
                continue
            series["batalhas"].append(b)
            for chave in ("win_rate", "epsilon", "reward", "estados"):
                if chave in idx and idx[chave] < len(linha):
                    try:
                        series[chave].append(float(linha[idx[chave]]))
                    except ValueError:
                        series[chave].append(float("nan"))
    return series


def setup_training_files(base_folder, agent_name):
    """Descobre o proximo ID de sessao para um agente (ficheiros por agente)."""
    os.makedirs(base_folder, exist_ok=True)
    prefixo = agent_name.lower()
    existentes = glob.glob(os.path.join(base_folder, f"{prefixo}_treino_*.csv"))
    ids = []
    for caminho in existentes:
        m = re.fullmatch(rf"{prefixo}_treino_(\d+)\.csv", os.path.basename(caminho))
        if m:
            ids.append(int(m.group(1)))
    proximo = (max(ids) + 1) if ids else 1
    return {
        "id": proximo,
        "folder": base_folder,
        "csv": os.path.join(base_folder, f"{prefixo}_treino_{proximo:02d}.csv"),
        "graph": os.path.join(base_folder, f"{prefixo}_grafico_{proximo:02d}.png"),
        "model": f"{prefixo}_brain.pkl",
    }


def generate_graph(csv_path, img_output_path, agent="Agente", opponent="Instinto",
                   phase="instinct", total_battles=0, final_win_rate=0.0,
                   final_states=0, title_suffix=None):
    """Gera o grafico da sessao. `title_suffix` existe apenas por compatibilidade
    com a assinatura antiga (se dado e agent nao, e usado como nome do agente)."""
    if title_suffix and agent == "Agente":
        agent = str(title_suffix)

    if not os.path.exists(csv_path):
        print(f"[GRAFICO] AVISO: CSV nao encontrado: {csv_path}")
        return

    try:
        s = ler_csv(csv_path)
    except Exception as e:
        print(f"[GRAFICO] AVISO: falha a ler o CSV: {e}")
        return

    battles = s["batalhas"]
    win_rates = s["win_rate"]
    epsilons = s["epsilon"]
    rewards = s["reward"]

    if not battles or not win_rates:
        print("[GRAFICO] AVISO: sem dados validos para plotar.")
        return

    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 14, 'axes.labelsize': 11, 'legend.fontsize': 9,
        'figure.facecolor': '#f8f9fa', 'axes.facecolor': '#ffffff', 'axes.edgecolor': '#dee2e6',
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#adb5bd',
    })
    fig, ax1 = plt.subplots(figsize=(12, 7))

    cor_wr = '#2563eb'
    ax1.set_xlabel('Batalhas Completadas', fontweight='bold', color='#495057')
    ax1.set_ylabel('Win Rate do Bloco (%)', color=cor_wr, fontweight='bold')
    linhas = ax1.plot(battles, win_rates, color=cor_wr, linewidth=2.5, marker='o',
                      markersize=4, label='Win Rate (Bloco)', zorder=3)
    ax1.tick_params(axis='y', labelcolor=cor_wr)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, max(battles) * 1.05)
    ax1.axhline(y=50, color='#6c757d', linestyle='--', alpha=0.5, linewidth=1.2,
                label='Equilibrio (50%)')

    if rewards and len(rewards) == len(battles):
        ax2 = ax1.twinx()
        cor_r = '#dc2626'
        ax2.set_ylabel('Recompensa', color=cor_r, fontweight='bold')
        linhas += ax2.plot(battles, rewards, color=cor_r, linewidth=2.0, linestyle='--',
                           alpha=0.7, label='Recompensa', zorder=2)
        ax2.tick_params(axis='y', labelcolor=cor_r)

    if epsilons and len(epsilons) == len(battles):
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        cor_e = '#059669'
        linhas += ax3.plot(battles, epsilons, color=cor_e, linewidth=2.0, linestyle='-.',
                           alpha=0.85, marker='s', markersize=3, label='Epsilon', zorder=2)
        ax3.set_ylabel('Epsilon (exploracao)', color=cor_e, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor=cor_e)
        ax3.set_ylim(0, max(epsilons) * 1.2 if max(epsilons) > 0 else 1.0)

    ax1.set_title(f"{agent} — Fase: {str(phase).upper()} | Oponente: {str(opponent).upper()}",
                  fontweight='bold', color='#212529', pad=20)

    wr_final = final_win_rate or (win_rates[-1] if win_rates else 0.0)
    estados_final = final_states or (int(s["estados"][-1]) if s["estados"] else 0)
    total = total_battles or (battles[-1] if battles else 0)
    texto = (f"Batalhas: {total:,}\n"
             f"Win Rate final: {wr_final:.1f}%\n"
             f"Estados Q-table: {estados_final:,}")
    if epsilons:
        texto += f"\nEpsilon final: {epsilons[-1]:.3f}"
    ax1.text(0.02, 0.98, texto, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', family='monospace', color='#343a40',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e9ecef', alpha=0.9,
                       edgecolor='#adb5bd'))

    ax1.legend(linhas, [l.get_label() for l in linhas], loc='lower right',
               framealpha=0.9, edgecolor='#dee2e6')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(img_output_path)), exist_ok=True)
    plt.savefig(img_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[GRAFICO] Salvo: {img_output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default="Agente")
    ap.add_argument("--opponent", default="Instinto")
    ap.add_argument("--phase", default="instinct")
    args = ap.parse_args()
    generate_graph(args.csv, args.out, agent=args.agent, opponent=args.opponent,
                   phase=args.phase)
