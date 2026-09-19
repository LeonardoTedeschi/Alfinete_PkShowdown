"""
shared/analysis/inspect_brain.py — dashboard de análise de uma Q-table treinada.

Gera uma imagem com: proporção de melhores ações (ataque/troca/suporte), histograma
de profundidade de aprendizado (visitas por estado) e tabela dos 20 estados mais
otimizados. Serve tanto o Blue como o Green — basta apontar para o .pkl respetivo.

Executar a partir da RAIZ do projeto:
    python -m shared.analysis.inspect_brain --brain artefatos/brains/blue_brain.pkl
    python -m shared.analysis.inspect_brain --brain artefatos/brains/green_brain.pkl

Correções face à versão antiga:
  - Caminhos por argumento (não hard-coded para a pasta 'Instinto' que já não existe).
  - A lista de ações é IMPORTADA do BlueBrain (fonte única de verdade), em vez de
    duplicada aqui — se as ações mudarem no cérebro, o dashboard acompanha.
  - O decode do estado usa a ordem REAL do StateParser (a versão antiga tinha as
    colunas trocadas: rotulava epsilon como clima, etc.).
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fonte única das ações: o próprio cérebro. Garante que o dashboard nunca fica
# dessincronizado do espaço de ações real.
from qlearning.brain import BlueBrain

ACTIONS = BlueBrain().actions  # 36 ações: 19 base, 17 variantes _MEC

# Dimensao importada do StateParser em vez de repetida aqui: e a mesma razao pela
# qual ACTIONS vem do BlueBrain. Quando o estado cresceu de 15 para 16 dimensoes
# (30/08/2026), uma constante local desactualizada faria a coluna nova desaparecer do
# dashboard SEM dar erro nenhum.
from shared.state import STATE_DIM

# Nomes das dimensões do estado, na ORDEM REAL do StateParser.
STATE_LABELS = [
    "my_role", "opp_role", "matchup", "my_hp", "opp_hp",
    "weather", "speed", "mechanic", "my_status", "opp_status",
    "my_boost", "opp_boost", "my_hazard", "opp_hazard", "macro",
    "banco",
]
assert len(STATE_LABELS) == STATE_DIM, (
    f"STATE_LABELS tem {len(STATE_LABELS)} nomes e STATE_DIM e {STATE_DIM}. "
    "Actualize os dois em conjunto.")


def decode_state(state_tuple):
    """Converte a tupla de estado em colunas legíveis, na ordem correta."""
    s = list(state_tuple)
    while len(s) < STATE_DIM:
        s.append("?")
    return {
        "Roles": f"{s[0]} v {s[1]}",
        "Matchup": str(s[2]).replace('OFFENSIVE_', 'OFF_').replace('DEFENSIVE_', 'DEF_'),
        "HP": f"{s[3]} v {s[4]}",
        "Clima": str(s[5]),
        "Speed": str(s[6]),
        "Mec": str(s[7]),
        "Status": f"{s[8]} v {s[9]}",
        "Boosts": f"{s[10]} v {s[11]}",
        "Hazards": f"{s[12]} v {s[13]}",
        "Contexto": str(s[14]),
        # Dimensao 15, acrescentada em 30/08/2026. E a unica que diz alguma coisa
        # sobre o BANCO, e sem ela a coluna de accoes SWITCH_* do dashboard fica sem
        # explicacao: o mesmo estado tanto podia ter um contra-tanque a espera como
        # cinco Pokemon mortos.
        "Banco": str(s[15]).replace("BANCO_", ""),
    }


def generate_dashboard(df, action_counts, visit_stats, massa, cobertura_pond,
                       filename, agent_name):
    fig = plt.figure(figsize=(26, 16))
    gs = GridSpec(2, 2, height_ratios=[1.2, 4], width_ratios=[1, 1.2], hspace=0.15)
    fig.patch.set_facecolor('#f4f4f9')
    plt.suptitle(f"ANÁLISE DO MODELO {agent_name.upper()} - {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                 fontsize=22, weight='bold', color='#333333', y=0.96)

    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Ataque', 'Troca', 'Suporte']
    sizes = [action_counts['attack'], action_counts['switch'], action_counts['support']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    fsz = [s for s in sizes if s > 0]
    flb = [l for s, l in zip(sizes, labels) if s > 0]
    fcl = [c for s, c in zip(sizes, colors) if s > 0]
    if sum(sizes) > 0:
        ax1.pie(fsz, labels=flb, colors=fcl, autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'black'}, textprops={'weight': 'bold', 'fontsize': 11})
        ax1.set_title("Proporção de Melhores Ações", fontsize=14, weight='bold', pad=10)
        ax1.set_aspect('equal')

    ax3 = fig.add_subplot(gs[0, 1])
    bars = ['1 Visita', '2 a 4\n(Mestre-Aluno)', '5 a 19\n(Exploração)', '20+\n(Maduro)']
    counts = [visit_stats['1'], visit_stats['2_4'], visit_stats['5_19'], visit_stats['20+']]
    bar_colors = ['#ff6666', '#ffcc66', '#99ccff', '#66cc66']
    x_pos = np.arange(len(bars))
    ax3.bar(x_pos, counts, color=bar_colors, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bars, fontsize=10, weight='bold')
    ax3.set_title(f"Profundidade de Aprendizado  |  COBERTURA PONDERADA: {cobertura_pond:.1f}% "
                  f"das decisoes em estados maduros",
                  fontsize=13, weight='bold', pad=10)
    ax3.set_ylabel("Quantidade de Estados", fontsize=12, weight='bold')
    # Duas percentagens por barra: quantos ESTADOS e, por baixo, quanta DECISAO.
    # A segunda e a que decide, e sem ela a primeira engana: o balde de 1 visita
    # parece enorme (25% dos estados) e vale 0,7% do jogo.
    total_counts = max(1, sum(counts))
    total_m = max(1, sum(massa.values()))
    massa_ord = [massa['1'], massa['2_4'], massa['5_19'], massa['20+']]
    for i, (v, m) in enumerate(zip(counts, massa_ord)):
        ax3.text(i, v + (max(counts) * 0.02 if counts else 1),
                 f"{v:,}\n{v/total_counts*100:.1f}% estados\n{m/total_m*100:.1f}% DECISOES",
                 ha='center', va='bottom', weight='bold', fontsize=9)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')
    if not df.empty:
        the_table = ax2.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(),
                              loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.8)
        for (i, j), cell in the_table.get_celld().items():
            if i == 0:
                cell.set_text_props(color='white', weight='bold')
                cell.set_facecolor("#2c3e50")
            else:
                action_text = df.values.tolist()[i-1][2]
                if "ATTACK" in action_text:
                    cell.set_facecolor("#ffe6e6")
                elif "SWITCH" in action_text:
                    cell.set_facecolor("#e6f2ff")
                else:
                    cell.set_facecolor("#e6ffe6")
    ax2.set_title("Top 20 Estados Mais Otimizados", fontsize=16, weight='bold', pad=15)

    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.02, right=0.98)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[IMAGEM] Dashboard salvo em: {filename}")


def analyze_brain(brain_file, out_dir, agent_name):
    if not os.path.exists(brain_file):
        print(f"ERRO: ficheiro não encontrado: {brain_file}")
        return
    if os.path.getsize(brain_file) == 0:
        print("ERRO: o .pkl está com 0 bytes (o bot deve estar a reescrevê-lo). Tenta de novo.")
        return

    with open(brain_file, "rb") as f:
        data = pickle.load(f)
    q_table = data.get("q_table", {})
    visit_counts = data.get("visit_counts", {})
    if not q_table:
        print("AVISO: Q-table vazia.")
        return

    # CONTAGEM DE ESTADOS **E** DE MASSA DE DECISAO (30/08/2026).
    #
    # A versao anterior contava so ESTADOS por balde, e essa e a metrica errada:
    # medido nos cerebros de 400k, 16,7% dos estados absorvem 90,7% de todas as
    # decisoes. "Um quarto dos estados tem uma visita" soa mal; "0,7% das decisoes
    # acontecem nesses estados" e a mesma realidade e leva a conclusao oposta.
    #
    # BUG CORRIGIDO no mesmo sitio: o `else` final apanhava tambem os estados com
    # ZERO visitas e contava-os como maduros. Havia exactamente um (a chave `None`
    # criada pelo bug da troca forcada), e era por isso que o dashboard dizia 71.524
    # maduros quando `>= 20` da 71.523.
    visit_stats = {'1': 0, '2_4': 0, '5_19': 0, '20+': 0}
    massa = {'1': 0, '2_4': 0, '5_19': 0, '20+': 0}
    for v in visit_counts.values():
        if v <= 0:
            continue          # estado criado e nunca actualizado: nao e maduro
        if v == 1:
            chave = '1'
        elif 2 <= v <= 4:
            chave = '2_4'
        elif 5 <= v <= 19:
            chave = '5_19'
        else:
            chave = '20+'
        visit_stats[chave] += 1
        massa[chave] += v
    total_massa = max(1, sum(massa.values()))
    cobertura_pond = massa['20+'] / total_massa * 100.0

    action_counts = {'attack': 0, 'switch': 0, 'support': 0}
    ranked = []
    for state, values in q_table.items():
        values = np.array(values)
        best = int(np.argmax(values))
        act = ACTIONS[best] if best < len(ACTIONS) else "UNKNOWN"
        if "ATTACK" in act:
            action_counts['attack'] += 1
        elif "SWITCH" in act:
            action_counts['switch'] += 1
        else:
            action_counts['support'] += 1
        ranked.append((float(np.max(values)), visit_counts.get(state, 0), state, best))

    ranked.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for val, visits, state, action in ranked[:20]:
        cols = decode_state(state)
        rows.append({"Visitas": visits, "Q-Value": f"{val:.2f}",
                     "Ação": ACTIONS[action] if action < len(ACTIONS) else "UNKNOWN", **cols})
    df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    # Nome deterministico por marco (ex.: Blue_050k). Reexecutar a mesma analise
    # substitui a fotografia daquele marco em vez de criar dezenas de timestamps.
    out = os.path.join(out_dir, f"analise_{agent_name}_dashboard.png")
    generate_dashboard(df, action_counts, visit_stats, massa, cobertura_pond,
                       out, agent_name)
    print(f"[RELATÓRIO] Concluído. Estados: {len(q_table):,} | "
          f"maduros: {visit_stats['20+']:,} | "
          f"cobertura ponderada: {cobertura_pond:.2f}% das decisões")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="artefatos/brains/blue_brain.pkl", help="caminho do .pkl")
    ap.add_argument("--out", default="artefatos/analise", help="pasta de saída")
    ap.add_argument("--name", default=None, help="nome do agente (auto do ficheiro se omitido)")
    args = ap.parse_args()
    name = args.name or os.path.basename(args.brain).replace("_brain.pkl", "").replace(".pkl", "")
    analyze_brain(args.brain, args.out, name)
