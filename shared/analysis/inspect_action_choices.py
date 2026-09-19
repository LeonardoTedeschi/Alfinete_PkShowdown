"""
shared/analysis/inspect_action_choices.py — diagnóstico N(s,a) nos estados maduros.

Pergunta que responde
---------------------
Dois cérebros podem ter a mesma cobertura ponderada por estado e, ainda assim,
experimentar repertórios de ações muito diferentes dentro desses estados. Esta análise
mede as escolhas REAIS registradas por BlueBrain.action_choice_counts depois que cada
estado alcança LIMIAR_MADURO.

A medição é deliberadamente diagnóstica:
- não entra em reward/PBRS;
- não altera Q-values;
- não altera epsilon, replay ou traces;
- não tenta reconstruir cérebros antigos que não possuíam N(s,a).

Saídas
------
<out>/<label>_acoes_estado_dashboard.png
<out>/<label>_acoes_estado.csv
"""

from __future__ import annotations

import csv
import math
import os
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _bit_count(x):
    try:
        return int(x).bit_count()
    except AttributeError:  # Python < 3.8, mantido por robustez
        return bin(int(x)).count("1")


def _metricas_estado(counts, valid_mask, actions):
    arr = np.asarray(counts, dtype=np.uint64)
    total = int(arr.sum())
    if total <= 0:
        return None

    idxs = np.flatnonzero(arr)
    unique = int(len(idxs))
    dominante_idx = int(np.argmax(arr))
    dominante_n = int(arr[dominante_idx])
    dominante_pct = dominante_n / total * 100.0

    p = arr[idxs].astype(float) / total
    entropia = float(-np.sum(p * np.log(p))) if unique else 0.0
    efetivas = float(math.exp(entropia)) if unique else 0.0

    validas = _bit_count(valid_mask)
    cobertura = (unique / validas * 100.0) if validas else 0.0

    return {
        "decisoes": total,
        "acoes_escolhidas": unique,
        "acoes_validas_vistas": validas,
        "cobertura_acoes_pct": cobertura,
        "dominante": actions[dominante_idx] if dominante_idx < len(actions) else str(dominante_idx),
        "dominante_pct": dominante_pct,
        "entropia": entropia,
        "acoes_efetivas": efetivas,
    }


def analyze_action_choices(brain_path, output_dir, label="brain"):
    os.makedirs(output_dir, exist_ok=True)
    with open(brain_path, "rb") as f:
        data = pickle.load(f)

    counts_by_state = data.get("action_choice_counts", {}) or {}
    masks_by_state = data.get("action_valid_masks", {}) or {}
    actions = list(data.get("actions", []) or [])

    if not counts_by_state:
        return "SEM DADOS: este cérebro é anterior à medição N(s,a) de 17/09/2026"

    # Compatibilidade defensiva se a lista de ações não estiver no .pkl.
    if not actions:
        first = next(iter(counts_by_state.values()))
        actions = [f"ACTION_{i}" for i in range(len(first))]

    rows = []
    global_actions = Counter()
    for state, counts in counts_by_state.items():
        m = _metricas_estado(counts, masks_by_state.get(state, 0), actions)
        if not m:
            continue
        arr = np.asarray(counts, dtype=np.uint64)
        for idx in np.flatnonzero(arr):
            global_actions[actions[int(idx)]] += int(arr[int(idx)])
        rows.append((state, m))

    if not rows:
        return "SEM DADOS: não houve decisões registradas após maturidade"

    total_dec = sum(m["decisoes"] for _, m in rows)
    dominant_weighted = sum(m["dominante_pct"] * m["decisoes"] for _, m in rows) / total_dec
    effective_weighted = sum(m["acoes_efetivas"] * m["decisoes"] for _, m in rows) / total_dec
    coverage_weighted = sum(m["cobertura_acoes_pct"] * m["decisoes"] for _, m in rows) / total_dec

    csv_path = os.path.join(output_dir, f"{label}_acoes_estado.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Estado", "Decisoes_Medidas", "Acoes_Escolhidas", "Acoes_Validas_Vistas",
            "Cobertura_Acoes_pct", "Acao_Dominante", "Dominante_pct",
            "Entropia", "Acoes_Efetivas",
        ])
        for state, m in sorted(rows, key=lambda x: x[1]["decisoes"], reverse=True):
            w.writerow([
                repr(state), m["decisoes"], m["acoes_escolhidas"], m["acoes_validas_vistas"],
                f"{m['cobertura_acoes_pct']:.3f}", m["dominante"],
                f"{m['dominante_pct']:.3f}", f"{m['entropia']:.6f}",
                f"{m['acoes_efetivas']:.4f}",
            ])

    # Dashboard com três visões complementares.
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    ax1 = fig.add_subplot(gs[0, 0])
    dom = [m["dominante_pct"] for _, m in rows]
    weights = [m["decisoes"] for _, m in rows]
    ax1.hist(dom, bins=[0, 50, 60, 70, 80, 90, 95, 100.01], weights=weights)
    ax1.set_title("Concentração da ação dominante\n(ponderada por decisões medidas)")
    ax1.set_xlabel("% das decisões do estado na ação mais escolhida")
    ax1.set_ylabel("Decisões")

    ax2 = fig.add_subplot(gs[0, 1])
    eff = [m["acoes_efetivas"] for _, m in rows]
    ax2.hist(eff, bins=np.arange(1, max(3.0, min(12.0, max(eff) + 1.0)), 0.5), weights=weights)
    ax2.set_title("Número efetivo de ações por estado\nexp(entropia de escolhas)")
    ax2.set_xlabel("Ações efetivas")
    ax2.set_ylabel("Decisões")

    ax3 = fig.add_subplot(gs[1, :])
    top = global_actions.most_common(18)
    labels = [a for a, _ in top]
    vals = [n / total_dec * 100.0 for _, n in top]
    ax3.bar(labels, vals)
    ax3.set_title("Distribuição global das decisões em estados maduros medidos")
    ax3.set_ylabel("% das decisões")
    ax3.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"{label} — DIVERSIDADE DE AÇÕES DENTRO DOS ESTADOS\n"
        f"{len(rows):,} estados | {total_dec:,} decisões | "
        f"dominante={dominant_weighted:.1f}% | ações efetivas={effective_weighted:.2f} | "
        f"cobertura das ações vistas={coverage_weighted:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(output_dir, f"{label}_acoes_estado_dashboard.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"[N(s,a)] CSV: {csv_path}")
    print(f"[N(s,a)] Dashboard: {png_path}")

    return png_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analisa N(s,a) em estados maduros")
    parser.add_argument("brain", help="caminho para blue_brain.pkl ou green_brain.pkl")
    parser.add_argument("--out", default="artefatos/logs/analise")
    parser.add_argument("--label", default="brain")
    args = parser.parse_args()
    print(analyze_action_choices(args.brain, args.out, args.label))
