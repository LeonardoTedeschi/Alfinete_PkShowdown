"""
scripts/analisar_estado.py — o que cada dimensao do estado esta a fazer pela politica.

LEITURA PURA. Nao treina, nao corre batalhas, nao altera o .pkl.

O QUE MUDOU NA VERSAO 2 (26/08/2026)
------------------------------------
A versao 1 tinha dois defeitos, ambos medidos nos cerebros v7/v4:

  1. BUG DE DESEMPENHO: fazia `k not in lista` sobre 340 mil elementos para cada uma
     das 340 mil chaves. 10^11 comparacoes. O script parecia travado.

  2. METRICA INUTIL: media com que frequencia a MELHOR ACAO (argmax) mudava entre
     estados irmaos. Deu 78% a 91% para TODAS as 15 dimensoes, o que nao discrimina
     nada. Causa: com 36 acoes e valores proximos no topo, o argmax e instavel;
     estava a medir esparsidade da Q-table, nao relevancia da dimensao.

A versao 2 usa PERDA DE TRANSFERENCIA, que olha a MAGNITUDE e nao ao lugar:

     para dois estados irmaos A e B (identicos menos na dimensao d),
     quanto se PERDE ao aplicar em B a melhor acao de A?

     perda = (Q_B[melhor de B] - Q_B[melhor de A]) / (Q_B[melhor] - media(Q_B))

  perda ~0   -> a acao de A serve em B: a dimensao d nao muda o que importa
  perda alta -> a dimensao d altera materialmente a decisao

E interpretavel porque vem com CONTROLO: a mesma perda medida entre pares de estados
ao acaso da o valor de "duas situacoes sem relacao nenhuma". Uma dimensao so conta se
a perda dela se aproximar desse teto.

A SECCAO 2 e a mais acionavel e nao depende de nenhuma metrica: mostra, por dimensao,
que fracao dos estados e da EXPERIENCIA esta em cada valor. Resolucao que existe na
tupla mas nunca e exercitada aparece ali, e diz diretamente que tipo de times faltam
no pool de treino.

USO
---
    python -m scripts.analisar_estado --agente blue
    python -m scripts.analisar_estado --agente green --visitas-min 20
    python -m scripts.analisar_estado --cerebro artefatos/brains/blue_brain.pkl
"""

import argparse
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BRAINS_DIR = os.path.join(ROOT, "artefatos", "brains")

# Contrato de shared/state.py (STATE_DIM = 15).
DIMENSOES = [
    "my_role", "opp_role", "matchup", "my_hp", "opp_hp",
    "weather", "speed_tier", "mechanic", "my_status", "opp_status",
    "my_boost", "opp_boost", "my_hazards", "opp_hazards", "macro",
]
N_DIM = len(DIMENSOES)
L = 96


def secao(t):
    print()
    print("=" * L)
    print(f"  {t}")
    print("=" * L)


def carregar(caminho):
    with open(caminho, "rb") as f:
        d = pickle.load(f)
    q = d.get("q_table")
    if q is None:
        raise SystemExit(f"Sem q_table. Chaves no ficheiro: {list(d)}")
    vc = d.get("visit_counts") or d.get("visits") or {}
    return q, vc


def total_visitas(vc, chave):
    v = vc.get(chave)
    if v is None:
        return 0
    return int(np.sum(v)) if hasattr(v, "__iter__") else int(v)


def perda(qb, acao_a):
    """Perda normalizada de aplicar `acao_a` no estado B. 0 = nenhuma."""
    melhor = float(qb.max())
    escala = melhor - float(qb.mean())
    if escala <= 1e-9:
        return None                      # estado plano: nao mede nada
    return (melhor - float(qb[acao_a])) / escala


def main(args):
    caminho = args.cerebro or os.path.join(BRAINS_DIR, f"{args.agente}_brain.pkl")
    if not os.path.exists(caminho):
        raise SystemExit(f"Cerebro nao encontrado: {caminho}")

    print(f"\nA ler {caminho} ...")
    q, vc = carregar(caminho)

    # Saneamento em O(n): filtra chaves que nao sao tuplas de 15 elementos.
    estados, invalidas = [], []
    for k in q:
        (estados if (isinstance(k, tuple) and len(k) == N_DIM) else invalidas).append(k)

    # ---------------- 1 ----------------
    secao("1. INVENTARIO")
    print(f"  Estados validos           : {len(estados):,}")
    if invalidas:
        print(f"  Chaves INVALIDAS          : {len(invalidas)}  (excluidas da analise)")
        for k in invalidas[:5]:
            print(f"      {k!r} (tipo {type(k).__name__}) -> "
                  f"{total_visitas(vc, k)} visitas")

    visitas = {e: total_visitas(vc, e) for e in estados}
    total = sum(visitas.values())
    maduros = [e for e in estados if visitas[e] >= args.visitas_min]
    massa = sum(visitas[e] for e in maduros)
    print(f"  Visitas totais            : {total:,}")
    print(f"  Media por estado          : {total / max(1, len(estados)):.1f}")
    print(f"  Estados com >= {args.visitas_min} visitas  : {len(maduros):,} "
          f"({100.0 * len(maduros) / max(1, len(estados)):.1f}%)")
    print(f"  MASSA de experiencia neles: {100.0 * massa / max(1, total):.1f}%")
    print()
    print("  A ultima linha e a que importa. Uma minoria de estados maduros pode")
    print("  concentrar a esmagadora maioria da experiencia: a cauda de estados pobres")
    print("  sao situacoes que ocorreram uma ou duas vezes e nao se repetem.")

    # ---------------- 2 ----------------
    secao("2. ONDE ESTA A EXPERIENCIA (a seccao acionavel)")
    print("  Por dimensao: fracao dos ESTADOS e fracao das VISITAS em cada valor.")
    print("  Valores com muitos estados e pouca visita sao resolucao paga e vazia.")
    alvos = []
    for i in range(N_DIM):
        c_est, c_vis = Counter(), Counter()
        for e in estados:
            c_est[e[i]] += 1
            c_vis[e[i]] += visitas[e]
        print()
        print(f"  [{i:>2}] {DIMENSOES[i]}   ({len(c_est)} valores)")
        for val, n in c_est.most_common():
            pe = 100.0 * n / len(estados)
            pv = 100.0 * c_vis[val] / max(1, total)
            print(f"        {str(val)[:20]:<20} estados {pe:>5.1f}%   "
                  f"visitas {pv:>5.1f}%  {'#' * int(pv / 2)}")
        dom_val = c_est.most_common(1)[0][0]
        dom_vis = 100.0 * c_vis[dom_val] / max(1, total)
        if dom_vis >= 55.0 and len(c_est) >= 3:
            alvos.append((dom_vis, DIMENSOES[i], dom_val, len(c_est)))

    # ---------------- 3 ----------------
    secao(f"3. PERDA DE TRANSFERENCIA (estados com >= {args.visitas_min} visitas)")
    melhores = {}
    for e in maduros:
        a = q[e]
        if float(a.max()) != float(a.min()):
            melhores[e] = int(a.argmax())
    print(f"  Base: {len(melhores):,} estados maduros com preferencia definida.")

    random.seed(42)
    chaves = list(melhores)
    amostras = []
    if chaves:
        for _ in range(min(20000, len(chaves) * 2)):
            a, b = random.choice(chaves), random.choice(chaves)
            if a == b:
                continue
            p = perda(q[b], melhores[a])
            if p is not None:
                amostras.append(p)
    teto = float(np.mean(amostras)) if amostras else 1.0
    print(f"  CONTROLO (pares ao acaso): perda media {teto:.3f}")
    print("  E o valor de aplicar a acao de uma situacao sem relacao nenhuma. Uma")
    print("  dimensao so e relevante se a perda dela se aproximar deste teto.")
    print()
    print(f"  {'#':>2} {'dimensao':<13} {'pares':>10} {'perda':>8} {'% do teto':>10}   avaliacao")
    print("  " + "-" * (L - 4))

    linhas = []
    for i in range(N_DIM):
        grupos = defaultdict(list)
        for e in melhores:
            grupos[e[:i] + e[i + 1:]].append(e)
        vals = []
        for irmaos in grupos.values():
            if len(irmaos) < 2:
                continue
            amostra = irmaos[:4]           # limita a explosao combinatoria
            for a in amostra:
                for b in amostra:
                    if a == b:
                        continue
                    p = perda(q[b], melhores[a])
                    if p is not None:
                        vals.append(p)
        media = float(np.mean(vals)) if vals else 0.0
        rel = 100.0 * media / teto if teto else 0.0
        if rel >= 70:
            av = "decisiva"
        elif rel >= 45:
            av = "relevante"
        elif rel >= 25:
            av = "marginal"
        else:
            av = "quase transparente"
        linhas.append((rel, i, media, len(vals)))
        print(f"  {i:>2} {DIMENSOES[i]:<13} {len(vals):>10,} {media:>8.3f} "
              f"{rel:>9.1f}%   {av}")

    # ---------------- 4 ----------------
    secao("4. SINTESE")
    linhas.sort()
    print("  Dimensoes que MENOS alteram a politica:")
    for rel, i, m, n in linhas[:4]:
        print(f"    {DIMENSOES[i]:<13} {rel:>5.1f}% do teto")
    print()
    print("  Dimensoes que MAIS alteram a politica:")
    for rel, i, m, n in sorted(linhas, reverse=True)[:4]:
        print(f"    {DIMENSOES[i]:<13} {rel:>5.1f}% do teto")

    if alvos:
        print()
        print("  RESOLUCAO PAGA E VAZIA — candidatos a diversificar no pool de treino:")
        for pv, nome, val, nvals in sorted(alvos, reverse=True):
            print(f"    {nome:<13} tem {nvals} valores, mas '{val}' concentra "
                  f"{pv:.0f}% da experiencia")
        print()
        print("  Cada linha acima e um eixo em que a tupla de estado tem resolucao que o")
        print("  pool de treino nunca exercita. Escolher times que povoem estes eixos")
        print("  preenche capacidade JA PAGA em memoria, sem alterar o espaco de estados.")
        print("  Depois de treinar, correr este script outra vez e comparar.")
    print("=" * L)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Analisa o espaco de estados de um cerebro treinado (leitura pura).")
    ap.add_argument("--agente", default="blue", choices=["blue", "green", "ash"])
    ap.add_argument("--cerebro", default=None, help="caminho explicito para um .pkl")
    ap.add_argument("--visitas-min", type=int, default=20,
                    help="corte de estado 'maduro' (default 20, igual ao dashboard)")
    main(ap.parse_args())
