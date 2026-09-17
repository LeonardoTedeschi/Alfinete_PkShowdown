"""
shared/analysis/plot_generalizacao.py — historico e grafico dos testes de holdout.

PORQUE EXISTE
-------------
O `scripts/avaliar_generalizacao.py` imprime o resultado na consola e perde-o. Cada
teste custa 10.000 batalhas (5.000 por condicao) e a serie entre versoes e o dado que
mostra se as alteracoes ao pool e ao instinto estao a reduzir o overfitting.

Mesmo padrao do `regua_historico.csv`: um CSV acumulativo com uma linha por teste, e
um grafico gerado a partir dele.

O QUE O GRAFICO MOSTRA
----------------------
Barras emparelhadas por agente (treino ao lado de holdout) com barras de erro de IC
95%, e a QUEDA anotada por cima de cada par. A linha dos 50% marca o acaso.

NOTA SOBRE A LEITURA DA QUEDA
-----------------------------
Na condicao de holdout o ADVERSARIO (InstinctBot) tambem joga com os times novos, logo
qualquer diferenca de dificuldade entre os pools contamina a queda medida. Mede-se com
o proprio InstinctBot: e a diferenca entre as duas ancoras (`medir_regua --pool eval`
menos `--pool treino`).

Historico dessa diferenca:

    instinto v12   +3,06 pp    o pool de eval favorecia claramente o instinto
    instinto v13   +1,50 pp
    instinto v14   -0,65 pp    1,0 sigma: indistinguivel de ZERO
    instinto v15   -0,35 pp    ancoras 74,37 treino / 74,02 eval
    instinto v19   +2,42 pp    ancoras 77,38 / 79,80, 4,2 sigma
    instinto v20   +1,70 pp    ancoras 78,82 / 80,52, 3,0 sigma
    instinto v21   +1,72 pp    ancoras 78,45 / 80,17, 3,0 sigma
    instinto v22   +2,35 pp    ancoras 78,91 / 81,26, 4,2 sigma  <-- EM USO (congelado)

O paragrafo que aqui afirmava que os dois pools sao igualmente dificeis DEIXOU DE SER
VERDADE a partir do v19. Com `VIES_POOL_PP = 0.0` a coluna corrigida sobrestimava a
falha de generalizacao, em silencio e com o subtitulo do grafico a afirmar "pools
equivalentes".

AS QUATRO MEDICOES (v19 a v22) ANDAM ENTRE 1,70 e 2,42 pp E SAO COMPATIVEIS ENTRE SI: a
maior diferenca entre duas delas fica dentro de ~1 sigma. **Nao se deve ler a oscilacao
como efeito das correccoes.** O que se sabe e que o vies EXISTE, ronda os 2 pp, e o pool
de eval (20 times contra 60) e intrinsecamente mais ruidoso.

ATUALIZAR sempre que o instinto ou os pools mudarem: correr as duas ancoras e por aqui
a diferenca.

COMPATIBILIDADE COM CORRIDAS ANTIGAS (11/09/2026)
-------------------------------------------------
O CSV acumula corridas feitas com VALORES DIFERENTES desta constante. Mudar a constante
NAO deve reescrever o passado: cada linha foi corrigida com o vies vigente quando o
instinto tinha aquela versao, e esse desconto era o CERTO nessa altura.

Por isso:

  1. Gravou-se a coluna `Vies_Pool_pp` a partir de 11/09/2026, com o valor usado nessa
     linha.
  2. Para as linhas ANTIGAS, que nao tem a coluna, o vies e RECUPERADO por subtraccao:
     `Queda_pp - Queda_Corrigida_pp`. Nao se adivinha nada — as duas colunas sempre
     estiveram la, e a diferenca entre elas E o vies aplicado.
  3. O ficheiro e MIGRADO uma vez, na primeira gravacao apos esta versao: reescreve-se
     com o cabecalho novo e preenche-se `Vies_Pool_pp` das linhas antigas pela regra 2.
     Um `.bak` e deixado ao lado antes de tocar no original.
  4. O subtitulo do grafico deixa de anunciar UM vies quando as barras desenhadas usam
     valores diferentes: nesse caso diz o intervalo. Anunciar 2,35 num grafico que
     contem barras descontadas a 2,42 seria mentir.

USO
---
Do `avaliar_generalizacao.py`, depois de `imprimir(...)`:

    from shared.analysis.plot_generalizacao import registar, gerar_grafico
    registar(agente, treino, holdout, etiqueta=args.etiqueta)
    gerar_grafico()

Ou avulso, para redesenhar o grafico a partir do CSV existente:

    python -m shared.analysis.plot_generalizacao
"""

import csv
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(ROOT, "artefatos", "logs")
CSV_HISTORICO = os.path.join(LOGS_DIR, "generalizacao_historico.csv")
PNG = os.path.join(LOGS_DIR, "generalizacao_historico.png")

# Vantagem do pool de eval sobre o de treino, medida com o InstinctBot.
#
# ATUALIZADO 06/09/2026, INSTINTO v22 (CONGELADO para o ciclo v10/v7, ver 6.47 J).
# +2,35 pp com erro padrao da diferenca de 0,57 pp, ou seja 4,2 sigma.
#
# LEITURA: na condicao de holdout o ADVERSARIO tambem joga com os times novos, logo
# 2,35 pp da queda medida sao do MATERIAL e nao do agente.
#
# ESTE E O VALOR DO CICLO. A regua esta congelada e este numero NAO deve mudar ate ao
# fim do v10/v7. Se a regua for descongelada, remedir as duas ancoras antes de lhe tocar.
# As linhas JA GRAVADAS mantem o vies com que foram calculadas (ver COMPATIBILIDADE).
VIES_POOL_PP = 2.35

# Nome da coluna acrescentada em 11/09/2026. Isolado numa constante porque e usado na
# escrita, na migracao e na leitura.
COL_VIES = "Vies_Pool_pp"

CABECALHO = [
    "Data", "Etiqueta", "Agente", "Batalhas_Por_Condicao",
    "WR_Treino", "WR_Holdout", "Queda_pp", "Queda_Corrigida_pp", "EP_Diferenca_pp",
    "Margem_Treino", "Margem_Holdout", "Duracao_Treino", "Duracao_Holdout",
    "Ties_Treino", "Ties_Holdout", "Estados_Novos_Treino", "Estados_Novos_Holdout",
    COL_VIES,
]

# Cabecalho ANTERIOR a 11/09/2026, sem `Vies_Pool_pp`. Guardado para reconhecer um
# ficheiro por migrar sem depender da contagem de colunas.
CABECALHO_LEGADO = [c for c in CABECALHO if c != COL_VIES]


def _ep_diferenca(p1, p2, n):
    return (((p1 * (1 - p1)) + (p2 * (1 - p2))) / max(1, n)) ** 0.5 * 100.0


def vies_da_linha(linha):
    """Vies aplicado NAQUELA linha, seja ela nova ou antiga.

    Linha nova: le a coluna `Vies_Pool_pp`.
    Linha antiga: RECUPERA por subtraccao, `Queda_pp - Queda_Corrigida_pp`. As duas
    colunas sempre existiram, logo isto nao adivinha — reconstroi exactamente o
    desconto que foi aplicado.

    Devolve `None` so quando nem uma coisa nem outra e legivel.
    """
    try:
        v = linha.get(COL_VIES)
        if v not in (None, ""):
            return float(v)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        return float(linha["Queda_pp"]) - float(linha["Queda_Corrigida_pp"])
    except (KeyError, TypeError, ValueError):
        return None


def _migrar_csv_se_preciso():
    """Acrescenta a coluna `Vies_Pool_pp` a um ficheiro gravado antes de 11/09/2026.

    PORQUE ISTO EXISTE. Fazer `append` de uma linha com uma coluna a mais num CSV cujo
    cabecalho nao a tem DESALINHA o ficheiro em silencio: o `DictReader` passa a ler o
    vies na coluna errada e nada avisa. Migra-se uma vez, antes da primeira escrita.

    Nao ha perda: as linhas antigas recebem o vies RECUPERADO por `vies_da_linha`, que
    e o valor exacto com que foram calculadas. Um `.bak` fica ao lado antes de tocar no
    original.
    """
    if not os.path.exists(CSV_HISTORICO):
        return
    with open(CSV_HISTORICO, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        campos = list(leitor.fieldnames or [])
        if COL_VIES in campos:
            return                      # ja migrado
        linhas = list(leitor)
    if not campos:
        return
    import shutil
    shutil.copy2(CSV_HISTORICO, CSV_HISTORICO + ".bak")
    for l in linhas:
        v = vies_da_linha(l)
        l[COL_VIES] = f"{v:.2f}" if v is not None else ""
    with open(CSV_HISTORICO, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos + [COL_VIES])
        w.writeheader()
        w.writerows(linhas)
    print(f"[plot_generalizacao] CSV migrado: coluna '{COL_VIES}' acrescentada a "
          f"{len(linhas)} linha(s). Copia em {os.path.basename(CSV_HISTORICO)}.bak")


def registar(agente, treino, holdout, etiqueta=""):
    """Acrescenta uma linha ao historico. `treino` e `holdout` sao os dicionarios
    devolvidos por `avaliar_condicao`.

    GUARDA CONTRA LINHAS VAZIAS (29/08/2026). O `regua_historico.csv` apareceu com o
    dobro das linhas, metade delas sem Win Rate — uma escrita no ARRANQUE e outra no
    FIM de cada corrida. O grafico desenhava duas barras com a mesma etiqueta.
    Aqui so se grava com resultado: sem batalhas terminadas, nao ha o que registar.
    """
    if not treino or not holdout:
        return None
    if min(treino.get("batalhas", 0), holdout.get("batalhas", 0)) <= 0:
        return None
    os.makedirs(LOGS_DIR, exist_ok=True)
    _migrar_csv_se_preciso()
    n = min(treino["batalhas"], holdout["batalhas"])
    queda = treino["wr"] - holdout["wr"]
    ep = _ep_diferenca(treino["wr"] / 100.0, holdout["wr"] / 100.0, n)

    novo = not os.path.exists(CSV_HISTORICO)
    with open(CSV_HISTORICO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(CABECALHO)
        w.writerow([
            datetime.now().strftime("%Y%m%d_%H%M%S"), etiqueta, agente.upper(), n,
            f"{treino['wr']:.2f}", f"{holdout['wr']:.2f}",
            f"{queda:+.2f}", f"{queda - VIES_POOL_PP:+.2f}", f"{ep:.2f}",
            f"{treino['margem']:.2f}", f"{holdout['margem']:.2f}",
            f"{treino['duracao']:.0f}", f"{holdout['duracao']:.0f}",
            treino["ties"], holdout["ties"],
            treino["estados_depois"] - treino["estados_antes"],
            holdout["estados_depois"] - holdout["estados_antes"],
            # Grava-se o vies USADO NESTA LINHA. Mudar a constante amanha nao
            # reescreve o passado: cada corrida guarda o desconto que lhe foi
            # aplicado, e o grafico le-o de volta linha a linha.
            f"{VIES_POOL_PP:.2f}",
        ])
    return CSV_HISTORICO


def ler_historico():
    if not os.path.exists(CSV_HISTORICO):
        return []
    with open(CSV_HISTORICO, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gerar_grafico(destino=None, ultimos=8):
    """Barras emparelhadas treino/holdout, com IC 95% e a queda anotada.

    `ultimos` limita quantos testes aparecem, para o grafico nao ficar ilegivel
    quando o historico crescer.
    """
    linhas = ler_historico()
    # Ignorar linhas sem Win Rate (ver a guarda em `registar`): ficheiros antigos
    # podem te-las, e desenha-las produzia barras a zero com a mesma etiqueta.
    validas = []
    for l in linhas:
        try:
            float(l["WR_Treino"]); float(l["WR_Holdout"])
            validas.append(l)
        except (KeyError, TypeError, ValueError):
            continue
    if not validas:
        return None
    linhas = validas[-ultimos:]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ROTULOS (11/09/2026): as corridas de ablacao foram gravadas SEM etiqueta, e tres
    # barras "BLUE 20260911" sao indistinguiveis. Sem etiqueta, cai-se na data COM
    # HORA, que e unica por corrida. Preferir sempre `--etiqueta` ao correr.
    def _rotulo(l):
        et = (l.get("Etiqueta") or "").strip()
        if et:
            return f"{l['Agente']}\n{et}"
        d = l.get("Data", "")
        return f"{l['Agente']}\n{d[:8]}\n{d[9:13]}" if len(d) >= 13 else f"{l['Agente']}\n{d[:8]}"

    rotulos = [_rotulo(l) for l in linhas]
    wt = [float(l["WR_Treino"]) for l in linhas]
    wh = [float(l["WR_Holdout"]) for l in linhas]
    n = [int(l["Batalhas_Por_Condicao"]) for l in linhas]

    def ic(p, k):
        return 1.96 * ((p / 100 * (1 - p / 100) / max(1, k)) ** 0.5) * 100

    et = [ic(p, k) for p, k in zip(wt, n)]
    eh = [ic(p, k) for p, k in zip(wh, n)]

    x = np.arange(len(linhas))
    larg = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(linhas) * 1.9), 6))
    ax.bar(x - larg / 2, wt, larg, yerr=et, capsize=4, label="Times de treino",
           color="#4C72B0")
    ax.bar(x + larg / 2, wh, larg, yerr=eh, capsize=4, label="Holdout (times novos)",
           color="#DD8452")
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="Acaso (50%)")

    for i, l in enumerate(linhas):
        q = float(l["Queda_pp"])
        qc = float(l["Queda_Corrigida_pp"])
        topo = max(wt[i] + et[i], wh[i] + eh[i])
        ax.text(i, topo + 2.5, f"queda {q:+.1f} pp", ha="center",
                fontweight="bold", fontsize=9)
        # So mostrar a corrigida quando ela DIFERE da bruta: com pools equivalentes,
        # repetir o mesmo numero duas vezes so confunde quem le.
        if abs(q - qc) >= 0.05:
            ax.text(i, topo + 0.6, f"(corrigida {qc:+.1f})", ha="center",
                    fontsize=8, color="#555555")

    ax.set_xticks(list(x))
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    # ==================================================================
    # O SUBTITULO LE O VIES DAS BARRAS DESENHADAS (11/09/2026)
    # ==================================================================
    # Antes anunciava `VIES_POOL_PP`, a constante ACTUAL. Como o CSV acumula corridas
    # feitas com valores diferentes, o grafico podia dizer "desconta 2,35" por cima de
    # barras descontadas a 2,42. Agora le linha a linha e, se divergirem, mostra o
    # intervalo em vez de escolher um.
    vieses = [v for v in (vies_da_linha(l) for l in linhas) if v is not None]
    if not vieses:
        sub = "barras de erro: IC 95%"
    elif max(vieses) - min(vieses) < 0.005:
        v = vieses[0]
        if abs(v) >= 0.005:
            sub = (f"barras de erro: IC 95%  |  queda corrigida desconta o vies de "
                   f"{v:+.2f} pp do pool de eval")
        else:
            sub = ("barras de erro: IC 95%  |  pools equivalentes: a queda e toda "
                   "atribuivel ao agente")
    else:
        sub = (f"barras de erro: IC 95%  |  vies do pool descontado por corrida: "
               f"{min(vieses):+.2f} a {max(vieses):+.2f} pp")
    ax.set_title("Generalizacao: desempenho em times conhecidos vs times novos\n"
                 + sub)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    destino = destino or PNG
    os.makedirs(os.path.dirname(os.path.abspath(destino)), exist_ok=True)
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return destino


if __name__ == "__main__":
    d = gerar_grafico()
    print(f"Historico: {CSV_HISTORICO}")
    print(f"Grafico  : {d or '(sem dados)'}")
