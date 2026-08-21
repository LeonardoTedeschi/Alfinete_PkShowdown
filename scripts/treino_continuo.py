"""
scripts/treino_continuo.py — ORQUESTRADOR de treino continuo do projeto ALFINETE.

Corre varias execucoes de 10k batalhas EM SEQUENCIA, cada uma como PROCESSO SEPARADO.
Quando uma repeticao termina, o processo fecha e o SO liberta toda a RAM antes da
proxima arrancar. Cada repeticao carrega o cerebro salvo pela anterior e continua de
onde ficou (treino continuo por acumulacao de blocos independentes).

CORRECOES desta versao:
  1. Cada execucao do treino cria o SEU proprio CSV numerado (blue_treino_01.csv,
     blue_treino_02.csv, ...). Antes, todas as execucoes abriam o mesmo CSV em modo
     "w" e SOBRESCREVIAM o log da anterior, por isso ao fim de N repeticoes existia
     log de apenas uma, e parecia que so um treino tinha corrido. O orquestrador
     deteta o CSV novo de cada repeticao para consolidar.
  2. VERIFICA progresso real: le o numero de estados da Q-table antes e depois de cada
     repeticao. Se nao cresceu, avisa (a repeticao nao treinou de facto) em vez de a
     marcar como concluida.
  3. Gera um CSV CONSOLIDADO com todas as repeticoes, com as batalhas acumuladas.

USO (da raiz do projeto):
    python -m scripts.treino_continuo                     (modo interativo)
    python -m scripts.treino_continuo --blue 5
    python -m scripts.treino_continuo --blue 5 --green 5
    python -m scripts.treino_continuo --blue 3 --reset     (apaga o cerebro antes)
"""

import argparse
import csv
import os
import pickle

import numpy as np
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINS_DIR = os.path.join(ROOT, "artefatos", "brains")
LOGS_DIR = os.path.join(ROOT, "artefatos", "logs")

BATALHAS_POR_REPETICAO = 10_000   # tem de bater com MAX_BATALHAS dos scripts

TREINOS = {
    "blue": "scripts.train_blue",
    "green": "scripts.train_green",
}


def caminho_cerebro(agente):
    return os.path.join(BRAINS_DIR, f"{agente}_brain.pkl")


def caminho_csv(agente):
    return os.path.join(LOGS_DIR, f"{agente}_treino.csv")


def contar_estados(agente):
    """Le o numero de estados da Q-table do .pkl. Devolve -1 se nao existir/falhar."""
    pkl = caminho_cerebro(agente)
    if not os.path.exists(pkl) or os.path.getsize(pkl) == 0:
        return -1
    try:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        return len(data.get("q_table", {}))
    except Exception:
        return -1


def auditar_cerebro(agente, limiar_q=200000.0):
    """Auditoria de SAUDE do cerebro, lida diretamente do .pkl.

    O orquestrador e a camada de SUPERVISAO: e aqui que se decide se o plano de
    treino continua. Antes so verificavamos se os estados cresceram, o que mede
    PROGRESSO mas nao SAUDE — e por isso o treino de 200k correu ate ao fim com os
    Q-values a divergirem (os estados cresceram sempre, ate durante a divergencia).

    Devolve (ok, relatorio) onde relatorio traz estados, maior |Q| e o motivo.
    """
    pkl = caminho_cerebro(agente)
    rel = {"estados": -1, "maior_q": float("nan"), "motivo": ""}
    if not os.path.exists(pkl) or os.path.getsize(pkl) == 0:
        rel["motivo"] = f"cerebro inexistente ou vazio: {pkl}"
        return False, rel
    try:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        rel["motivo"] = f"falha a ler o cerebro: {e}"
        return False, rel

    q_table = data.get("q_table", {})
    rel["estados"] = len(q_table)
    if not q_table:
        rel["motivo"] = "Q-table vazia"
        return False, rel

    maior = 0.0
    nao_finito = False
    for v in q_table.values():
        try:
            arr = np.asarray(v, dtype=float)
        except Exception:
            continue
        if not np.all(np.isfinite(arr)):
            nao_finito = True
            break
        m = float(np.max(np.abs(arr)))
        if m > maior:
            maior = m
    rel["maior_q"] = maior

    if nao_finito:
        rel["motivo"] = "Q-values NAO FINITOS (inf/nan): divergencia grave."
        return False, rel
    if maior > limiar_q:
        rel["motivo"] = (f"maior |Q| = {maior:,.0f} acima do limiar {limiar_q:,.0f}: "
                         "divergencia numerica.")
        return False, rel
    rel["motivo"] = "saudavel"
    return True, rel


# --- Criterio de convergencia AO NIVEL DO PLANO -------------------------------
# Avaliado entre repeticoes, sobre TODOS os blocos acumulados, e nao dentro de uma
# corrida de 10k. Julgar convergencia sobre 5 blocos (5000 batalhas) e fragil: o erro
# padrao de um bloco de 1000 batalhas e ~1.58pp, logo com ruido puro a amplitude de 5
# blocos fica abaixo de 2pp em ~10% das janelas, cortando corridas ao acaso.
CONV_BLOCOS = 20        # blocos recentes considerados (= 20k batalhas)
CONV_AMPLITUDE_PP = 2.0 # amplitude maxima do WR nesses blocos


def wr_recentes(agente, n=CONV_BLOCOS):
    """Le os ultimos n valores de WinRate_Bloco de todos os CSVs de sessao."""
    valores = []
    for caminho in listar_csvs(agente):
        try:
            with open(caminho, "r", newline="") as f:
                leitor = csv.reader(f)
                cab = next(leitor, None)
                if not cab or "WinRate_Bloco" not in cab:
                    continue
                i = cab.index("WinRate_Bloco")
                for linha in leitor:
                    if linha and len(linha) > i:
                        try:
                            valores.append(float(linha[i]))
                        except ValueError:
                            pass
        except Exception:
            continue
    return valores[-n:]


def avaliar_convergencia(agente):
    """Devolve (convergiu, mensagem) com base nos blocos acumulados do agente."""
    v = wr_recentes(agente)
    if len(v) < CONV_BLOCOS:
        return False, f"{len(v)}/{CONV_BLOCOS} blocos acumulados (ainda a reunir dados)"
    amplitude = max(v) - min(v)
    media = sum(v) / len(v)
    if amplitude <= CONV_AMPLITUDE_PP:
        return True, (f"WR estavel em {media:.1f}% (amplitude {amplitude:.1f}pp em "
                      f"{CONV_BLOCOS} blocos = {CONV_BLOCOS}k batalhas)")
    return False, (f"WR medio {media:.1f}%, amplitude {amplitude:.1f}pp "
                   f"(> {CONV_AMPLITUDE_PP}pp): ainda nao estabilizou")


def apagar_cerebro(agente):
    pkl = caminho_cerebro(agente)
    if os.path.exists(pkl):
        os.remove(pkl)
        print(f"[ORQUESTRADOR] Cerebro de {agente} apagado (treino comeca do zero).")
    else:
        print(f"[ORQUESTRADOR] Nao havia cerebro de {agente} (ja comeca do zero).")


def listar_csvs(agente):
    """Lista os CSVs de sessao existentes deste agente (blue_treino_01.csv, ...)."""
    import glob
    import re
    encontrados = []
    for caminho in glob.glob(os.path.join(LOGS_DIR, f"{agente}_treino_*.csv")):
        if re.fullmatch(rf"{agente}_treino_(\d+)\.csv", os.path.basename(caminho)):
            encontrados.append(caminho)
    return sorted(encontrados)


def consolidar(agente, arquivos):
    """Junta os CSVs das repeticoes num so, acumulando a coluna Batalhas."""
    arquivos = [a for a in arquivos if a and os.path.exists(a)]
    if not arquivos:
        return None
    destino = os.path.join(LOGS_DIR, f"{agente}_treino_consolidado.csv")
    offset = 0
    cabecalho_escrito = False
    with open(destino, "w", newline="") as saida:
        w = csv.writer(saida)
        for arq in arquivos:
            with open(arq, "r", newline="") as entrada:
                r = csv.reader(entrada)
                cabecalho = next(r, None)
                if cabecalho and not cabecalho_escrito:
                    w.writerow(cabecalho)
                    cabecalho_escrito = True
                ultimo = 0
                for linha in r:
                    if not linha:
                        continue
                    try:
                        batalhas = int(linha[0])
                        linha[0] = batalhas + offset
                        ultimo = batalhas
                    except (ValueError, IndexError):
                        pass
                    w.writerow(linha)
                offset += ultimo
    print(f"[ORQUESTRADOR] CSV consolidado: {destino}")
    return destino


def correr_uma_repeticao(agente, indice, total):
    modulo = TREINOS[agente]
    print("\n" + "#" * 70)
    print(f"#  {agente.upper()} — repeticao {indice}/{total}  (processo separado)")
    print(f"#  comando: {sys.executable} -m {modulo}   (cwd={ROOT})")
    print("#" * 70)
    resultado = subprocess.run([sys.executable, "-m", modulo], cwd=ROOT)
    return resultado.returncode


def executar_plano(plano, reset):
    for agente, repeticoes in plano:
        if repeticoes <= 0:
            continue
        if reset:
            apagar_cerebro(agente)

        print(f"\n[ORQUESTRADOR] === {agente.upper()}: {repeticoes} repeticao(oes) de 10k ===")

        # Auditoria de saude ANTES de comecar: nao faz sentido empilhar treino
        # sobre um cerebro ja divergido.
        if os.path.exists(caminho_cerebro(agente)):
            ok0, rel0 = auditar_cerebro(agente)
            print(f"[ORQUESTRADOR] Estado inicial: {rel0['estados']:,} estados | "
                  f"maior |Q| = {rel0['maior_q']:,.0f} | {rel0['motivo']}")
            if not ok0:
                print(f"[ORQUESTRADOR] RECUSADO: o cerebro de {agente} ja esta doente.")
                print("               Apaga-o (--reset) ou corrige os hiperparametros.")
                continue
        arquivos = []
        for i in range(1, repeticoes + 1):
            estados_antes = contar_estados(agente)
            csvs_antes = listar_csvs(agente)
            t0 = time.time()
            codigo = correr_uma_repeticao(agente, i, repeticoes)
            dt = time.time() - t0
            estados_depois = contar_estados(agente)

            if codigo != 0:
                print(f"[ORQUESTRADOR] ERRO: repeticao {i} de {agente} saiu com codigo "
                      f"{codigo}. A parar este agente (nao empilha repeticoes sobre um erro).")
                break

            # --- SUPERVISAO: duas verificacoes independentes ---
            # (1) PROGRESSO: a Q-table cresceu? (deteta uma repeticao que nao treinou)
            if estados_depois < 0:
                print(f"[ORQUESTRADOR] ATENCAO: nao foi possivel ler a Q-table em "
                      f"{caminho_cerebro(agente)} depois da repeticao {i}.")
                print("               O treino pode nao estar a persistir o cerebro. A parar.")
                break
            if estados_antes >= 0 and estados_depois <= estados_antes:
                # AVISO, nao paragem. Numa fase avancada do treino a descoberta de
                # estados satura naturalmente, e abortar aqui daria a um agente menos
                # batalhas que a outro, invalidando a comparacao de orcamento fixo.
                # A paragem fica reservada a divergencia (problema real de saude).
                print(f"[ORQUESTRADOR] AVISO: a Q-table nao cresceu na repeticao {i} "
                      f"({estados_antes:,} -> {estados_depois:,}). Pode ser saturacao "
                      f"normal da descoberta. A continuar (orcamento fixo).")

            # (2) SAUDE: os valores estao numa escala sa? Esta verificacao FALTAVA:
            # os estados crescem mesmo durante a divergencia, por isso o teste de
            # progresso sozinho deixou o treino de 200k correr ate ao fim com os
            # Q-values a explodir.
            ok_saude, rel = auditar_cerebro(agente)
            if not ok_saude:
                print()
                print("!" * 70)
                print(f"  [ORQUESTRADOR] PLANO ABORTADO na repeticao {i}/{repeticoes}")
                print(f"  Agente: {agente} | {rel['motivo']}")
                print(f"  Estados: {rel['estados']:,} | maior |Q|: {rel['maior_q']:,.0f}")
                print("  Nao serao lancadas mais repeticoes deste agente.")
                print("!" * 70)
                break

            # Cada execucao cria o SEU CSV numerado; detetamos qual e o novo.
            csvs_depois = listar_csvs(agente)
            novos = [c for c in csvs_depois if c not in csvs_antes]
            # Convergencia avaliada AQUI, sobre os blocos acumulados de todas as
            # sessoes, e nao dentro de cada corrida de 10k.
            convergiu, msg_conv = avaliar_convergencia(agente)

            arq = novos[0] if novos else None
            if arq:
                arquivos.append(arq)
            crescimento = estados_depois - max(0, estados_antes)
            print(f"[ORQUESTRADOR] {agente} repeticao {i}/{repeticoes} OK em {dt:.0f}s | "
                  f"estados: {max(0, estados_antes):,} -> {estados_depois:,} (+{crescimento:,}) | "
                  f"maior |Q|: {rel['maior_q']:,.0f}")
            if arq:
                print(f"               log desta sessao: {os.path.basename(arq)}")
            print(f"               convergencia: {msg_conv}")
            if convergiu:
                print(f"[ORQUESTRADOR] {agente.upper()} CONVERGIU. A parar o plano "
                      f"(restavam {repeticoes - i} repeticoes).")
                break
            else:
                print("               AVISO: nao foi criado um CSV novo nesta repeticao.")

        if arquivos:
            consolidar(agente, arquivos)

    print("\n[ORQUESTRADOR] Plano de treino concluido.")


def modo_interativo():
    print("=" * 60)
    print("  TREINO CONTINUO — ALFINETE")
    print("=" * 60)
    print("Agentes: blue (hibrido), green (Q-puro). Cada repeticao = 10k batalhas.")
    plano = []
    for agente in ["blue", "green"]:
        while True:
            resp = input(f"Quantas repeticoes de 10k para o {agente.upper()}? (0 = nenhuma): ").strip()
            if resp == "":
                resp = "0"
            if resp.isdigit():
                plano.append((agente, int(resp)))
                break
            print("  Escreve um numero inteiro.")

    reset = input("Apagar cerebros antes (comecar do zero)? [s/N]: ").strip().lower() in ("s", "sim", "y", "yes")

    if sum(r for _, r in plano) == 0:
        print("Nada a treinar. A sair.")
        return
    print("\nPlano: " + ", ".join(f"{a}={r}" for a, r in plano if r > 0) +
          f" | reset={'sim' if reset else 'nao'}")
    if input("Confirmar e iniciar? [S/n]: ").strip().lower() in ("", "s", "sim", "y", "yes"):
        executar_plano(plano, reset)
    else:
        print("Cancelado.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Orquestrador de treino continuo.")
    ap.add_argument("--blue", type=int, default=None, help="repeticoes de 10k do Blue")
    ap.add_argument("--green", type=int, default=None, help="repeticoes de 10k do Green")
    ap.add_argument("--batalhas", type=int, default=None,
                    help="ORCAMENTO FIXO em batalhas por agente (ex.: 200000). "
                         "Converte-se em repeticoes de 10k e aplica-se a todos os "
                         "agentes indicados, garantindo comparacao controlada.")
    ap.add_argument("--reset", action="store_true", help="apaga os cerebros antes de treinar")
    args = ap.parse_args()

    if args.batalhas is not None:
        # ORCAMENTO FIXO: mesmo numero de batalhas para todos os agentes indicados.
        # E o desenho correto para comparar algoritmos: sem isto, um agente que
        # parasse mais cedo teria menos experiencia e a diferenca de desempenho
        # confundir-se-ia com a diferenca de orcamento.
        reps = max(1, round(args.batalhas / BATALHAS_POR_REPETICAO))
        efetivo = reps * BATALHAS_POR_REPETICAO
        agentes = [a for a in ("blue", "green")
                   if getattr(args, a) is not None or (args.blue is None and args.green is None)]
        if not agentes:
            agentes = ["blue", "green"]
        print(f"[ORQUESTRADOR] ORCAMENTO FIXO: {efetivo:,} batalhas por agente "
              f"({reps} repeticoes de {BATALHAS_POR_REPETICAO:,})")
        if efetivo != args.batalhas:
            print(f"               (ajustado de {args.batalhas:,} para o multiplo mais "
                  f"proximo de {BATALHAS_POR_REPETICAO:,})")
        print(f"[ORQUESTRADOR] Agentes: {', '.join(agentes)}")
        executar_plano([(a, reps) for a in agentes], args.reset)
    elif args.blue is None and args.green is None:
        modo_interativo()
    else:
        executar_plano([("blue", args.blue or 0), ("green", args.green or 0)], args.reset)
