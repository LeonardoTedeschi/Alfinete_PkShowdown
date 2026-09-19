"""
scripts/treino_continuo.py — ORQUESTRADOR de treino continuo do projeto ALFINETE.

Blue/Green usam o pipeline novo:
  - cada processo treina 10k e deixa um CSV de emergencia;
  - o orquestrador valida e incorpora atomicamente esse CSV ao consolidado;
  - depois da incorporacao, o CSV de emergencia e removido;
  - o processo fecha entre sessoes, libertando a RAM;
  - a cada 50k acumulados gera um grafico CUMULATIVO e fotografias do brain/N(s,a).

Os scripts individuais cuidam da persistencia do brain:
  - emergencia em 5k;
  - oficial em 10k.

O Ash permanece no pipeline legado porque o respetivo train_ash.py nao foi fornecido
nesta revisao.

USO (da raiz do projeto):
    python -m scripts.treino_continuo
    python -m scripts.treino_continuo --blue 5
    python -m scripts.treino_continuo --blue 5 --green 5
    python -m scripts.treino_continuo --blue 1 --green 1 --batalhas 650000 --reset
"""

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.analysis.plot_graph import generate_graph
from shared.analysis.inspect_brain import analyze_brain
from shared.analysis.inspect_action_choices import analyze_action_choices

BRAINS_DIR = os.path.join(ROOT, "artefatos", "brains")
LOGS_DIR = os.path.join(ROOT, "artefatos", "logs")
ANALISE_DIR = os.path.join(LOGS_DIR, "analise")

BLOCO_BATALHAS = 1_000
BATALHAS_POR_REPETICAO = 10_000
ANALISE_CADA_BATALHAS = 50_000
PIPELINE_NOVO = {"blue", "green"}

TREINOS = {
    "blue": "scripts.train_blue",
    "green": "scripts.train_green",
    "ash": "scripts.train_ash",
}


def caminho_cerebro(agente):
    return os.path.join(BRAINS_DIR, f"{agente}_brain.pkl")


def caminho_cerebro_emergencia(agente):
    return os.path.join(BRAINS_DIR, f"{agente}_brain_emergency.pkl")


def caminho_csv_emergencia(agente):
    return os.path.join(LOGS_DIR, f"{agente}_treino_emergencia.csv")


def caminho_csv_consolidado(agente):
    return os.path.join(LOGS_DIR, f"{agente}_treino_consolidado.csv")


def contar_estados(agente):
    """Le o numero de estados da Q-table oficial. -1 se nao existir/falhar."""
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
    """Auditoria de saude do brain oficial persistido."""
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
    for v in q_table.values():
        try:
            arr = np.asarray(v, dtype=float)
        except Exception:
            continue
        if not np.all(np.isfinite(arr)):
            rel["motivo"] = "Q-values NAO FINITOS (inf/nan): divergencia grave."
            return False, rel
        m = float(np.max(np.abs(arr)))
        if m > maior:
            maior = m
    rel["maior_q"] = maior

    if maior > limiar_q:
        rel["motivo"] = (f"maior |Q| = {maior:,.0f} acima do limiar {limiar_q:,.0f}: "
                         "divergencia numerica.")
        return False, rel
    rel["motivo"] = "saudavel"
    return True, rel


CONV_BLOCOS = 20
CONV_AMPLITUDE_PP = 2.0


def _ler_csv(caminho):
    if not os.path.exists(caminho):
        return None, []
    with open(caminho, "r", newline="") as f:
        r = csv.reader(f)
        cab = next(r, None)
        linhas = [row for row in r if row]
    return cab, linhas


def wr_recentes(agente, n=CONV_BLOCOS):
    """Le os ultimos WR do consolidado; Ash usa os CSVs legados."""
    valores = []
    if agente in PIPELINE_NOVO:
        caminhos = [caminho_csv_consolidado(agente)]
    else:
        caminhos = listar_csvs_legado(agente)

    for caminho in caminhos:
        try:
            cab, linhas = _ler_csv(caminho)
            if not cab or "WinRate_Bloco" not in cab:
                continue
            idx = cab.index("WinRate_Bloco")
            for linha in linhas:
                if len(linha) > idx:
                    try:
                        valores.append(float(linha[idx]))
                    except ValueError:
                        pass
        except Exception:
            continue
    return valores[-n:]


def avaliar_convergencia(agente):
    """Indicador apenas; nunca interrompe o orcamento fixo."""
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


def limpar_estado_ativo(agente):
    """Reset do ciclo ativo sem tocar nas pastas historicas B_*/G_* do usuario."""
    alvos = [caminho_cerebro(agente)]
    if agente in PIPELINE_NOVO:
        alvos += [
            caminho_cerebro_emergencia(agente),
            caminho_cerebro_emergencia(agente) + ".tmp",
            caminho_csv_emergencia(agente),
            caminho_csv_consolidado(agente),
            caminho_csv_consolidado(agente) + ".tmp",
        ]
    removidos = []
    for p in alvos:
        if os.path.exists(p):
            os.remove(p)
            removidos.append(p)
    if removidos:
        print(f"[ORQUESTRADOR] Reset de {agente}: {len(removidos)} ficheiro(s) ativo(s) removido(s).")
    else:
        print(f"[ORQUESTRADOR] Reset de {agente}: nao havia estado ativo.")


def _validar_emergencia(agente):
    caminho = caminho_csv_emergencia(agente)
    cab, linhas = _ler_csv(caminho)
    if not cab:
        raise RuntimeError(f"CSV de emergencia ausente/vazio: {caminho}")
    if len(linhas) != BATALHAS_POR_REPETICAO // BLOCO_BATALHAS:
        raise RuntimeError(
            f"CSV de emergencia incompleto: esperado 10 blocos, recebido {len(linhas)}. "
            "O ficheiro foi preservado."
        )
    esperadas = list(range(BLOCO_BATALHAS, BATALHAS_POR_REPETICAO + 1, BLOCO_BATALHAS))
    observadas = []
    for linha in linhas:
        try:
            observadas.append(int(float(linha[0])))
        except (ValueError, IndexError) as e:
            raise RuntimeError(f"Coluna Batalhas invalida no CSV de emergencia: {e}")
    if observadas != esperadas:
        raise RuntimeError(f"Sequencia Batalhas invalida: {observadas}; esperado {esperadas}")
    return cab, linhas


def _linhas_com_offset(linhas, offset):
    saida = []
    for linha in linhas:
        nova = list(linha)
        nova[0] = str(int(float(nova[0])) + offset)
        saida.append(nova)
    return saida


def consolidar_emergencia(agente):
    """Incorpora uma sessao completa de 10k ao consolidado de forma atomica.

    O consolidado e a fonte primaria. O CSV de emergencia so e apagado depois do
    os.replace() bem sucedido. Se o processo cair depois do replace e antes do remove,
    a deteccao de cauda evita duplicar a mesma sessao na proxima execucao.
    """
    cab_novo, linhas_novas = _validar_emergencia(agente)
    destino = caminho_csv_consolidado(agente)
    cab_antigo, linhas_antigas = _ler_csv(destino)

    if cab_antigo and cab_antigo != cab_novo:
        raise RuntimeError("Cabecalho do CSV consolidado difere do CSV de emergencia.")

    # Idempotencia: se a sessao ja for exatamente a cauda do consolidado, apenas
    # remove o emergency remanescente de um crash entre replace e remove.
    if linhas_antigas and len(linhas_antigas) >= len(linhas_novas):
        try:
            offset_anterior = int(float(linhas_antigas[-1][0])) - int(float(linhas_novas[-1][0]))
            candidato = _linhas_com_offset(linhas_novas, offset_anterior)
            if linhas_antigas[-len(candidato):] == candidato:
                os.remove(caminho_csv_emergencia(agente))
                total = int(float(linhas_antigas[-1][0]))
                print(f"[ORQUESTRADOR] Sessao ja estava consolidada; emergency removido ({total:,}).")
                return destino, total
        except Exception:
            pass

    offset = int(float(linhas_antigas[-1][0])) if linhas_antigas else 0
    ajustadas = _linhas_com_offset(linhas_novas, offset)
    todas = linhas_antigas + ajustadas

    temp = destino + ".tmp"
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(temp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab_antigo or cab_novo)
        w.writerows(todas)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, destino)

    # Verificacao simples antes de apagar a copia de emergencia.
    _, verificacao = _ler_csv(destino)
    if len(verificacao) != len(todas):
        raise RuntimeError("Consolidado nao passou verificacao de quantidade de linhas; emergency preservado.")

    os.remove(caminho_csv_emergencia(agente))
    total = int(float(todas[-1][0]))
    print(f"[ORQUESTRADOR] CSV consolidado atualizado: {destino} ({total:,} batalhas)")
    return destino, total


def _ultima_metrica(caminho):
    cab, linhas = _ler_csv(caminho)
    if not cab or not linhas:
        return None
    return dict(zip(cab, linhas[-1]))


def gerar_artefatos_marco(agente, consolidado, total_batalhas, estados):
    """Gera somente nos marcos de 50k; cada PNG e cumulativo e fica preservado."""
    if total_batalhas <= 0 or total_batalhas % ANALISE_CADA_BATALHAS != 0:
        return

    os.makedirs(ANALISE_DIR, exist_ok=True)
    k = total_batalhas // 1000
    nome = agente.capitalize()
    ultima = _ultima_metrica(consolidado) or {}
    try:
        wr = float(ultima.get("WinRate_Bloco", 0.0))
    except (TypeError, ValueError):
        wr = 0.0

    graf = os.path.join(LOGS_DIR, f"{agente}_treino_{k:03d}k.png")
    try:
        generate_graph(
            consolidado, graf,
            agent=nome, opponent="Instinto",
            total_battles=total_batalhas,
            final_win_rate=wr,
            final_states=estados,
        )
        print(f"[ORQUESTRADOR] Grafico cumulativo {k}k: {graf}")
    except Exception as e:
        print(f"[ORQUESTRADOR] AVISO: grafico {k}k falhou: {e}")

    brain = caminho_cerebro(agente)
    etiqueta = f"{nome}_{k:03d}k"
    try:
        analyze_brain(brain, ANALISE_DIR, etiqueta)
        analyze_action_choices(brain, ANALISE_DIR, etiqueta)
        print(f"[ORQUESTRADOR] Fotografias do brain {k}k: {ANALISE_DIR}")
    except Exception as e:
        print(f"[ORQUESTRADOR] AVISO: analise do brain {k}k falhou: {e}")


# ---------------------------------------------------------------------------
# Compatibilidade legado para Ash. Blue/Green nao usam estes CSVs numerados.
# ---------------------------------------------------------------------------
def listar_csvs_legado(agente):
    import glob
    import re
    encontrados = []
    for caminho in glob.glob(os.path.join(LOGS_DIR, f"{agente}_treino_*.csv")):
        if re.fullmatch(rf"{agente}_treino_(\d+)\.csv", os.path.basename(caminho)):
            encontrados.append(caminho)
    return sorted(encontrados)


def consolidar_legado(agente, arquivos):
    arquivos = [a for a in arquivos if a and os.path.exists(a)]
    if not arquivos:
        return None
    destino = caminho_csv_consolidado(agente)
    offset = 0
    cabecalho_escrito = False
    with open(destino, "w", newline="") as saida:
        w = csv.writer(saida)
        for arq in arquivos:
            with open(arq, "r", newline="") as entrada:
                r = csv.reader(entrada)
                cab = next(r, None)
                if cab and not cabecalho_escrito:
                    w.writerow(cab)
                    cabecalho_escrito = True
                ultimo = 0
                for linha in r:
                    if not linha:
                        continue
                    try:
                        batalhas = int(float(linha[0]))
                        if batalhas < BLOCO_BATALHAS:
                            batalhas *= BLOCO_BATALHAS
                        linha[0] = batalhas + offset
                        ultimo = batalhas
                    except (ValueError, IndexError):
                        pass
                    w.writerow(linha)
                offset += ultimo
    print(f"[ORQUESTRADOR] CSV legado consolidado: {destino}")
    return destino


def correr_uma_repeticao(agente, indice, total):
    modulo = TREINOS[agente]
    print("\n" + "#" * 70)
    print(f"#  {agente.upper()} — repeticao {indice}/{total}  (processo separado)")
    print(f"#  comando: {sys.executable} -m {modulo}   (cwd={ROOT})")
    print("#" * 70)
    resultado = subprocess.run([sys.executable, "-m", modulo], cwd=ROOT)
    return resultado.returncode


def _executar_pipeline_novo(agente, repeticoes, reset):
    if reset:
        limpar_estado_ativo(agente)

    print(f"\n[ORQUESTRADOR] === {agente.upper()}: {repeticoes} repeticao(oes) de 10k ===")

    # Nao sobrescreve recuperacao pendente.
    pendentes = [p for p in (caminho_csv_emergencia(agente), caminho_cerebro_emergencia(agente))
                 if os.path.exists(p)]
    if pendentes:
        print("[ORQUESTRADOR] RECUSADO: ha recuperacao pendente:")
        for p in pendentes:
            print(f"               {p}")
        print("               Preserve/recupere antes de continuar.")
        return

    if os.path.exists(caminho_cerebro(agente)):
        ok0, rel0 = auditar_cerebro(agente)
        print(f"[ORQUESTRADOR] Estado inicial: {rel0['estados']:,} estados | "
              f"maior |Q| = {rel0['maior_q']:,.0f} | {rel0['motivo']}")
        if not ok0:
            print(f"[ORQUESTRADOR] RECUSADO: o cerebro de {agente} ja esta doente.")
            return

    for i in range(1, repeticoes + 1):
        estados_antes = contar_estados(agente)
        t0 = time.time()
        codigo = correr_uma_repeticao(agente, i, repeticoes)
        dt = time.time() - t0

        if codigo != 0:
            print(f"[ORQUESTRADOR] ERRO: repeticao {i} saiu com codigo {codigo}.")
            print("               Emergency CSV/brain foram preservados para diagnostico.")
            break

        estados_depois = contar_estados(agente)
        if estados_depois < 0:
            print("[ORQUESTRADOR] ERRO: brain oficial nao foi persistido no fim dos 10k.")
            break
        if estados_antes >= 0 and estados_depois <= estados_antes:
            print(f"[ORQUESTRADOR] AVISO: Q-table nao cresceu ({estados_antes:,} -> "
                  f"{estados_depois:,}); pode ser saturacao normal.")

        ok_saude, rel = auditar_cerebro(agente)
        if not ok_saude:
            print("!" * 70)
            print(f"[ORQUESTRADOR] PLANO ABORTADO: {rel['motivo']}")
            print("!" * 70)
            break

        try:
            consolidado, total_cumulativo = consolidar_emergencia(agente)
        except Exception as e:
            print(f"[ORQUESTRADOR] ERRO AO CONSOLIDAR: {e}")
            print("               CSV de emergencia preservado. A parar para nao perder dados.")
            break

        estavel, msg_conv = avaliar_convergencia(agente)
        crescimento = estados_depois - max(0, estados_antes)
        print(f"[ORQUESTRADOR] {agente} repeticao {i}/{repeticoes} OK em {dt:.0f}s | "
              f"total {total_cumulativo:,} | estados {max(0, estados_antes):,} -> "
              f"{estados_depois:,} (+{crescimento:,}) | maior |Q| {rel['maior_q']:,.0f}")
        print(f"               estabilidade: {msg_conv}{'  [ESTAVEL]' if estavel else ''}")

        gerar_artefatos_marco(agente, consolidado, total_cumulativo, estados_depois)


def _executar_pipeline_legado(agente, repeticoes, reset):
    if reset:
        limpar_estado_ativo(agente)
    arquivos = []
    print(f"\n[ORQUESTRADOR] === {agente.upper()} (LEGADO): {repeticoes} repeticao(oes) ===")
    for i in range(1, repeticoes + 1):
        antes = listar_csvs_legado(agente)
        codigo = correr_uma_repeticao(agente, i, repeticoes)
        if codigo != 0:
            break
        depois = listar_csvs_legado(agente)
        novos = [p for p in depois if p not in antes]
        if novos:
            arquivos.append(novos[0])
    if arquivos:
        consolidar_legado(agente, arquivos)


def executar_plano(plano, reset):
    os.makedirs(BRAINS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(ANALISE_DIR, exist_ok=True)
    for agente, repeticoes in plano:
        if repeticoes <= 0:
            continue
        if agente in PIPELINE_NOVO:
            _executar_pipeline_novo(agente, repeticoes, reset)
        else:
            _executar_pipeline_legado(agente, repeticoes, reset)
    print("\n[ORQUESTRADOR] Plano de treino concluido.")


def modo_interativo():
    print("=" * 60)
    print("  TREINO CONTINUO — ALFINETE")
    print("=" * 60)
    print("Blue/Green: pipeline consolidado + emergency. Cada repeticao = 10k.")
    plano = []
    for agente in ["blue", "green", "ash"]:
        while True:
            resp = input(f"Quantas repeticoes de 10k para o {agente.upper()}? (0 = nenhuma): ").strip()
            if resp == "":
                resp = "0"
            if resp.isdigit():
                plano.append((agente, int(resp)))
                break
            print("  Escreve um numero inteiro.")

    reset = input("Apagar estado ATIVO antes (comecar do zero)? [s/N]: ").strip().lower() in (
        "s", "sim", "y", "yes")

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
    ap.add_argument("--ash", type=int, default=None, help="repeticoes de 10k do Ash (pipeline legado)")
    ap.add_argument("--batalhas", type=int, default=None,
                    help="orcamento fixo em batalhas por agente; arredondado para multiplo de 10k")
    ap.add_argument("--reset", action="store_true",
                    help="apaga brain/consolidado/emergencias ATIVOS antes de treinar")
    args = ap.parse_args()

    if args.batalhas is not None:
        reps = max(1, round(args.batalhas / BATALHAS_POR_REPETICAO))
        efetivo = reps * BATALHAS_POR_REPETICAO
        pedidos = {"blue": args.blue, "green": args.green, "ash": args.ash}
        agentes = [a for a, v in pedidos.items() if v is not None]
        if not agentes:
            agentes = ["blue", "green"]
        print(f"[ORQUESTRADOR] ORCAMENTO FIXO: {efetivo:,} batalhas por agente "
              f"({reps} repeticoes de {BATALHAS_POR_REPETICAO:,})")
        if efetivo != args.batalhas:
            print(f"               (ajustado de {args.batalhas:,} para {efetivo:,})")
        print(f"[ORQUESTRADOR] Agentes: {', '.join(agentes)}")
        executar_plano([(a, reps) for a in agentes], args.reset)
    elif args.blue is None and args.green is None and args.ash is None:
        modo_interativo()
    else:
        executar_plano([
            ("blue", args.blue or 0),
            ("green", args.green or 0),
            ("ash", args.ash or 0),
        ], args.reset)
