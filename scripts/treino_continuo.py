"""
scripts/treino_continuo.py — ORQUESTRADOR de treino continuo do projeto ALFINETE.

Blue/Green usam o pipeline novo:
  - cada processo treina 10k e deixa um CSV de emergencia;
  - o orquestrador valida e incorpora atomicamente esse CSV ao consolidado;
  - depois da incorporacao, o CSV de emergencia e removido;
  - o processo fecha entre sessoes, libertando a RAM;
  - a cada 50k acumulados gera dois graficos (cumulativo + ultimos 10k),
    fotografias do brain/N(s,a) e checkpoint recuperavel, organizados por ciclo
    e, opcionalmente, por etiqueta experimental.

Os scripts individuais cuidam da persistencia do brain:
  - emergencia em 5k;
  - oficial em 10k.

O Ash permanece no pipeline legado porque o respetivo train_ash.py nao foi fornecido
nesta revisao.

USO (da raiz do projeto):
    python -m scripts.treino_continuo
    python -m scripts.treino_continuo --blue 5
    python -m scripts.treino_continuo --blue 5 --green 5
    python -m scripts.treino_continuo --blue 1 --green 1 --batalhas 650000 --ciclo B11 --etiqueta "baseline" --reset
"""

import argparse
import csv
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

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
CHECKPOINT_CADA_BATALHAS = 50_000
ORCAMENTO_PRINCIPAL_BATALHAS = 650_000
PIPELINE_NOVO = {"blue", "green"}

TREINOS = {
    "blue": "scripts.train_blue",
    "green": "scripts.train_green",
    "ash": "scripts.train_ash",
}


def normalizar_ciclo(ciclo):
    """Normaliza o identificador do ciclo para a convencao historica B_N.

    Aceita, por exemplo, ``B11`` ou ``B_11`` e devolve ``B_11``. Outros
    identificadores continuam permitidos, apenas sanitizados.
    """
    if ciclo is None:
        return None
    bruto = str(ciclo).strip()
    m = re.fullmatch(r"[Bb]_?(\d+)", bruto)
    if m:
        return f"B_{int(m.group(1))}"
    valor = re.sub(r"[^A-Za-z0-9_-]+", "-", bruto).strip("-_")
    return valor or None


def normalizar_etiqueta(etiqueta):
    """Normaliza uma etiqueta humana para uso seguro em nomes de pasta.

    A etiqueta e opcional e serve apenas para identificar a corrida/hipotese.
    Ela nao e enviada aos agentes e nao altera qualquer hiperparametro.
    """
    if etiqueta is None:
        return None
    bruto = str(etiqueta).strip()
    if not bruto:
        return None
    valor = re.sub(r"[^A-Za-z0-9._-]+", "-", bruto).strip("-_.")
    return valor or None


def caminho_analise_marco(ciclo, agente, total_batalhas, etiqueta=None):
    """Pasta dos artefatos de analise de um agente num marco de 50k.

    Sem etiqueta, preserva a estrutura historica:
        artefatos/logs/analise/B_11/Blue/050k

    Com etiqueta, isola corridas experimentais sem sobrescrever artefatos:
        artefatos/logs/analise/B_11/replay-fixo/Blue/050k
    """
    ciclo = normalizar_ciclo(ciclo)
    if not ciclo:
        return None
    partes = [ANALISE_DIR, ciclo]
    etiqueta_slug = normalizar_etiqueta(etiqueta)
    if etiqueta_slug:
        partes.append(etiqueta_slug)
    k = total_batalhas // 1000
    partes.extend([agente.capitalize(), f"{k:03d}k"])
    return os.path.join(*partes)


def caminho_checkpoint_dir(ciclo, total_batalhas, etiqueta=None):
    ciclo = normalizar_ciclo(ciclo)
    if not ciclo:
        return None
    partes = [BRAINS_DIR, ciclo]
    etiqueta_slug = normalizar_etiqueta(etiqueta)
    if etiqueta_slug:
        partes.append(etiqueta_slug)
    partes.append("checkpoints")
    k = total_batalhas // 1000
    partes.append(f"{k:03d}k")
    return os.path.join(*partes)


def _total_consolidado(agente):
    cab, linhas = _ler_csv(caminho_csv_consolidado(agente))
    if not cab or not linhas:
        return 0
    try:
        return int(float(linhas[-1][0]))
    except (ValueError, IndexError):
        return 0


def criar_checkpoint(agente, ciclo, consolidado, total_batalhas, estados, rel_saude, etiqueta=None):
    """Preserva automaticamente brain + CSV a cada marco de 50k.

    Nao pausa o treinamento. O checkpoint e uma fotografia recuperavel do ciclo e
    fica separado dos brains ativos.
    """
    if total_batalhas <= 0 or total_batalhas % CHECKPOINT_CADA_BATALHAS != 0:
        return None
    pasta = caminho_checkpoint_dir(ciclo, total_batalhas, etiqueta)
    if not pasta:
        return None
    os.makedirs(pasta, exist_ok=True)

    brain_src = caminho_cerebro(agente)
    csv_src = consolidado
    brain_dst = os.path.join(pasta, f"{agente}_brain.pkl")
    csv_dst = os.path.join(pasta, f"{agente}_treino_consolidado.csv")
    meta_dst = os.path.join(pasta, f"checkpoint_{agente}.txt")

    def copia_atomica(origem, destino):
        temp = destino + ".tmp"
        shutil.copy2(origem, temp)
        os.replace(temp, destino)

    copia_atomica(brain_src, brain_dst)
    copia_atomica(csv_src, csv_dst)

    with open(meta_dst + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"Ciclo: {normalizar_ciclo(ciclo)}\n")
        f.write(f"Etiqueta: {str(etiqueta).strip() if etiqueta else '(sem etiqueta)'}\n")
        f.write(f"Agente: {agente.capitalize()}\n")
        f.write(f"Batalhas: {total_batalhas}\n")
        f.write(f"Estados_Q: {estados}\n")
        f.write(f"Maior_abs_Q: {rel_saude.get('maior_q', float('nan'))}\n")
        f.write(f"Saude: {rel_saude.get('motivo', '')}\n")
        f.write(f"Data: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Brain_ativo: {brain_src}\n")
        f.write(f"CSV_ativo: {csv_src}\n")
    os.replace(meta_dst + ".tmp", meta_dst)

    print(f"[ORQUESTRADOR] Checkpoint {total_batalhas // 1000:03d}k preservado: {pasta}")
    return pasta


def confirmar_extensao(agente, total_atual, total_planejado):
    """Pede confirmacao somente para ultrapassar o orcamento principal de 650k."""
    if total_atual < ORCAMENTO_PRINCIPAL_BATALHAS:
        return True
    print("\n" + "=" * 70)
    print(f"[ORQUESTRADOR] {agente.upper()} atingiu o orcamento principal de "
          f"{ORCAMENTO_PRINCIPAL_BATALHAS:,} batalhas.")
    print(f"O plano atual pretende continuar ate aproximadamente {total_planejado:,}.")
    print("A extensao (por exemplo, 650k -> 800k) e um novo trecho de treino e exige confirmacao.")
    print("=" * 70)
    resp = input("Autorizar treinamento alem de 650k? [s/N]: ").strip().lower()
    return resp in ("s", "sim", "y", "yes")


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


def _csv_ultimos_blocos(caminho_origem, caminho_temp, n_blocos=10):
    """Cria um CSV temporario com os ultimos blocos, renumerados desde 1k.

    A renumeracao reproduz a visualizacao das antigas sessoes independentes de 10k:
    mesmo num marco de 100k, o grafico da janela mostra 1k..10k, enquanto o nome do
    ficheiro e o quadro de resumo deixam claro que se trata do marco de 100k.
    """
    cab, linhas = _ler_csv(caminho_origem)
    if not cab or not linhas:
        raise RuntimeError(f"CSV consolidado vazio: {caminho_origem}")
    janela = [list(row) for row in linhas[-n_blocos:]]
    if not janela:
        raise RuntimeError("Nao ha blocos suficientes para gerar a janela de treino.")

    primeiro = int(float(janela[0][0]))
    offset = primeiro - BLOCO_BATALHAS
    for row in janela:
        row[0] = str(int(float(row[0])) - offset)

    with open(caminho_temp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        w.writerows(janela)


def gerar_artefatos_marco(agente, consolidado, total_batalhas, estados, ciclo, etiqueta=None):
    """Gera artefatos permanentes somente nos marcos de 50k.

    Os processos individuais de 10k continuam produzindo apenas o CSV temporario
    necessario para consolidacao e o brain. O orquestrador concentra a
    instrumentacao permanente em 50k, 100k, 150k...

    Em cada marco sao gerados:
      - grafico cumulativo desde o inicio da corrida;
      - grafico dos ultimos 10k, para comparacao com as series historicas;
      - dashboard de analise do brain;
      - CSV + dashboard N(s,a) das escolhas de acao.
    """
    if total_batalhas <= 0 or total_batalhas % ANALISE_CADA_BATALHAS != 0:
        return None

    k = total_batalhas // 1000
    nome = agente.capitalize()
    etiqueta_txt = str(etiqueta).strip() if etiqueta else None
    nome_grafico = f"{nome} [{etiqueta_txt}]" if etiqueta_txt else nome

    print("\n" + "=" * 72)
    print(f"[MARCO {k:03d}k] iniciando artefatos de {nome}" +
          (f" | etiqueta: {etiqueta_txt}" if etiqueta_txt else ""))

    pasta = caminho_analise_marco(ciclo, agente, total_batalhas, etiqueta)
    if not pasta:
        print(f"[MARCO {k:03d}k] ERRO: ciclo ausente; artefatos nao gerados.")
        print("=" * 72)
        return None

    try:
        os.makedirs(pasta, exist_ok=True)
    except Exception as e:
        print(f"[MARCO {k:03d}k] ERRO AO CRIAR PASTA: {pasta} | {e}")
        print("=" * 72)
        return None

    print(f"[MARCO {k:03d}k] pasta: {pasta}")

    ultima = _ultima_metrica(consolidado) or {}
    try:
        wr = float(ultima.get("WinRate_Bloco", 0.0))
    except (TypeError, ValueError):
        wr = 0.0

    falhas = []

    # 1) Curva acumulada: 0 -> marco atual.
    graf_acumulado = os.path.join(pasta, f"{agente}_treino_{k:03d}k_acumulado.png")
    try:
        generate_graph(
            consolidado, graf_acumulado,
            agent=nome_grafico, opponent="Instinto",
            total_battles=total_batalhas,
            final_win_rate=wr,
            final_states=estados,
        )
        if not os.path.exists(graf_acumulado) or os.path.getsize(graf_acumulado) == 0:
            raise RuntimeError("generate_graph terminou sem produzir PNG valido")
        print(f"[MARCO {k:03d}k] grafico acumulado OK: {graf_acumulado}")
    except Exception as e:
        falhas.append(f"grafico acumulado: {e}")
        print(f"[MARCO {k:03d}k] FALHA grafico acumulado: {e}")

    # 2) Ultimos 10k: mantido explicitamente para comparabilidade historica.
    graf_10k = os.path.join(pasta, f"{agente}_treino_{k:03d}k_ultimos_10k.png")
    csv_temp = os.path.join(pasta, f".{agente}_{k:03d}k_ultimos_10k.tmp.csv")
    try:
        _csv_ultimos_blocos(consolidado, csv_temp, n_blocos=10)
        generate_graph(
            csv_temp, graf_10k,
            agent=nome_grafico, opponent="Instinto",
            total_battles=10_000,
            final_win_rate=wr,
            final_states=estados,
        )
        if not os.path.exists(graf_10k) or os.path.getsize(graf_10k) == 0:
            raise RuntimeError("generate_graph terminou sem produzir PNG valido")
        print(f"[MARCO {k:03d}k] grafico ultimos 10k OK: {graf_10k}")
    except Exception as e:
        falhas.append(f"grafico ultimos 10k: {e}")
        print(f"[MARCO {k:03d}k] FALHA grafico ultimos 10k: {e}")
    finally:
        try:
            if os.path.exists(csv_temp):
                os.remove(csv_temp)
        except OSError:
            pass

    # 3) Dashboard do brain e 4) analise N(s,a).
    brain = caminho_cerebro(agente)
    etiqueta_arquivo = f"{nome}_{k:03d}k"
    if normalizar_etiqueta(etiqueta):
        etiqueta_arquivo += f"_{normalizar_etiqueta(etiqueta)}"

    try:
        analyze_brain(brain, pasta, etiqueta_arquivo)
        print(f"[MARCO {k:03d}k] dashboard brain OK")
    except Exception as e:
        falhas.append(f"dashboard brain: {e}")
        print(f"[MARCO {k:03d}k] FALHA dashboard brain: {e}")

    try:
        resultado_acoes = analyze_action_choices(brain, pasta, etiqueta_arquivo)
        if isinstance(resultado_acoes, str) and resultado_acoes.startswith("SEM DADOS"):
            falhas.append(f"N(s,a): {resultado_acoes}")
            print(f"[MARCO {k:03d}k] AVISO N(s,a): {resultado_acoes}")
        else:
            print(f"[MARCO {k:03d}k] N(s,a) OK" +
                  (f": {resultado_acoes}" if resultado_acoes else ""))
    except Exception as e:
        falhas.append(f"N(s,a): {e}")
        print(f"[MARCO {k:03d}k] FALHA N(s,a): {e}")

    if falhas:
        print(f"[MARCO {k:03d}k] CONCLUIDO COM {len(falhas)} AVISO(S)/FALHA(S):")
        for item in falhas:
            print(f"    - {item}")
    else:
        print(f"[MARCO {k:03d}k] CONCLUIDO: todos os artefatos gerados.")
    print("=" * 72)
    return pasta


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


def _executar_pipeline_novo(agente, repeticoes, reset, ciclo, etiqueta=None):
    if reset:
        limpar_estado_ativo(agente)

    ciclo = normalizar_ciclo(ciclo)
    if not ciclo:
        print(f"[ORQUESTRADOR] RECUSADO: informe --ciclo para {agente.upper()} (ex.: B11).")
        return

    total_inicial = _total_consolidado(agente)
    total_planejado = total_inicial + repeticoes * BATALHAS_POR_REPETICAO
    extensao_confirmada = total_inicial > ORCAMENTO_PRINCIPAL_BATALHAS

    print(f"\n[ORQUESTRADOR] === {agente.upper()}: {repeticoes} repeticao(oes) de 10k | ciclo {ciclo}"
          f" | etiqueta {str(etiqueta).strip() if etiqueta else '(sem etiqueta)'} ===")
    print(f"[ORQUESTRADOR] Total inicial: {total_inicial:,} | planejado: {total_planejado:,}")

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

    total_atual = total_inicial
    for i in range(1, repeticoes + 1):
        if (not extensao_confirmada
                and total_planejado > ORCAMENTO_PRINCIPAL_BATALHAS
                and total_atual >= ORCAMENTO_PRINCIPAL_BATALHAS):
            if not confirmar_extensao(agente, total_atual, total_planejado):
                print(f"[ORQUESTRADOR] Treino de {agente} encerrado em {total_atual:,} batalhas por opcao do utilizador.")
                break
            extensao_confirmada = True
            print(f"[ORQUESTRADOR] Extensao alem de {ORCAMENTO_PRINCIPAL_BATALHAS:,} autorizada.")

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

        # O proprio total acumulado e a fonte de verdade do marco. Cada processo
        # tem 10k, logo 50k corresponde a cinco execucoes completas.
        if total_cumulativo % ANALISE_CADA_BATALHAS == 0:
            gerar_artefatos_marco(
                agente, consolidado, total_cumulativo, estados_depois, ciclo, etiqueta
            )
            criar_checkpoint(
                agente, ciclo, consolidado, total_cumulativo, estados_depois, rel, etiqueta
            )
        else:
            faltam = ANALISE_CADA_BATALHAS - (total_cumulativo % ANALISE_CADA_BATALHAS)
            print(f"[ORQUESTRADOR] Proximo marco de analise em {faltam:,} batalhas.")
        total_atual = total_cumulativo


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


def executar_plano(plano, reset, ciclo=None, etiqueta=None):
    os.makedirs(BRAINS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(ANALISE_DIR, exist_ok=True)
    for agente, repeticoes in plano:
        if repeticoes <= 0:
            continue
        if agente in PIPELINE_NOVO:
            _executar_pipeline_novo(agente, repeticoes, reset, ciclo, etiqueta)
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
    ciclo = input("Ciclo experimental (ex.: B11): ").strip() or None
    etiqueta = input("Etiqueta da corrida (opcional; ex.: replay-fixo): ").strip() or None

    if sum(r for _, r in plano) == 0:
        print("Nada a treinar. A sair.")
        return
    print("\nPlano: " + ", ".join(f"{a}={r}" for a, r in plano if r > 0) +
          f" | reset={'sim' if reset else 'nao'}")
    if input("Confirmar e iniciar? [S/n]: ").strip().lower() in ("", "s", "sim", "y", "yes"):
        executar_plano(plano, reset, ciclo, etiqueta)
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
    ap.add_argument("--ciclo", type=str, default=None,
                    help="versao do Blue que identifica o ciclo experimental, ex.: B11")
    ap.add_argument("--etiqueta", type=str, default=None,
                    help="rotulo opcional da corrida/hipotese; organiza artefatos e aparece nos graficos")
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
        executar_plano([(a, reps) for a in agentes], args.reset, args.ciclo, args.etiqueta)
    elif args.blue is None and args.green is None and args.ash is None:
        modo_interativo()
    else:
        executar_plano([
            ("blue", args.blue or 0),
            ("green", args.green or 0),
            ("ash", args.ash or 0),
        ], args.reset, args.ciclo, args.etiqueta)
