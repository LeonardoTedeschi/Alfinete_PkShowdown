"""
scripts/medir_regua.py — ANCORA DE CALIBRACAO DA REGUA.

Mede o InstinctBot contra o MaxDamage. Sempre que policy.py, execution.py,
physics.py ou masking.py mudarem, correr isto antes e depois permite saber quanto o
adversario de referencia mudou, e portanto interpretar as variacoes de Win Rate dos
agentes aprendizes.

  - Batalhas por omissao      : 1.000 (erro padrao ~1,3 pp; com 200 seria ~2,8 pp)
  - Bloco de consolidacao     : 200 batalhas
  - Batalhas simultaneas      : 3
  - Timer do servidor DESLIGADO (mesma razao do treino)
  - Ao fim: grafico da sessao + grafico HISTORICO das medicoes acumuladas

NAO treina. NAO grava cerebro. NAO altera nenhum ficheiro do projeto. So le.
Nao toca em policy.py, execution.py, physics.py nem masking.py, logo NAO afeta a
regua.

Executar da raiz do projeto:
    python -m scripts.medir_regua --batalhas 1000 --etiqueta "pos-6.16"
    python -m scripts.medir_regua --batalhas 200 --adversario instinct

Nota sobre o gráfico da sessao: o InstinctBot NAO aprende. O Win Rate por bloco e
uma reta horizontal com ruido binomial, nao uma curva de convergencia. O grafico que
importa e o HISTORICO (uma barra por versao do instinto, com IC 95%), gerado no fim.
"""

import argparse
import asyncio
import csv
import glob
import logging
import math
import os
import random
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from collections import deque

import numpy as np

from poke_env import AccountConfiguration, ServerConfiguration

from instinct.instinct_player import InstinctBot
from shared.env.maxdamage import MaxDamagePlayer
from shared.env.teams_train import RandomTeamFromPool, TEAMS_LIST as TIMES_TREINO
from shared.env.teams_eval import TEAMS_LIST as TIMES_EVAL

# --------------------------------------------------------------------------
# POOLS DE TIMES (--pool, 26/08/2026)
# --------------------------------------------------------------------------
# Porque isto existe: a avaliacao de generalizacao do Blue deu uma queda de 22,28 pp
# entre o pool de treino e o de holdout. Mas nessa avaliacao o ADVERSARIO tambem
# troca de pool, logo parte da queda pode nao ser o agente a falhar: pode ser o pool
# de eval ser simplesmente mais facil ou mais dificil.
#
# O InstinctBot NAO APRENDE: a politica dele e identica nos dois pools. Portanto
# medi-lo contra o MaxDamage em cada pool isola a DIFICULDADE DO MATERIAL da
# competencia do agente.
#
#   pools equivalentes  -> a queda de 22 pp e toda do agente
#   pools diferentes    -> parte da queda e do material, e tem de ser descontada
POOLS = {"treino": TIMES_TREINO, "eval": TIMES_EVAL}
from shared.console_report import RelatorioConsola
from shared.analysis.plot_graph import generate_graph

LOCAL = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

# ---- PROTOCOLO ----
BATALHAS_PADRAO = 1_000      # erro padrao ~1,3 pp num WR de 80%
BLOCO = 200                  # consolidacao (log + linha no CSV)
CONCORRENCIA = 3
BATTLE_FORMAT = "gen9nationaldex"
TURNOS_AUTO_TIE = 900        # o servidor encerra em 1000; acima disto e arrastamento

BRAINS_DIR = os.path.join(ROOT, "artefatos", "brains")
LOGS_DIR = os.path.join(ROOT, "artefatos", "logs")
CSV_HISTORICO = os.path.join(LOGS_DIR, "regua_historico.csv")


class InstinctBotMedido(InstinctBot):
    """InstinctBot com cronometro no choose_move.

    NAO altera nenhuma decisao: chama o super() e limita-se a registar o tempo.
    Existe porque o InstinctBot nao herda de TabularAgent e portanto nao tem
    pop_block_metrics(). A regua permanece intacta.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latencias = []
        # DIAGNOSTICO DE IMPASSE (28/08/2026). Buffer circular por batalha com os
        # ultimos N turnos. Ligado por --diagnostico; a zero nao guarda nada e nao
        # custa nada. NAO altera nenhuma decisao: le o estado e regista.
        self.turnos_por_batalha = {}
        self.janela_diagnostico = 0

    def _instantanea(self, battle, ordem):
        """Fotografia do turno: o que o instinto VIU e o que DECIDIU."""
        eu = getattr(battle, "active_pokemon", None)
        opp = getattr(battle, "opponent_active_pokemon", None)
        par = self.instinct.parser
        fis = self.instinct.physics

        def especie(m):
            return str(getattr(m, "species", "?"))

        def papel(m):
            try:
                return fis.get_role(m).name
            except Exception:
                return "?"

        try:
            matchup = par.get_matchup_state(eu, opp).name
        except Exception:
            matchup = "?"

        # O que saiu: nome do golpe, ou "SWITCH->especie" se foi troca.
        alvo = getattr(ordem, "order", ordem)
        if hasattr(alvo, "base_power"):
            decisao = str(getattr(alvo, "id", "?"))
        elif alvo is not None and hasattr(alvo, "species"):
            decisao = "SWITCH->" + especie(alvo)
        else:
            decisao = "?"

        meu_hp = sum(m.current_hp_fraction for m in battle.team.values())
        opp_hp = sum(m.current_hp_fraction for m in battle.opponent_team.values())

        return {
            "turno": getattr(battle, "turn", 0),
            "meu": especie(eu) if eu else "-",
            "meu_papel": papel(eu) if eu else "-",
            "meu_hp": f"{getattr(eu, 'current_hp_fraction', 0.0):.2f}" if eu else "",
            "opp": especie(opp) if opp else "-",
            "opp_papel": papel(opp) if opp else "-",
            "opp_hp": f"{getattr(opp, 'current_hp_fraction', 0.0):.2f}" if opp else "",
            "matchup": matchup,
            "decisao": decisao,
            "hp_equipa_meu": f"{meu_hp:.2f}",
            "hp_equipa_opp": f"{opp_hp:.2f}",
        }

    def choose_move(self, battle):
        t0 = time.perf_counter()
        ordem = super().choose_move(battle)
        self._latencias.append((time.perf_counter() - t0) * 1000.0)
        if self.janela_diagnostico:
            tag = getattr(battle, "battle_tag", None)
            if tag:
                buf = self.turnos_por_batalha.get(tag)
                if buf is None:
                    buf = deque(maxlen=self.janela_diagnostico)
                    self.turnos_por_batalha[tag] = buf
                try:
                    buf.append(self._instantanea(battle, ordem))
                except Exception:
                    pass   # diagnostico nunca pode partir a corrida
        return ordem

    def pop_latencia_ms(self):
        if not self._latencias:
            return 0.0
        media = sum(self._latencias) / len(self._latencias)
        self._latencias.clear()
        return media


def despejar_impasses(bot, limiar, csv_bat, csv_turnos):
    """Grava as batalhas ACIMA do limiar (ou empatadas) e os ultimos turnos delas.

    Chamar ANTES de reset_battles(), que limpa o dicionario de batalhas.

    Porque existe: `medir_regua --adversario instinct` deu 2,88% de empates e duracao
    media de 62,4 turnos, contra 0,00% e 27,9 turnos face ao MaxDamage — replicado em
    duas sementes. Sabe-se QUE ha impasse; falta saber ONDE. As duas hipoteses tem
    correcoes opostas:

      CICLO DE TROCAS      os ultimos turnos mostram SWITCH-> a alternar
      IMPASSE DEFENSIVO    mostram cura e status a alternar, com o HP de equipa
                           praticamente parado nos dois lados

    A coluna `hp_equipa_*` e o desempate: num impasse defensivo real, os dois totais
    ficam quase constantes ao longo da janela.
    """
    n_bat = 0
    novo_b = not os.path.exists(csv_bat)
    novo_t = not os.path.exists(csv_turnos)
    with open(csv_bat, "a", newline="", encoding="utf-8") as fb, \
         open(csv_turnos, "a", newline="", encoding="utf-8") as ft:
        wb, wt = csv.writer(fb), csv.writer(ft)
        if novo_b:
            wb.writerow(["Batalha", "Turnos", "Resultado", "Meu_Time", "Time_Adversario",
                         "Meus_Papeis", "Papeis_Adversario"])
        if novo_t:
            wt.writerow(["Batalha", "Turno", "Meu", "Meu_Papel", "Meu_HP",
                         "Adversario", "Papel_Adv", "HP_Adv", "Matchup", "Decisao",
                         "HP_Equipa_Meu", "HP_Equipa_Adv"])

        for tag, batalha in bot.battles.items():
            if not getattr(batalha, "finished", False):
                continue
            turnos = getattr(batalha, "turn", 0)
            empate = (batalha.won is None and not batalha.lost)
            if turnos < limiar and not empate:
                continue

            n_bat += 1
            resultado = "EMPATE" if empate else ("VITORIA" if batalha.won else "DERROTA")

            def papeis(equipa):
                out = []
                for m in equipa.values():
                    try:
                        out.append(bot.instinct.physics.get_role(m).name[:4])
                    except Exception:
                        out.append("?")
                return "/".join(out)

            wb.writerow([tag, turnos, resultado,
                         "/".join(str(m.species) for m in batalha.team.values()),
                         "/".join(str(m.species) for m in batalha.opponent_team.values()),
                         papeis(batalha.team), papeis(batalha.opponent_team)])

            for t in bot.turnos_por_batalha.get(tag, []):
                wt.writerow([tag, t["turno"], t["meu"], t["meu_papel"], t["meu_hp"],
                             t["opp"], t["opp_papel"], t["opp_hp"], t["matchup"],
                             t["decisao"], t["hp_equipa_meu"], t["hp_equipa_opp"]])

    bot.turnos_por_batalha.clear()
    return n_bat


def slug(texto):
    """Converte a etiqueta num nome de ficheiro seguro."""
    limpo = re.sub(r"[^A-Za-z0-9]+", "-", texto.strip()).strip("-").lower()
    return limpo[:60] or "sem-etiqueta"


CAB_HISTORICO = ["Data", "Etiqueta", "Estado", "Pool", "Adversario", "Semente", "Batalhas",
                 "Vitorias", "Derrotas", "Empates", "WinRate", "ErroPadrao_pp",
                 "Duracao_Media", "Duracao_Max", "Margem_Media", "Auto_Ties",
                 "Tempo_s", "Sessao"]


def escrever_historico(linha):
    """Acrescenta uma linha ao historico. A ETIQUETA e a chave: e ela que diz que
    estado do codigo foi medido. O indice de sessao vem de contar ficheiros na pasta
    e dessincroniza assim que se movem ficheiros entre pastas."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    novo = not os.path.exists(CSV_HISTORICO)
    with open(CSV_HISTORICO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(CAB_HISTORICO)
        w.writerow(linha)


def proximo_indice_log(prefixo):
    """Proximo numero de sessao livre, mesmo padrao do train_blue.py."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    existentes = glob.glob(os.path.join(LOGS_DIR, f"{prefixo}_*.csv"))
    ids = []
    for caminho in existentes:
        # aceita o formato antigo (regua_sessao_04.csv) e o novo, com etiqueta
        # (regua_sessao_05_v8-instinto-com-mec.csv)
        m = re.match(rf"{prefixo}_(\d+)(?:_.*)?\.csv$", os.path.basename(caminho))
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 1


def resumo_batalhas(jogador):
    """Metricas por batalha do bloco atual. Chamar ANTES de reset_battles()."""
    turnos, margens = [], []
    auto_ties = 0

    for batalha in jogador.battles.values():
        if not batalha.finished:
            continue
        turnos.append(batalha.turn)
        if batalha.turn >= TURNOS_AUTO_TIE:
            auto_ties += 1
        meus = sum(1 for p in batalha.team.values() if not p.fainted)
        dele = sum(1 for p in batalha.opponent_team.values() if not p.fainted)
        margens.append(meus - dele)

    n = max(len(turnos), 1)
    return {
        "duracao_media": sum(turnos) / n,
        "duracao_max": max(turnos) if turnos else 0,
        "margem_media": sum(margens) / n,
        "auto_ties": auto_ties,
    }


def gerar_grafico_historico(csv_historico, destino):
    """Serie historica das medicoes da regua: uma barra por execucao, com IC 95%.

    Este e o grafico que responde a pergunta real: quanto e que a regua mudou entre
    versoes do instinto? Uma curva por bloco nao responde a isso, porque o
    InstinctBot nao aprende.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    linhas = []
    with open(csv_historico, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha.get("Pool", "treino") != "treino":
                continue  # a ancora oficial e sempre medida no pool de treino
            if linha.get("Estado", "OK") != "OK":
                continue  # linhas INCOMPLETA sao de corridas abortadas
            if linha.get("Adversario") != "MaxDamage":
                continue  # o modo instinct-vs-instinct nao e ancora
            linhas.append(linha)

    if len(linhas) < 2:
        return None  # com uma so medicao nao ha serie para desenhar

    rotulos = [(l["Etiqueta"] or l["Data"][:8]) for l in linhas]
    wr = [float(l["WinRate"]) for l in linhas]
    erros = [1.96 * float(l["ErroPadrao_pp"]) for l in linhas]

    fig, ax = plt.subplots(figsize=(max(7, len(linhas) * 1.4), 5))
    x = range(len(linhas))
    ax.bar(x, wr, yerr=erros, capsize=6, color="#4C72B0", alpha=0.85)
    ax.axhline(50, color="grey", linestyle="--", linewidth=1)

    for i, (v, e) in enumerate(zip(wr, erros)):
        ax.text(i, v + e + 0.8, f"{v:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(rotulos, rotation=20, ha="right")
    ax.set_ylabel("Win Rate do InstinctBot vs MaxDamage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Ancora da regua: evolucao entre versoes do instinto\n"
                 "(barras de erro: IC 95%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return destino


async def main(args):
    os.makedirs(LOGS_DIR, exist_ok=True)

    # As equipas sao sorteadas com np.random.choice dentro do RandomTeamFromPool,
    # logo semear o modulo `random` sozinho NAO daria reprodutibilidade.
    random.seed(args.semente)
    np.random.seed(args.semente)

    instinct_vs_instinct = (args.adversario == "instinct")
    nome_adversario = "InstinctBot" if instinct_vs_instinct else "MaxDamage"
    times = POOLS[args.pool]

    bot = InstinctBotMedido(
        account_configuration=AccountConfiguration("ReguaInstinct", None),
        server_configuration=LOCAL, battle_format=BATTLE_FORMAT,
        team=RandomTeamFromPool(times),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False,
        log_level=logging.CRITICAL,
    )

    Adversario = InstinctBot if instinct_vs_instinct else MaxDamagePlayer
    adversario = Adversario(
        account_configuration=AccountConfiguration("ReguaOponente", None),
        server_configuration=LOCAL, battle_format=BATTLE_FORMAT,
        team=RandomTeamFromPool(times),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False,
        log_level=logging.CRITICAL,
    )

    # O ficheiro de sessao passa a levar a ETIQUETA no nome, alem do indice. Assim
    # um `ls` da pasta ja diz que estado do codigo cada corrida mediu, e mover
    # ficheiros entre pastas deixa de dessincronizar o indice do historico.
    etq = slug(f"{args.etiqueta}-{args.pool}")
    sessao = proximo_indice_log("regua_sessao")
    csv_path = os.path.join(LOGS_DIR, f"regua_sessao_{sessao:02d}_{etq}.csv")

    # --- DIAGNOSTICO DE IMPASSE ---
    bot.janela_diagnostico = args.diagnostico
    csv_impasse_bat = os.path.join(LOGS_DIR, f"impasse_{sessao:02d}_{etq}_batalhas.csv")
    csv_impasse_tur = os.path.join(LOGS_DIR, f"impasse_{sessao:02d}_{etq}_turnos.csv")
    impasses_total = 0

    # MESMAS 16 colunas do CSV de treino, para o ferramental de analise existente
    # funcionar sem alteracao. As colunas que nao se aplicam a um agente sem
    # aprendizado (Estados_Q, Epsilon, Visitas_Est, Confianca, Reward,
    # Ghost_Battles, Tamanho_KB) vao a ZERO por construcao, nao por falha.
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Batalhas", "WinRate_Bloco", "Vitorias", "Derrotas", "Estados_Q", "Epsilon",
             "Visitas_Est", "Confianca", "Reward", "Ghost_Battles",
             "Latencia_ms", "Margem_Media", "Duracao_Media", "Auto_Ties", "Tamanho_KB", "Tempo_s"])

    rel = RelatorioConsola(
        agente="REGUA",
        descricao=f"InstinctBot vs {nome_adversario} (ancora de calibracao)")
    rel.cabecalho(
        config={
            "Oponente": nome_adversario,
            "Formato": BATTLE_FORMAT,
            "Batalhas": args.batalhas,
            "Bloco": f"{BLOCO} batalhas",
            "Concorrencia": CONCORRENCIA,
            "Timer do servidor": "DESLIGADO",
            "Etiqueta": args.etiqueta or "(sem etiqueta)",
            "Pool de times": f"{args.pool} ({len(times)} times)",
            "Semente": args.semente,
            "Aprendizado": "NENHUM (o InstinctBot nao treina)",
        },
        caminhos={
            "Sessao": f"#{sessao:02d}",
            "Log CSV": csv_path,
            "Historico": CSV_HISTORICO,
        })

    # LINHA PARCIAL: escrita ANTES de comecar, marcada como INCOMPLETA. Se a corrida
    # abortar a meio (ver sessao 03, que morreu com [Invalid choice] de Z-move e nao
    # deixou registo nenhum no historico), fica pelo menos a prova de que existiu e
    # com que etiqueta. A linha final e escrita no fim; esta serve de sentinela.
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    escrever_historico([carimbo, args.etiqueta, "INCOMPLETA", args.pool, nome_adversario,
                        args.semente, args.batalhas, "", "", "", "", "", "", "", "",
                        "", "", f"{sessao:02d}"])

    total = 0
    vit_acum = emp_acum = fin_acum = 0
    dur_acum, marg_acum, ties_acum = [], [], 0
    dur_max_global = 0
    t0 = time.time()

    while total < args.batalhas:
        n_bloco = min(BLOCO, args.batalhas - total)
        await bot.battle_against(adversario, n_battles=n_bloco)
        total += n_bloco

        won = bot.n_won_battles
        tied = bot.n_tied_battles
        finished = bot.n_finished_battles
        derrotas = finished - won - tied
        wr = won / max(1, finished) * 100.0

        m = resumo_batalhas(bot)
        lat = bot.pop_latencia_ms()
        if args.diagnostico:
            # ANTES do reset_battles(), que limpa bot.battles.
            impasses_total += despejar_impasses(
                bot, args.limiar_impasse, csv_impasse_bat, csv_impasse_tur)
        bot.reset_battles()
        adversario.reset_battles()

        vit_acum += won
        emp_acum += tied
        fin_acum += finished
        dur_acum.append(m["duracao_media"])
        marg_acum.append(m["margem_media"])
        ties_acum += m["auto_ties"]
        dur_max_global = max(dur_max_global, m["duracao_max"])

        wall = time.time() - t0

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [total, f"{wr:.2f}", won, derrotas, 0, "0.0000", "0.00", "0.00",
                 "0", 0, f"{lat:.2f}", f"{m['margem_media']:.2f}",
                 f"{m['duracao_media']:.1f}", m["auto_ties"], "0", f"{wall:.0f}"])

        rel.bloco(batalhas=total, metricas={
            "win_rate": wr, "estados": 0, "epsilon": 0.0, "visitas": 0.0,
            "confianca": 0.0, "latencia_ms": lat,
            "margem_media": m["margem_media"], "duracao_media": m["duracao_media"],
            "auto_ties": m["auto_ties"], "tempo_s": wall, "reward": 0.0,
        })

    # ---- AGREGADO DA SESSAO ----
    wr_final = vit_acum / max(1, fin_acum)
    erro_padrao = math.sqrt(wr_final * (1 - wr_final) / max(1, fin_acum)) * 100.0
    wr_pct = wr_final * 100.0
    dur_media = sum(dur_acum) / max(1, len(dur_acum))
    marg_media = sum(marg_acum) / max(1, len(marg_acum))
    decorrido = time.time() - t0

    # ---- HISTORICO ACUMULATIVO (a ancora propriamente dita) ----
    escrever_historico([carimbo, args.etiqueta, "OK", args.pool, nome_adversario, args.semente,
                        fin_acum, vit_acum, fin_acum - vit_acum - emp_acum, emp_acum,
                        f"{wr_pct:.2f}", f"{erro_padrao:.2f}", f"{dur_media:.1f}",
                        dur_max_global, f"{marg_media:.2f}", ties_acum,
                        f"{decorrido:.0f}", f"{sessao:02d}"])

    # ---- GRAFICOS ----
    resultados = {
        "WIN RATE": f"{wr_pct:.2f}%  (erro padrao {erro_padrao:.2f} pp)",
        "IC 95%": f"[{wr_pct - 1.96 * erro_padrao:.2f}%, {wr_pct + 1.96 * erro_padrao:.2f}%]",
        "Duracao media": f"{dur_media:.1f} turnos (maxima {dur_max_global})",
        "Margem media": f"{marg_media:.2f} pokemon",
        "Auto-ties": f"{ties_acum} em {fin_acum}",
        "Historico CSV": CSV_HISTORICO,
    }
    if args.diagnostico:
        resultados["Impasses registados"] = (
            f"{impasses_total} batalha(s) com >= {args.limiar_impasse} turnos ou empatadas")
        resultados["Impasse CSV (batalhas)"] = csv_impasse_bat
        resultados["Impasse CSV (turnos)"] = csv_impasse_tur

    graf_path = os.path.join(LOGS_DIR, f"regua_grafico_{sessao:02d}.png")
    try:
        generate_graph(csv_path, graf_path,
                       agent="InstinctBot", opponent=nome_adversario,
                       total_battles=fin_acum, final_win_rate=wr_pct, final_states=0)
        resultados["Grafico sessao"] = f"{graf_path}  (reta + ruido: o instinto nao aprende)"
    except Exception as e:
        resultados["Grafico sessao"] = f"FALHOU: {e}"

    hist_path = os.path.join(LOGS_DIR, "regua_grafico_historico.png")
    try:
        feito = gerar_grafico_historico(CSV_HISTORICO, hist_path)
        resultados["Grafico historico"] = feito or "so ha 1 medicao; sem serie para desenhar"
    except Exception as e:
        resultados["Grafico historico"] = f"FALHOU: {e}"

    if dur_max_global >= TURNOS_AUTO_TIE:
        resultados["ALERTA"] = (
            f"batalha de {dur_max_global} turnos. Politica determinista dos dois lados: "
            "o ciclo de trocas de 6.16 NAO esta resolvido. Nao arrancar o ciclo de 200k.")

    rel.resumo_final(extra=resultados)


def parse_args():
    p = argparse.ArgumentParser(
        description="Mede o InstinctBot contra o MaxDamage (ancora da regua).")
    p.add_argument("--batalhas", type=int, default=BATALHAS_PADRAO,
                   help="numero de batalhas (1000 da erro padrao ~1,3 pp)")
    p.add_argument("--etiqueta", type=str, required=True,
                   help="OBRIGATORIA. Identifica o ESTADO DO CODIGO desta medicao, "
                        "ex: v8-instinto-com-mec. E a chave do historico: o numero de "
                        "sessao nao diz nada quando o instinto muda varias vezes no "
                        "mesmo dia.")
    p.add_argument("--semente", type=int, default=42,
                   help="semente para reprodutibilidade (random e numpy)")
    p.add_argument("--pool", type=str, default="treino", choices=["treino", "eval"],
                   help="pool de times para AMBOS os lados. Correr os dois e comparar "
                        "mede a dificuldade relativa do material, nao do agente.")
    p.add_argument("--diagnostico", type=int, default=0, metavar="N",
                   help="grava os ULTIMOS N turnos das batalhas longas ou empatadas, "
                        "para diagnosticar impasses. 0 (default) desliga. 5 a 10 "
                        "chega. Leitura pura: nao altera nenhuma decisao.")
    p.add_argument("--limiar-impasse", type=int, default=80, metavar="T",
                   help="batalhas com T ou mais turnos entram no diagnostico. Empates "
                        "entram sempre. Default 80 (a media instinto-vs-instinto e "
                        "62 turnos; vs MaxDamage e 28).")
    p.add_argument("--adversario", type=str, default="maxdamage",
                   choices=["maxdamage", "instinct"],
                   help="maxdamage para a ancora; instinct para o teste do ciclo de trocas")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(_args))
    else:
        asyncio.run(main(_args))
