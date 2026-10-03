"""
scripts/avaliar_generalizacao.py — avaliacao externa unificada contra Cynthia.

PAPEL NO PROTOCOLO DO ALFINETE
------------------------------
O treino principal continua a ser feito contra o InstinctBot (relacao mestre-aprendiz).
A avaliacao principal passa a ser EXTERNA: todos os agentes sao medidos contra a mesma
CynthiaPlatinumBot, que nao aprende e nao partilha policy/masking/physics com os agentes.

Este unico script substitui duas funcoes que antes estavam separadas:

  1. avaliacao de generalizacao treino x holdout de Blue/Green/Ash;
  2. calibracao de baselines nao-aprendizes (Instinct e MaxDamage).

AGENTES SUPORTADOS
------------------
  blue       HybridAgent, cerebro congelado durante a avaliacao
  green      PureAgent, cerebro congelado durante a avaliacao
  ash        AshAgent, se existir no projeto e houver cerebro
  instinct   InstinctBot, nao aprende
  maxdamage  MaxDamagePlayer, nao aprende

REGUA
-----
Sempre CynthiaPlatinumBot, com perfil:

    AI_FLAG_BASIC | AI_FLAG_EVAL_ATTACK | AI_FLAG_EXPERT

Formato: gen9nationaldex.

CONDICOES
---------
TREINO:
    avaliado e Cynthia sorteiam equipes de shared/env/teams_train.py.

HOLDOUT:
    avaliado e Cynthia sorteiam equipes de shared/env/teams_eval.py.

Logo a diferenca TREINO -> HOLDOUT mede desempenho sob mudanca de distribuicao do
material, mantendo a POLITICA adversaria fixa. Nao se aplica mais o antigo desconto
fixo de vies do Instinct: aquele valor pertencia a outra regua e nao e transferivel
para Cynthia. MaxDamage e Instinct podem ser executados no mesmo script para ajudar a
contextualizar a dificuldade relativa dos dois pools, sem correcao automatica.

APRENDIZADO DESLIGADO
---------------------
Para agentes com cerebro:
  alpha = 0
  epsilon = 0
  replay bloqueado
  eligibility traces desligadas

O .pkl e apenas lido. Nunca e salvo por este script.

USO
---
Da raiz do projeto:

    python -m scripts.avaliar_generalizacao --agente blue --ciclo B11 --batalhas 5000 --etiqueta "100k"
    python -m scripts.avaliar_generalizacao --agente instinct --ciclo B11 --batalhas 5000 --etiqueta "pre-main"
    python -m scripts.avaliar_generalizacao --agente maxdamage --ciclo B11 --batalhas 5000 --etiqueta "baseline"
    python -m scripts.avaliar_generalizacao --todos --ciclo B11 --batalhas 5000 --etiqueta "100k"

Para smoke test da Cynthia:

    python -m scripts.avaliar_generalizacao --agente maxdamage --ciclo B11 --batalhas 20 \
        --etiqueta "smoke-cynthia" --diagnostico-cynthia
"""

import argparse
import asyncio
import csv
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env import AccountConfiguration, ServerConfiguration

from instinct.instinct_player import InstinctBot
from qlearning.hybrid_agent import HybridAgent
from qlearning.pure_agent import PureAgent
from shared.env.cynthia_platinum import (
    AI_PROFILE,
    BASELINE_NAME,
    BASELINE_SPEC_VERSION,
    CynthiaPlatinumBot,
)
from shared.env.maxdamage import MaxDamagePlayer
from shared.env.teams_train import RandomTeamFromPool, TEAMS_LIST as TIMES_TREINO
from shared.env.teams_eval import TEAMS_LIST as TIMES_HOLDOUT
from shared.analysis.plot_generalizacao import (
    DiagnosticoDecisoes,
    salvar_diagnosticos,
    gerar_grafico,
)

try:
    from qlearning.ash_agent import AshAgent
except ImportError:
    AshAgent = None


# --------------------------------------------------------------------------
# Configuracao
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINS_DIR = os.path.join(ROOT, "artefatos", "brains")
LOGS_DIR = os.path.join(ROOT, "artefatos", "logs")
GENERALIZACAO_DIR = os.path.join(LOGS_DIR, "Generalizacao")

SERVIDOR = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/",
)
FORMATO = "gen9nationaldex"
BATALHAS = 1_000  # por condicao
CONCORRENCIA = 5  # mantido em 1 para reprodutibilidade da avaliacao
SEMENTE_PADRAO = 42

NIVEIS_LOG = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
NIVEL_LOG = logging.CRITICAL

# Cada entrada explicita se o agente possui cerebro. Isso evita espalhar ifs por todo
# o script e permite que baselines nao-aprendizes usem exatamente o mesmo protocolo.
AGENTES = {
    "blue": {
        "classe": HybridAgent,
        "brain": "blue_brain.pkl",
        "nome": "EvBlue",
        "aprende": True,
    },
    "green": {
        "classe": PureAgent,
        "brain": "green_brain.pkl",
        "nome": "EvGreen",
        "aprende": True,
    },
    "instinct": {
        "classe": InstinctBot,
        "brain": None,
        "nome": "EvInst",
        "aprende": False,
    },
    "maxdamage": {
        "classe": MaxDamagePlayer,
        "brain": None,
        "nome": "EvMax",
        "aprende": False,
    },
}

if AshAgent is not None:
    AGENTES["ash"] = {
        "classe": AshAgent,
        "brain": "ash_brain.pkl",
        "nome": "EvAsh",
        "aprende": True,
    }

# Ordem deliberada para --todos: primeiro as duas baselines, depois os aprendizes.
ORDEM_TODOS = ["maxdamage", "instinct", "green", "blue"]
if "ash" in AGENTES:
    ORDEM_TODOS.append("ash")

NOMES_ARQUIVO = {
    "blue": "Blue",
    "green": "Green",
    "instinct": "Instinct",
    "maxdamage": "MaxDamage",
    "ash": "Ash",
}


def _ciclo_seguro(ciclo):
    """Normaliza o identificador usado em nomes de pasta/arquivo."""
    valor = re.sub(r"[^A-Za-z0-9_-]+", "-", str(ciclo).strip()).strip("-_")
    if not valor:
        raise ValueError("ciclo vazio ou invalido")
    return valor


def _pasta_tentativas(agente, ciclo):
    nome = NOMES_ARQUIVO.get(agente, agente.capitalize())
    return os.path.join(
        GENERALIZACAO_DIR,
        f"{nome}VsCynthia",
        _ciclo_seguro(ciclo),
    )


def _proxima_tentativa(agente, ciclo):
    """Retorna (numero, caminho) sem sobrescrever tentativas anteriores."""
    pasta = _pasta_tentativas(agente, ciclo)
    os.makedirs(pasta, exist_ok=True)
    nome = NOMES_ARQUIVO.get(agente, agente.capitalize())
    ciclo_norm = _ciclo_seguro(ciclo)
    padrao = re.compile(
        rf"^Generalizacao_{re.escape(nome)}_{re.escape(ciclo_norm)}_(\d+)\.csv$",
        re.IGNORECASE,
    )
    usados = []
    for arquivo in os.listdir(pasta):
        m = padrao.match(arquivo)
        if m:
            usados.append(int(m.group(1)))
    numero = max(usados, default=0) + 1
    caminho = os.path.join(
        pasta, f"Generalizacao_{nome}_{ciclo_norm}_{numero:02d}.csv"
    )
    return numero, caminho


# --------------------------------------------------------------------------
# Instrumentacao generica, neutra em relacao a policy
# --------------------------------------------------------------------------
def _instrumentar(jogador, etiqueta):
    """Mede latencia e expoe excecoes sem alterar a decisao devolvida."""
    original = jogador.choose_move
    jogador._eval_tempo_decisao = 0.0
    jogador._eval_n_decisoes = 0
    jogador._eval_n_erros = 0

    def com_instrumentacao(battle):
        t0 = time.perf_counter()
        try:
            return original(battle)
        except Exception:
            jogador._eval_n_erros += 1
            print()
            print("!" * 78)
            print(f"  EXCECAO em choose_move de {etiqueta}")
            print(
                f"  batalha: {getattr(battle, 'battle_tag', '?')} | "
                f"turno: {getattr(battle, 'turn', '?')}"
            )
            activo = getattr(battle, "active_pokemon", None)
            if activo is not None:
                print(
                    f"  ativo: {getattr(activo, 'species', '?')} | "
                    f"item: {getattr(activo, 'item', '?')} | "
                    f"can_mega: {getattr(battle, 'can_mega_evolve', '?')} | "
                    f"can_z: {getattr(battle, 'can_z_move', '?')} | "
                    f"can_tera: {getattr(battle, 'can_tera', '?')}"
                )
            print("!" * 78)
            traceback.print_exc()
            print("!" * 78, flush=True)
            raise
        finally:
            jogador._eval_tempo_decisao += time.perf_counter() - t0
            jogador._eval_n_decisoes += 1

    jogador.choose_move = com_instrumentacao
    return jogador


def _latencia_ms(jogador):
    n = int(getattr(jogador, "_eval_n_decisoes", 0) or 0)
    if n <= 0:
        return 0.0
    total = float(getattr(jogador, "_eval_tempo_decisao", 0.0) or 0.0)
    return total / n * 1000.0


def _brain(jogador):
    return getattr(jogador, "brain", None)


def _congelar_aprendizado(jogador):
    """Desliga aprendizado sem gravar o cerebro posteriormente."""
    b = _brain(jogador)
    if b is None:
        return

    if hasattr(b, "epsilon"):
        b.epsilon = 0.0
    if hasattr(b, "min_epsilon"):
        b.min_epsilon = 0.0
    if hasattr(b, "alpha"):
        b.alpha = 0.0
    if hasattr(b, "min_alpha"):
        b.min_alpha = 0.0
    if hasattr(b, "replay_min_states"):
        b.replay_min_states = 10 ** 12
    if hasattr(b, "use_traces"):
        b.use_traces = False
    if hasattr(b, "usar_encolhimento"):
        b.usar_encolhimento = False


def _numero_estados(jogador):
    b = _brain(jogador)
    if b is None:
        return 0
    q = getattr(b, "q_table", None)
    return len(q) if q is not None else 0


# --------------------------------------------------------------------------
# Construcao dos jogadores
# --------------------------------------------------------------------------
def construir(agente, times, sufixo, semente, diagnostico_cynthia=False):
    """Cria agente avaliado + Cynthia para uma condicao."""
    cfg = AGENTES[agente]
    classe = cfg["classe"]

    kwargs = dict(
        account_configuration=AccountConfiguration(f"{cfg['nome']}{sufixo}", None),
        server_configuration=SERVIDOR,
        battle_format=FORMATO,
        team=RandomTeamFromPool(times),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False,
        log_level=NIVEL_LOG,
    )

    if cfg["brain"] is not None:
        kwargs["brain_file"] = os.path.join(BRAINS_DIR, cfg["brain"])

    jogador = classe(**kwargs)
    _congelar_aprendizado(jogador)

    # A policy da regua e fixa; a seed controla apenas os desempates/rolls internos.
    cynthia = CynthiaPlatinumBot(
        account_configuration=AccountConfiguration(f"Cynthia{sufixo}", None),
        server_configuration=SERVIDOR,
        battle_format=FORMATO,
        team=RandomTeamFromPool(times),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False,
        log_level=NIVEL_LOG,
        seed=semente,
        diagnostico=diagnostico_cynthia,
    )

    _instrumentar(jogador, f"{agente.upper()}{sufixo}")
    return jogador, cynthia


# --------------------------------------------------------------------------
# Metricas uniformes para agentes com e sem cerebro
# --------------------------------------------------------------------------
def _resumo_batalhas(jogador):
    turnos = []
    margens = []

    for batalha in getattr(jogador, "battles", {}).values():
        if not getattr(batalha, "finished", False):
            continue

        turnos.append(int(getattr(batalha, "turn", 0) or 0))
        meus = sum(1 for p in batalha.team.values() if not p.fainted)
        dele = sum(1 for p in batalha.opponent_team.values() if not p.fainted)
        # Margem ASSINADA do ponto de vista do agente avaliado.
        margens.append(meus - dele)

    return {
        "duracao": sum(turnos) / len(turnos) if turnos else 0.0,
        "margem": sum(margens) / len(margens) if margens else 0.0,
        "duracao_max": max(turnos) if turnos else 0,
    }


async def avaliar_condicao(
    agente,
    times,
    rotulo,
    sufixo,
    n_batalhas,
    semente,
    diagnostico_cynthia=False,
):
    """Corre uma condicao contra Cynthia e devolve metricas uniformes."""
    # Controla os geradores locais. Concorrencia 1 reduz variacao de ordem,
    # mas nao fixa a aleatoriedade do servidor Showdown.
    random.seed(semente)
    np.random.seed(semente)

    jogador, cynthia = construir(
        agente,
        times,
        sufixo,
        semente,
        diagnostico_cynthia=diagnostico_cynthia,
    )

    # 03/10/2026: referencia coletada ANTES das batalhas. O plot antigo recebia
    # apenas WR e contagem de estados novos; nao podia reconstruir cobertura.
    # Observadores chamam cada decisao/executor original exatamente uma vez.
    diagnostico = DiagnosticoDecisoes(jogador)
    estados_antes = _numero_estados(jogador)
    t0 = time.perf_counter()
    try:
        await jogador.battle_against(cynthia, n_battles=n_batalhas)
    finally:
        diagnostico.chamadas_choose_move = jogador._eval_n_decisoes
        diagnostico.erros_choose_move = jogador._eval_n_erros
        diagnostico.fechar()
    tempo = time.perf_counter() - t0
    estados_depois = _numero_estados(jogador)

    terminadas = int(getattr(jogador, "n_finished_battles", 0) or 0)
    vitorias = int(getattr(jogador, "n_won_battles", 0) or 0)
    empates = int(getattr(jogador, "n_tied_battles", 0) or 0)
    derrotas = max(0, terminadas - vitorias - empates)
    wr = vitorias / max(1, terminadas) * 100.0

    m = _resumo_batalhas(jogador)

    return {
        "rotulo": rotulo,
        "wr": wr,
        "batalhas": terminadas,
        "vitorias": vitorias,
        "derrotas": derrotas,
        "empates": empates,
        "margem": m["margem"],
        "duracao": m["duracao"],
        "duracao_max": m["duracao_max"],
        "latencia": _latencia_ms(jogador),
        "estados_antes": estados_antes,
        "estados_depois": estados_depois,
        "tempo": tempo,
        "diagnostico": diagnostico,
    }


# --------------------------------------------------------------------------
# Estatistica e persistencia
# --------------------------------------------------------------------------
def _ep_diferenca(treino, holdout):
    n1 = max(1, treino["batalhas"])
    n2 = max(1, holdout["batalhas"])
    p1 = treino["wr"] / 100.0
    p2 = holdout["wr"] / 100.0
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) * 100.0


CABECALHO = [
    "Data",
    "Ciclo",
    "Tentativa",
    "Etiqueta",
    "Regua",
    "Regua_Versao",
    "Formato",
    "AI_Profile",
    "Semente",
    "Agente",
    "Batalhas_Por_Condicao",
    "WR_Treino",
    "WR_Holdout",
    "Delta_Treino_Menos_Holdout_pp",
    "EP_Diferenca_pp",
    "Margem_Treino",
    "Margem_Holdout",
    "Duracao_Treino",
    "Duracao_Holdout",
    "Ties_Treino",
    "Ties_Holdout",
    "Estados_Novos_Treino",
    "Estados_Novos_Holdout",
    "Tempo_Treino_s",
    "Tempo_Holdout_s",
    "Batalhas_Treino",
    "Batalhas_Holdout",
    "Diagnostico_Versao",
]


def registar(agente, treino, holdout, etiqueta, semente, ciclo):
    """Cria um CSV independente por tentativa, agente, confronto e ciclo.

    Estrutura:
      artefatos/logs/Generalizacao/<Agente>VsCynthia/<Ciclo>/
          Generalizacao_<Agente>_<Ciclo>_<NN>.csv

    O numero NN e descoberto a partir dos arquivos existentes e nunca sobrescreve
    uma tentativa anterior.
    """
    if not treino or not holdout:
        return None
    if min(treino["batalhas"], holdout["batalhas"]) <= 0:
        return None

    tentativa, caminho = _proxima_tentativa(agente, ciclo)
    delta = treino["wr"] - holdout["wr"]
    ep = _ep_diferenca(treino, holdout)

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CABECALHO)
        w.writerow([
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            _ciclo_seguro(ciclo),
            tentativa,
            etiqueta,
            BASELINE_NAME,
            BASELINE_SPEC_VERSION,
            FORMATO,
            "|".join(AI_PROFILE),
            semente,
            agente.upper(),
            min(treino["batalhas"], holdout["batalhas"]),
            f"{treino['wr']:.2f}",
            f"{holdout['wr']:.2f}",
            f"{delta:+.2f}",
            f"{ep:.2f}",
            f"{treino['margem']:.2f}",
            f"{holdout['margem']:.2f}",
            f"{treino['duracao']:.1f}",
            f"{holdout['duracao']:.1f}",
            treino["empates"],
            holdout["empates"],
            treino["estados_depois"] - treino["estados_antes"],
            holdout["estados_depois"] - holdout["estados_antes"],
            f"{treino['tempo']:.1f}",
            f"{holdout['tempo']:.1f}",
            treino["batalhas"],
            holdout["batalhas"],
            "1",
        ])

    salvar_diagnosticos(caminho, treino, holdout)
    return caminho


# --------------------------------------------------------------------------
# Relatorio
# --------------------------------------------------------------------------
def imprimir(agente, treino, holdout, etiqueta=""):
    delta = treino["wr"] - holdout["wr"]
    ep = _ep_diferenca(treino, holdout)
    desvios = abs(delta) / ep if ep > 0 else 0.0
    largura = 88

    print()
    print("=" * largura)
    print(f"  AVALIACAO EXTERNA — {agente.upper()} vs {BASELINE_NAME} v{BASELINE_SPEC_VERSION}".center(largura))
    print("=" * largura)
    print(f"  Etiqueta: {etiqueta or '(sem etiqueta)'}")
    print(f"  Perfil Cynthia: {' | '.join(AI_PROFILE)}")
    print()
    print(
        f"  {'Condicao':<22} {'WR':>9} {'V-D-E':>15} {'Margem':>9} "
        f"{'Duracao':>10} {'Latencia':>10}"
    )
    print("  " + "-" * (largura - 4))

    for r in (treino, holdout):
        vde = f"{r['vitorias']}-{r['derrotas']}-{r['empates']}"
        print(
            f"  {r['rotulo']:<22} {r['wr']:>8.2f}% {vde:>15} "
            f"{r['margem']:>+9.2f} {r['duracao']:>9.1f}t {r['latencia']:>9.2f}ms"
        )

    print("  " + "-" * (largura - 4))
    print(f"  Delta TREINO - HOLDOUT : {delta:+.2f} pp")
    print(f"  EP da diferenca        : {ep:.2f} pp ({desvios:.1f} EP)")

    cfg = AGENTES[agente]
    if cfg["aprende"]:
        novos_t = treino["estados_depois"] - treino["estados_antes"]
        novos_h = holdout["estados_depois"] - holdout["estados_antes"]
        print(f"  Estados novos (T/H)    : {novos_t} / {novos_h}")
        print("  Aprendizado            : DESLIGADO; cerebro nao e salvo")
    else:
        print("  Aprendizado            : N/A (agente nao-aprendiz)")

    print()
    print("  Leitura: a policy adversaria e a mesma nas duas condicoes. O delta mede a")
    print("  mudanca de desempenho sob troca do pool completo; nao e corrigido pelo antigo")
    print("  vies de +2,35 pp do Instinct, que pertencia a outra regua.")
    print("=" * largura)
    print()


async def executar(agente, n_batalhas, etiqueta, semente, ciclo, diagnostico_cynthia=False):
    if agente not in AGENTES:
        print(f"[ERRO] Agente desconhecido: {agente}")
        return None

    cfg = AGENTES[agente]
    if cfg["brain"] is not None:
        caminho = os.path.join(BRAINS_DIR, cfg["brain"])
        if not os.path.exists(caminho):
            print(f"[ERRO] Cerebro nao encontrado para {agente}: {caminho}")
            return None

    print(
        f"\n[{agente.upper()}] {n_batalhas:,} batalhas por condicao contra "
        f"{BASELINE_NAME} v{BASELINE_SPEC_VERSION}..."
    )

    print(f"  -> TREINO  ({len(TIMES_TREINO)} times)")
    treino = await avaliar_condicao(
        agente,
        TIMES_TREINO,
        "Pool de treino",
        "T",
        n_batalhas,
        semente,
        diagnostico_cynthia=diagnostico_cynthia,
    )

    print(f"  -> HOLDOUT ({len(TIMES_HOLDOUT)} times)")
    holdout = await avaliar_condicao(
        agente,
        TIMES_HOLDOUT,
        "Pool holdout",
        "H",
        n_batalhas,
        semente,
        diagnostico_cynthia=diagnostico_cynthia,
    )

    imprimir(agente, treino, holdout, etiqueta=etiqueta)
    caminho = registar(agente, treino, holdout, etiqueta, semente, ciclo)
    if caminho:
        print(f"  Resultado: {caminho}")
        # Os CSVs ja estao salvos se a biblioteca grafica estiver indisponivel.
        try:
            grafico = gerar_grafico(caminho)
            print(f"  Dashboard: {grafico}")
        except Exception as exc:
            print(f"  [AVISO] CSVs preservados; falha ao gerar graficos: {exc}")
            print("  Regerar: python -m shared.analysis.plot_generalizacao")

    return treino, holdout


async def main():
    ap = argparse.ArgumentParser(
        description=(
            "Avalia MaxDamage/Instinct/Blue/Green/Ash contra CynthiaPlatinumBot "
            "nos pools treino e holdout."
        )
    )
    ap.add_argument("--agente", default=None, choices=list(AGENTES.keys()))
    ap.add_argument(
        "--todos",
        action="store_true",
        help="avalia todas as baselines e agentes disponiveis",
    )
    ap.add_argument(
        "--batalhas",
        type=int,
        default=BATALHAS,
        help=f"batalhas por condicao (default {BATALHAS})",
    )
    ap.add_argument(
        "--ciclo",
        type=str,
        required=True,
        help="versao do Blue que identifica o ciclo experimental, ex.: B11",
    )
    ap.add_argument(
        "--etiqueta",
        type=str,
        default="",
        help="marco/descricao da tentativa, ex.: pre-main, 100k ou final",
    )
    ap.add_argument(
        "--semente",
        type=int,
        default=SEMENTE_PADRAO,
        help=f"seed de times e desempates da Cynthia (default {SEMENTE_PADRAO})",
    )
    ap.add_argument(
        "--log",
        default="critical",
        choices=list(NIVEIS_LOG.keys()),
        help="nivel de log do poke-env; 'error' e util para diagnostico",
    )
    ap.add_argument(
        "--diagnostico-cynthia",
        action="store_true",
        help="faz a Cynthia imprimir erros internos; recomendado apenas em smoke tests",
    )
    args = ap.parse_args()

    if args.batalhas <= 0:
        ap.error("--batalhas deve ser > 0")

    global NIVEL_LOG
    NIVEL_LOG = NIVEIS_LOG[args.log]

    print("=" * 88)
    print("  PROTOCOLO DE AVALIACAO EXTERNA".center(88))
    print("=" * 88)
    print(f"  Ciclo       : {_ciclo_seguro(args.ciclo)}")
    print(f"  Regua       : {BASELINE_NAME} v{BASELINE_SPEC_VERSION}")
    print(f"  Perfil      : {' | '.join(AI_PROFILE)}")
    print(f"  Formato     : {FORMATO}")
    print(f"  Seed        : {args.semente}")
    print(f"  Batalhas    : {args.batalhas:,} por condicao")
    print(f"  Concorrencia: {CONCORRENCIA}")
    print("=" * 88)

    if args.todos:
        alvos = [a for a in ORDEM_TODOS if a in AGENTES]
    else:
        alvos = [args.agente or "blue"]

    resumo = {}
    for agente in alvos:
        r = await executar(
            agente,
            args.batalhas,
            etiqueta=args.etiqueta,
            semente=args.semente,
            ciclo=args.ciclo,
            diagnostico_cynthia=args.diagnostico_cynthia,
        )
        if r:
            resumo[agente] = r

    if len(resumo) > 1:
        print("=" * 88)
        print("  RESUMO COMPARATIVO CONTRA CYNTHIA".center(88))
        print("=" * 88)
        print(f"  {'Agente':<12} {'Treino':>11} {'Holdout':>11} {'Delta':>11} {'EP delta':>11}")
        print("  " + "-" * 84)
        for agente, (treino, holdout) in resumo.items():
            delta = treino["wr"] - holdout["wr"]
            ep = _ep_diferenca(treino, holdout)
            print(
                f"  {agente.upper():<12} {treino['wr']:>10.2f}% "
                f"{holdout['wr']:>10.2f}% {delta:>+10.2f} {ep:>10.2f}"
            )
        print("=" * 88)
        print()


if __name__ == "__main__":
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    else:
        asyncio.run(main())
