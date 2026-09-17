"""
scripts/avaliar_generalizacao.py — teste de generalização (holdout).

PERGUNTA QUE RESPONDE
---------------------
O agente aprendeu a JOGAR, ou apenas decorou os times de treino?

COMO RESPONDE
-------------
Corre o mesmo cérebro, já treinado e com o APRENDIZADO DESLIGADO, em duas condições:

  1. TREINO  — os times que viu durante o treino (shared/env/teams_train.py)
  2. HOLDOUT — times NUNCA vistos (shared/env/teams_eval.py)

A diferença entre os dois Win Rates é a medida de generalização.

ATENÇÃO À LEITURA DA QUEDA
--------------------------
A queda NÃO é toda atribuível ao agente. Na condição de holdout o ADVERSÁRIO
(InstinctBot) também joga com os times novos, e mediu-se que o pool de eval o
favorece em ~2,75 pp (secção 6.24 do ESTADO_DO_PROJETO.md). O viés é sempre no mesmo
sentido: INFLA a queda, nunca a esconde. A queda corrigida é registada no CSV.

Os limiares do veredicto (5 e 15 pp) são CONVENÇÃO ADOTADA, não norma. Com o ciclo
v8/v5, Blue 12,06 e Green 15,78 caíram em categorias opostas apesar de distarem
3,72 pp: o veredicto é um rótulo de leitura rápida, não um resultado. Citar sempre a
queda e a significância, nunca só a etiqueta.

APRENDIZADO DESLIGADO
---------------------
Durante toda a avaliação:
  alpha = 0      → a Q-table não é alterada
  epsilon = 0    → sem jogadas aleatórias; mede a política pura
  replay off     → sem consolidação

O cérebro é carregado, avaliado e NUNCA gravado. O ficheiro .pkl fica intacto.

USO
---
    python -m scripts.avaliar_generalizacao --agente blue --etiqueta "v8-60times"
    python -m scripts.avaliar_generalizacao --agente green --batalhas 5000
    python -m scripts.avaliar_generalizacao --todos --batalhas 5000 --etiqueta "v8-v5"
"""

import argparse
import asyncio
import copy
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env import AccountConfiguration, ServerConfiguration

from instinct.instinct_player import InstinctBot
from qlearning.hybrid_agent import HybridAgent
from qlearning.pure_agent import PureAgent
from shared.env.teams_train import RandomTeamFromPool, TEAMS_LIST as TIMES_TREINO
from shared.env.teams_eval import TEAMS_LIST as TIMES_HOLDOUT
from shared.analysis.plot_generalizacao import registar, gerar_grafico

try:
    from qlearning.ash_agent import AshAgent
except ImportError:
    AshAgent = None

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAINS_DIR = os.path.join(ROOT, "artefatos", "brains")

# CORRIGIDO 25/08/2026. O segundo campo do ServerConfiguration e o URL de
# AUTENTICACAO, e apontava para "https://play.pokemonshowdown.com/action.php?".
# O poke-env pedia o token ao Showdown PUBLICO e apresentava-o ao servidor LOCAL,
# que o rejeitava:
#
#     |nametaken|EvalBlueT|Your authentication token was invalid.
#
# Sintoma inicial mais dificil de ler: as batalhas ja iniciadas terminavam
# normalmente, mas NENHUMA NOVA arrancava, porque cada batalha nova precisa do
# handshake de desafio. O processo ficava pendurado a espera de batalhas que nunca
# comecavam, sem erro visivel. Todos os outros ficheiros do projeto ja usavam o
# endereco local nos dois campos; este era o unico que nao usava.
SERVIDOR = ServerConfiguration("ws://localhost:8000/showdown/websocket",
                               "http://localhost:8000/")
FORMATO = "gen9nationaldex"
BATALHAS = 1_000          # por condição (treino e holdout)
CONCORRENCIA = 1

# --------------------------------------------------------------------------
# NÍVEL DE LOG — configurável por --log (26/08/2026)
# --------------------------------------------------------------------------
# Estava fixo em CRITICAL, e foi isso que escondeu a causa do travamento.
#
# O que se observou: a batalha ficava parada à espera da decisão do agente, o
# servidor NÃO devolvia "[Invalid choice]" (essa é registada a CRITICAL e aparecia),
# e a consola ficava em branco. Ou seja, a jogada nunca chegou a ser enviada.
#
# A leitura: se o choose_move levantar uma exceção, a tarefa que trata aquela batalha
# morre em silêncio. Não há ordem, não há erro do servidor, e o poke-env nunca mais
# toca naquela batalha. Como a tarefa morreu, o contador de batalhas ativas fica
# dessincronizado — o que explica o segundo sintoma: arrancar uma SEGUNDA batalha com
# CONCORRENCIA = 1.
#
# Exceções no poke-env são registadas a ERROR/EXCEPTION, ambos ABAIXO de CRITICAL.
# O nível de log estava a filtrar exatamente a mensagem necessária para diagnosticar.
#
#   --log critical  (default) corrida normal, consola limpa
#   --log error     apanha exceções e tracebacks, sem o protocolo da batalha
#   --log debug     tudo, incluindo o protocolo. Só para corridas curtas.
NIVEIS_LOG = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
NIVEL_LOG = logging.CRITICAL   # sobreposto por --log em main()

AGENTES = {
    "blue": (HybridAgent, "blue_brain.pkl", "EvalBlue"),
    "green": (PureAgent, "green_brain.pkl", "EvalGreen"),
}
if AshAgent is not None:
    AGENTES["ash"] = (AshAgent, "ash_brain.pkl", "EvalAsh")


def _expor_excecoes(jogador, etiqueta):
    """Faz o choose_move gritar em vez de morrer em silêncio.

    ACRESCENTADO 26/08/2026. O poke-env trata a decisão de cada batalha numa tarefa
    própria. Se o choose_move levantar uma exceção, a tarefa morre, nenhuma ordem é
    enviada, e a batalha fica pendurada para sempre — sem erro do servidor e sem nada
    na consola. Foi assim que o travamento se manifestou.

    Este wrapper NÃO altera nenhuma decisão: chama o original e só intercepta a
    exceção para a imprimir com traceback completo antes de a deixar seguir. É
    instrumentação de diagnóstico, não correção de comportamento.
    """
    original = jogador.choose_move

    def com_traceback(battle):
        try:
            return original(battle)
        except Exception:
            print()
            print("!" * 78)
            print(f"  EXCECAO em choose_move de {etiqueta}")
            print(f"  batalha: {getattr(battle, 'battle_tag', '?')} | "
                  f"turno: {getattr(battle, 'turn', '?')}")
            activo = getattr(battle, "active_pokemon", None)
            if activo is not None:
                print(f"  ativo: {getattr(activo, 'species', '?')} | "
                      f"item: {getattr(activo, 'item', '?')} | "
                      f"can_mega: {getattr(battle, 'can_mega_evolve', '?')} | "
                      f"can_z: {getattr(battle, 'can_z_move', '?')}")
            print("!" * 78)
            traceback.print_exc()
            print("!" * 78, flush=True)
            raise

    jogador.choose_move = com_traceback
    return jogador


def construir(agente, times, sufixo):
    """Cria o agente e o oponente para uma condição, com aprendizado DESLIGADO."""
    classe, ficheiro_cerebro, nome = AGENTES[agente]

    jogador = classe(
        account_configuration=AccountConfiguration(f"{nome}{sufixo}", None),
        server_configuration=SERVIDOR,
        battle_format=FORMATO,
        team=RandomTeamFromPool(times),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False,
        log_level=NIVEL_LOG,
        brain_file=os.path.join(BRAINS_DIR, ficheiro_cerebro),
    )

    # ---- CONGELAR O APRENDIZADO ----
    b = jogador.brain
    b.epsilon = 0.0            # política pura, sem exploração
    b.min_epsilon = 0.0
    b.alpha = 0.0              # nenhum update altera a Q-table
    b.min_alpha = 0.0
    b.replay_min_states = 10 ** 12   # gate impossível de atingir: replay nunca corre
    b.use_traces = False       # sem eligibility traces
    if hasattr(b, "usar_encolhimento"):
        b.usar_encolhimento = False

    oponente = InstinctBot(
        account_configuration=AccountConfiguration(f"RefBot{sufixo}", None),
        server_configuration=SERVIDOR,
        battle_format=FORMATO,
        team=RandomTeamFromPool(times),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False,
        log_level=NIVEL_LOG,
    )

    _expor_excecoes(jogador, f"{nome}{sufixo}")
    _expor_excecoes(oponente, f"RefBot{sufixo}")
    return jogador, oponente


async def avaliar_condicao(agente, times, rotulo, sufixo, n_batalhas):
    """Corre n batalhas numa condição e devolve as métricas."""
    jogador, oponente = construir(agente, times, sufixo)

    estados_antes = len(jogador.brain.q_table)
    t0 = time.perf_counter()
    await jogador.battle_against(oponente, n_battles=n_batalhas)
    dt = time.perf_counter() - t0

    terminadas = max(1, jogador.n_finished_battles)
    wr = jogador.n_won_battles / terminadas * 100.0
    m = jogador.pop_block_metrics()
    estados_depois = len(jogador.brain.q_table)

    return {
        "rotulo": rotulo,
        "wr": wr,
        "batalhas": terminadas,
        "margem": m.get("margem_media", 0.0),
        "duracao": m.get("duracao_media", 0.0),
        "ties": m.get("auto_ties", 0),
        "latencia": m.get("latencia_ms", 0.0),
        "estados_antes": estados_antes,
        "estados_depois": estados_depois,
        "tempo": dt,
    }


def imprimir(agente, treino, holdout, etiqueta=""):
    queda = treino["wr"] - holdout["wr"]
    largura = 78
    print()
    print("=" * largura)
    print(f"  GENERALIZAÇÃO — {agente.upper()}".center(largura))
    print("=" * largura)
    print(f"  {'Condição':<20} {'Win Rate':>10} {'Margem':>9} {'Duração':>9} {'Ties':>6}")
    print("  " + "-" * (largura - 4))
    for r in (treino, holdout):
        print(f"  {r['rotulo']:<20} {r['wr']:>9.2f}% {r['margem']:>9.2f} "
              f"{r['duracao']:>8.0f}t {r['ties']:>6}")
    print("  " + "-" * (largura - 4))
    print(f"  {'QUEDA':<20} {queda:>+9.2f} pp")
    print()

    # Significância: erro padrão da diferença entre duas proporções
    n = min(treino["batalhas"], holdout["batalhas"])
    p1, p2 = treino["wr"] / 100.0, holdout["wr"] / 100.0
    se = ((p1 * (1 - p1) + p2 * (1 - p2)) / n) ** 0.5 * 100.0
    desvios = abs(queda) / se if se > 0 else 0.0
    print(f"  Erro padrão da diferença: {se:.2f} pp  ({desvios:.1f} desvios)")
    if desvios < 2:
        print("  A queda NÃO é estatisticamente significativa.")
    print()

    if queda < 5:
        veredicto = "GENERALIZA BEM — aprendeu princípios de jogo, não os times"
    elif queda < 15:
        veredicto = "GENERALIZAÇÃO PARCIAL — há alguma especialização nos times de treino"
    else:
        veredicto = "OVERFITTING — o desempenho depende dos times vistos no treino"
    print(f"  VEREDICTO: {veredicto}")
    print()

    # Prova de que nada foi aprendido durante a avaliação
    # NOTA (25/08/2026): a mensagem anterior dava FALSO ALARME garantido. Visitar um
    # estado cria a entrada na tabela mesmo com alpha=0, e na condicao holdout os
    # estados sao novos POR DEFINICAO. Crescimento aqui e esperado e nao significa
    # aprendizado: com alpha=0 nenhum VALOR pode mudar, e o cerebro nunca e gravado,
    # logo o .pkl fica intacto de qualquer forma.
    novos_t = treino["estados_depois"] - treino["estados_antes"]
    novos_h = holdout["estados_depois"] - holdout["estados_antes"]
    print(f"  Estados novos criados: {novos_t} no treino, {novos_h} no holdout.")
    print("  (Esperado no holdout: sao times nunca vistos. Com alpha=0 nenhum VALOR")
    print("   foi alterado, e o .pkl nao foi gravado.)")
    if novos_t > 0:
        print(f"  Nota: {novos_t} estados novos na condicao de TREINO, nos mesmos")
        print(f"  {len(TIMES_TREINO)} times ja vistos: cobertura incompleta do espaco.")

    # ---- HISTORICO E GRAFICO (28/08/2026) ----
    # Cada teste custa 2 x n batalhas. Sem registo, a serie entre versoes perde-se, e
    # e ela que mostra se as alteracoes ao pool e ao instinto reduzem o overfitting.
    try:
        caminho_csv = registar(agente, treino, holdout, etiqueta=etiqueta)
        png = gerar_grafico()
        print()
        print(f"  Historico: {caminho_csv}")
        print(f"  Grafico  : {png or '(1 so teste: sem serie para desenhar)'}")
    except Exception as e:
        print(f"  AVISO: falha a registar o historico: {e}")
    print("=" * largura)
    print()


async def executar(agente, n_batalhas, etiqueta=""):
    if agente not in AGENTES:
        print(f"[ERRO] Agente desconhecido: {agente}")
        return None
    caminho = os.path.join(BRAINS_DIR, AGENTES[agente][1])
    if not os.path.exists(caminho):
        print(f"[ERRO] Cérebro não encontrado: {caminho}")
        return None

    print(f"\n[{agente.upper()}] a avaliar {n_batalhas:,} batalhas por condição "
          f"(aprendizado desligado)...")

    print(f"  → condição TREINO  ({len(TIMES_TREINO)} times conhecidos)")
    treino = await avaliar_condicao(agente, TIMES_TREINO, "Times de treino", "T", n_batalhas)

    print(f"  → condição HOLDOUT ({len(TIMES_HOLDOUT)} times nunca vistos)")
    holdout = await avaliar_condicao(agente, TIMES_HOLDOUT, "Times NOVOS (holdout)", "H", n_batalhas)

    imprimir(agente, treino, holdout, etiqueta=etiqueta)
    return treino, holdout


async def main():
    ap = argparse.ArgumentParser(description="Teste de generalização (holdout)")
    ap.add_argument("--agente", default=None, choices=list(AGENTES.keys()))
    ap.add_argument("--todos", action="store_true", help="avalia todos os agentes")
    ap.add_argument("--batalhas", type=int, default=BATALHAS,
                    help=f"batalhas por condição (default {BATALHAS})")
    ap.add_argument("--etiqueta", type=str, default="",
                    help="identifica o ESTADO DO CODIGO desta avaliacao no historico, "
                         "ex: v8-v5-60times. E a chave da serie: sem ela, daqui a tres "
                         "versoes nao se sabe que medicao corresponde a que cerebro.")
    ap.add_argument("--log", default="critical", choices=list(NIVEIS_LOG.keys()),
                    help="nivel de log do poke-env. Use 'error' para apanhar excecoes "
                         "e tracebacks, 'debug' para o protocolo todo (so em corridas "
                         "curtas). Default: critical")
    args = ap.parse_args()

    global NIVEL_LOG
    NIVEL_LOG = NIVEIS_LOG[args.log]
    if args.log != "critical":
        print(f"[LOG] nivel do poke-env: {args.log.upper()}")

    alvos = list(AGENTES.keys()) if args.todos else [args.agente or "blue"]

    resumo = {}
    for a in alvos:
        r = await executar(a, args.batalhas, etiqueta=args.etiqueta)
        if r:
            resumo[a] = r

    if len(resumo) > 1:
        print("=" * 78)
        print("  RESUMO COMPARATIVO".center(78))
        print("=" * 78)
        print(f"  {'Agente':<10} {'Treino':>10} {'Holdout':>10} {'Queda':>10}")
        print("  " + "-" * 74)
        for a, (t, h) in resumo.items():
            print(f"  {a.upper():<10} {t['wr']:>9.2f}% {h['wr']:>9.2f}% {t['wr']-h['wr']:>+9.2f}pp")
        if len(resumo) == 2:
            (a1, (t1, h1)), (a2, (t2, h2)) = list(resumo.items())
            n = min(t1["batalhas"], h1["batalhas"], t2["batalhas"], h2["batalhas"])
            p1, p2 = h1["wr"] / 100.0, h2["wr"] / 100.0
            se = ((p1 * (1 - p1) + p2 * (1 - p2)) / max(1, n)) ** 0.5 * 100.0
            d = h1["wr"] - h2["wr"]
            print("  " + "-" * 74)
            print(f"  No HOLDOUT: {a1.upper()} - {a2.upper()} = {d:+.2f} pp  "
                  f"(EP {se:.2f} pp, {abs(d)/se if se else 0:.1f} desvios)")
            print("  E esta a comparacao que responde a pergunta da tese: qual dos dois")
            print("  se aguenta melhor em material que nunca viu.")
        print("=" * 78)
        print()


if __name__ == "__main__":
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
