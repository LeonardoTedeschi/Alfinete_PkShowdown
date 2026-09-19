"""
scripts/train_green.py — treino individual do agente GREEN.

Configuracao:
  - Max. de batalhas POR EXECUCAO : 10.000
  - Orcamento do ciclo            : definido por treino_continuo (candidato v11/v8: 650.000)
  - Bloco de aprendizagem/log     : 1.000 batalhas
  - Persistencia do cerebro       : emergencia em 5.000; oficial em 10.000
  - Batalhas simultaneas          : 5
  - Timer do servidor DESLIGADO
  - Graficos e dashboards         : centralizados no treino_continuo a cada 50.000

Executar da raiz do projeto:
    python -m scripts.train_green

IMPORTANTE: o cerebro permanece em RAM durante toda a execucao de 10k. Salvar menos
vezes nao aumenta a memoria permanente; apenas reduz serializacao/I/O. Se uma sessao
for interrompida, os ficheiros *_brain_emergency.pkl e *_treino_emergencia.csv sao
preservados para diagnostico/recuperacao e uma nova sessao nao os sobrescreve.
"""

import asyncio
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poke_env import AccountConfiguration, ServerConfiguration

import logging

from instinct.instinct_player import InstinctBot
from qlearning.pure_agent import PureAgent
from shared.env.teams_train import RandomTeamFromPool, TEAMS_LIST
from shared.console_report import RelatorioConsola

LOCAL = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

# ---- HIPERPARÂMETROS ----
ALPHA_START = 0.15
MIN_ALPHA = 0.02
GAMMA = 0.99
EPSILON_START = 0.40
MIN_EPSILON = 0.05     # nunca menos de 5% de exploracao (aprendizado continuo)
EPSILON_DECAY = 0.00039
# CALIBRACAO DO CALENDARIO DO EPSILON (19/09/2026)
# ------------------------------------------------
# Unica alteracao ALGORITMICA deste ciclo. EPSILON_START e MIN_EPSILON permanecem
# iguais; apenas se prolonga a fase de exploracao para a representacao 16D.
#
# A formula em brain.decay_epsilon continua SUBTRATIVA POR bloco de 1.000:
#
#     actual_decay = max(DECAY_FLOOR,
#                        EPSILON_DECAY * min(3, 3/discovery_rate))
#     epsilon -= actual_decay
#
# Dados do v10/v7: 0.00046 / 0.00069 levaram epsilon ao piso por volta de
# 475k-480k. A nova calibracao 0.00039 / 0.00058 projeta, usando a curva real de
# descoberta anterior, o piso aproximadamente em 540k-555k. Num orcamento candidato
# de 650k isso deixa ~100k para consolidacao em epsilon=0.05.
#
# O bloco de aprendizagem continua em 1.000 batalhas; mudar a frequencia de SAVE nao
# altera esta calibracao, porque decay/replay/metricas continuam a cada 1.000.
DECAY_FLOOR = 0.00058
NOVELTY_K = 30.0

# ---- PROTOCOLO ----
MAX_BATALHAS = 10_000        # uma sessao/processo
BLOCO_SALVAMENTO = 1000      # nome historico: agora e bloco de treino/log/decay/replay
SAVE_EMERGENCIA = 5_000      # snapshot recuperavel no meio da sessao
SAVE_OFICIAL = 10_000        # persistencia oficial ao concluir a sessao
CONCORRENCIA = 5             # mantida para nao introduzir outra variavel neste ciclo
REPLAY_CICLOS = 20           # inalterado
BATTLE_FORMAT = "gen9nationaldex"
AGENT = "green"
# Ancorados à RAIZ do projeto (ROOT), não ao CWD — evita criar artefatos/ em
# scripts/ quando executado de lá. Sempre em Bot-QV-Pokemon/artefatos/.
BRAINS_DIR = os.path.join(ROOT, "artefatos", "brains")
LOGS_DIR = os.path.join(ROOT, "artefatos", "logs")

# Convergência: só quando epsilon no mínimo E WR estável.
# Paragem antecipada por convergencia DESLIGADA por omissao.
# Motivo: com o cerebro ja carregado com epsilon no minimo, o criterio pode
# disparar logo ao 5o bloco, cortando a corrida a 5k em vez de 10k e tornando os
# logs inconsistentes entre sessoes. Julgar convergencia sobre 5 blocos (5000
# batalhas) e estatisticamente fragil: com ruido puro a amplitude fica <= 2.0pp
# em ~10% das janelas, ou seja 1 em cada 10 corridas seria cortada ao acaso.
# A convergencia passa a ser avaliada pelo ORQUESTRADOR, entre repeticoes, onde

WR_STABILITY_PP = 2.0
STABILITY_BLOCKS = 5



def _salvar_brain_verificado(agent, destino):
    """Salva por staging e so substitui o destino se a serializacao terminou."""
    staging = destino + ".stage"
    for p in (staging, staging + ".tmp"):
        if os.path.exists(p):
            os.remove(p)
    agent.brain.save_model(staging)
    if not os.path.exists(staging) or os.path.getsize(staging) == 0:
        raise RuntimeError(f"Falha ao persistir cerebro em {destino}")
    os.replace(staging, destino)


def _tamanho_ultimo_save_kb(brain_path, emergency_path):
    caminho = emergency_path if os.path.exists(emergency_path) else brain_path
    return os.path.getsize(caminho) / 1024.0 if os.path.exists(caminho) else 0.0


async def main():
    os.makedirs(BRAINS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    brain_path = os.path.join(BRAINS_DIR, f"{AGENT}_brain.pkl")
    emergency_brain_path = os.path.join(BRAINS_DIR, f"{AGENT}_brain_emergency.pkl")
    csv_path = os.path.join(LOGS_DIR, f"{AGENT}_treino_emergencia.csv")

    # Nunca sobrescrever uma sessao interrompida. O orquestrador remove estes
    # ficheiros apenas depois de uma sessao completa ser consolidada.
    restos = [p for p in (emergency_brain_path, csv_path) if os.path.exists(p)]
    if restos:
        print("[TREINO] RECUSADO: existe recuperacao pendente de sessao anterior:")
        for p in restos:
            print(f"         {p}")
        print("         Preserve/recupere esses ficheiros antes de iniciar nova sessao.")
        return False

    agent = PureAgent(
        account_configuration=AccountConfiguration("GreenTrain", None),
        server_configuration=LOCAL, battle_format=BATTLE_FORMAT,
        team=RandomTeamFromPool(TEAMS_LIST),
        alpha=ALPHA_START, gamma=GAMMA, epsilon=EPSILON_START,
        min_epsilon=MIN_EPSILON, decay=EPSILON_DECAY,
        brain_file=brain_path,
        max_concurrent_battles=CONCORRENCIA,
        # timer DESLIGADO: evita derrotas por timeout de decisao contaminarem os dados
        start_timer_on_battle_start=False,
        # log_level e o parametro NATIVO do poke-env para o nivel de log deste jogador.
        # CRITICAL cala os WARNING ruidosos (o "You will auto-tie...", habilidades nao
        # mapeadas, popups de desafio) MAS mantem visiveis os erros CRITICAL, como o
        # "[Invalid choice]" do servidor, que sao os que interessam mesmo.
        # Nao mexemos na configuracao global de logging: foi isso que antes fez o
        # terminal ser inundado com todo o protocolo da batalha.
        log_level=logging.CRITICAL,
    )
    agent.brain.initial_alpha = ALPHA_START
    agent.brain.min_alpha = MIN_ALPHA
    agent.brain.alpha = ALPHA_START
    agent.brain.novelty_k = NOVELTY_K
    agent.brain.decay_floor = DECAY_FLOOR

    opponent = InstinctBot(
        account_configuration=AccountConfiguration("InstinctRef", None),
        server_configuration=LOCAL, battle_format=BATTLE_FORMAT,
        team=RandomTeamFromPool(TEAMS_LIST),
        max_concurrent_battles=CONCORRENCIA,
        # timer DESLIGADO: evita derrotas por timeout de decisao contaminarem os dados
        start_timer_on_battle_start=False,
        # log_level e o parametro NATIVO do poke-env para o nivel de log deste jogador.
        # CRITICAL cala os WARNING ruidosos (o "You will auto-tie...", habilidades nao
        # mapeadas, popups de desafio) MAS mantem visiveis os erros CRITICAL, como o
        # "[Invalid choice]" do servidor, que sao os que interessam mesmo.
        # Nao mexemos na configuracao global de logging: foi isso que antes fez o
        # terminal ser inundado com todo o protocolo da batalha.
        log_level=logging.CRITICAL,
    )

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Batalhas", "WinRate_Bloco", "Vitorias", "Derrotas", "Estados_Q", "Epsilon",
             "Cobertura_Pond", "Descoberta_Turno", "Churn_Politica", "Delta_Q",
             "Margem_Decisao", "Estados_Maduros", "Reward", "Ghost_Battles",
             "Latencia_ms", "Margem_Media", "Duracao_Media", "Auto_Ties", "Tamanho_KB", "Tempo_s"])

    if os.path.exists(brain_path):
        estado_cerebro = (f"{brain_path}  "
                          f"({os.path.getsize(brain_path)/1024:.1f} KB, "
                          f"{len(agent.brain.q_table)} estados)")
    else:
        estado_cerebro = (f"novo em RAM (primeiro save de emergencia em {SAVE_EMERGENCIA:,}; "
                          f"oficial em {SAVE_OFICIAL:,})")

    rel = RelatorioConsola(agente="GREEN", descricao="Q-puro (Q-Learning sem instinto)")
    rel.cabecalho(
        config={
            "Oponente": "Instinto-puro",
            "Formato": BATTLE_FORMAT,
            "Max batalhas": MAX_BATALHAS,
            "Bloco treino/log": f"{BLOCO_SALVAMENTO} batalhas",
            "Save cerebro": f"emergencia {SAVE_EMERGENCIA:,} / oficial {SAVE_OFICIAL:,}",
            "Concorrencia": CONCORRENCIA,
            "Timer do servidor": "DESLIGADO",
            "alpha": f"{ALPHA_START} -> {MIN_ALPHA}",
            "epsilon": f"{EPSILON_START} -> {MIN_EPSILON} (piso decay {DECAY_FLOOR})",
            "Convergencia": f"{WR_STABILITY_PP}pp em {STABILITY_BLOCKS} blocos, eps no minimo",
        },
        caminhos={
            "Cerebro": estado_cerebro,
            "CSV emergencia": csv_path,
        })

    total, recent_wr, t0 = 0, [], time.time()
    convergiu = "NAO"
    abortado = False
    blocos_estaveis = 0
    while total < MAX_BATALHAS:
        prev_states = len(agent.brain.q_table)
        await agent.battle_against(opponent, n_battles=BLOCO_SALVAMENTO)
        total += BLOCO_SALVAMENTO

        won = agent.n_won_battles
        finished = agent.n_finished_battles
        wr = won / max(1, finished) * 100.0
        derrotas = finished - won   # auto-tie residual não conta como derrota real
        # Métricas do bloco (latência, margem, duração) e tamanho do modelo.
        m = agent.pop_block_metrics()
        agent.reset_battles()
        # REPLAY reforcado: 20 chamadas por bloco (batch inalterado). Antes era 1
        # chamada, o que dava ~1.3% das atualizacoes da Q-table; com 20 sobe para
        # ~21%. Suficiente para reaproveitar a experiencia sem que o replay domine
        # o aprendizado (o alvo bootstrap melhora entre chamadas, logo cada
        # repeticao injeta informacao nova, nao apenas repete a mesma conta).
        for _ in range(REPLAY_CICLOS):
            agent.replay()
        new_states = len(agent.brain.q_table) - prev_states
        agent.brain.decay_epsilon(new_states=new_states, battles_in_block=BLOCO_SALVAMENTO)

        # Metricas de convergencia continuam UMA vez por bloco de 1k. A frequencia
        # de persistencia nao altera replay, decay, reward ou atualizacoes Q.
        conv = agent.brain.metricas_convergencia()
        _, avg_visits, conf = agent.brain.inspect_brain()
        ghost = len(agent.brain.active_battles_reward)

        lat = m.get("latencia_ms", 0.0)
        marg = m.get("margem_media", 0.0)
        dur = m.get("duracao_media", 0.0)
        ties = m.get("auto_ties", 0)
        reward = m.get("reward_batalha", 0.0)

        # Verificar divergencia ANTES de qualquer save. Assim um bloco doente nunca
        # substitui o ultimo checkpoint saudavel.
        ok_div, maior_q, msg_div = agent.brain.verificar_divergencia()

        if ok_div:
            if total == SAVE_EMERGENCIA:
                _salvar_brain_verificado(agent, emergency_brain_path)
                print(f"[{AGENT.upper()}] checkpoint de emergencia salvo em {total:,} batalhas")
            elif total == SAVE_OFICIAL:
                _salvar_brain_verificado(agent, brain_path)
                if os.path.exists(emergency_brain_path):
                    os.remove(emergency_brain_path)
                print(f"[{AGENT.upper()}] cerebro oficial salvo em {total:,} batalhas")

        tamanho_kb = _tamanho_ultimo_save_kb(brain_path, emergency_brain_path)
        wall = time.time() - t0

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [total, f"{wr:.2f}", won, derrotas, len(agent.brain.q_table),
                 f"{agent.brain.epsilon:.4f}",
                 f"{conv['cobertura_ponderada']:.2f}",
                 f"{conv['descoberta_por_turno']:.6f}",
                 f"{conv['churn_politica']:.3f}",
                 f"{conv['delta_q']:.4f}",
                 f"{conv['margem_decisao']:.1f}",
                 conv['estados_maduros'],
                 f"{reward:.0f}", ghost, f"{lat:.2f}", f"{marg:.2f}", f"{dur:.1f}",
                 ties, f"{tamanho_kb:.0f}", f"{wall:.0f}"])

        rel.bloco(batalhas=total, metricas={
            "win_rate": wr, "estados": len(agent.brain.q_table),
            "epsilon": agent.brain.epsilon,
            "cobertura_pond": conv["cobertura_ponderada"],
            "descoberta_turno": conv["descoberta_por_turno"],
            "churn_politica": conv["churn_politica"],
            "delta_q": conv["delta_q"],
            "margem_decisao": conv["margem_decisao"],
            "visitas": avg_visits, "confianca": conf,
            "latencia_ms": lat, "margem_media": marg, "duracao_media": dur,
            "auto_ties": ties, "tempo_s": wall, "reward": reward,
        })

        if not ok_div:
            print()
            print("!" * 70)
            print(f"  [GREEN] TREINO ABORTADO — DIVERGENCIA DETETADA")
            print(f"  {msg_div}")
            print("  Nenhum bloco divergente foi salvo no brain oficial/emergencia.")
            print("!" * 70)
            convergiu = f"ABORTADO (divergencia: maior |Q| = {maior_q:,.0f})"
            abortado = True
            break

        recent_wr.append(wr)
        # ESTABILIDADE: contada, mas por omissao NAO interrompe o treino.
        #
        # Com o cerebro a continuar de sessoes anteriores, o epsilon ja carrega no
        # minimo desde o bloco 1, logo essa condicao esta SEMPRE satisfeita e so
        # falta o acaso do ruido. Com n=1000 o desvio padrao do WR e 1.58pp, e 5
        # blocos puramente aleatorios ficam dentro de 2.0pp de amplitude em ~10% das
        # janelas. O treino terminava aos 5000 por sorte estatistica, produzindo
        # logs de tamanhos diferentes e inconsistentes.
        #
        # Convergencia e propriedade do treino INTEIRO, nao de um bloco de 10k:
        # quem a deve avaliar e o orquestrador, sobre varias sessoes.
        epsilon_no_minimo = agent.brain.epsilon <= (agent.brain.min_epsilon + 1e-6)
        if epsilon_no_minimo and len(recent_wr) >= STABILITY_BLOCKS:
            w = recent_wr[-STABILITY_BLOCKS:]
            if max(w) - min(w) <= WR_STABILITY_PP:
                blocos_estaveis += 1

    if abortado:
        resultados = {
            "Convergencia": convergiu,
            "Cerebro oficial": brain_path,
            "Cerebro emergencia": emergency_brain_path if os.path.exists(emergency_brain_path) else "(nao criado)",
            "CSV emergencia": csv_path,
        }
        rel.resumo_final(extra=resultados)
        return False

    # Orcamento fixo da SESSAO cumprido. O brain oficial ja foi salvo no bloco 10k.
    convergiu = (f"orcamento cumprido: {total:,} batalhas | "
                 f"{blocos_estaveis} janela(s) de {STABILITY_BLOCKS} blocos "
                 f"dentro de {WR_STABILITY_PP}pp (indicador, nao criterio de paragem)")

    if not os.path.exists(brain_path) or os.path.getsize(brain_path) == 0:
        raise RuntimeError("Sessao terminou sem brain oficial persistido.")

    resultados = {
        "Convergencia": convergiu,
        "Cerebro": brain_path,
        "CSV emergencia": csv_path,
        "Analise": "centralizada no treino_continuo; marco a cada 50k",
    }
    rel.resumo_final(extra=resultados)
    return True


if __name__ == "__main__":
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        ok = loop.run_until_complete(main())
    else:
        ok = asyncio.run(main())
    if not ok:
        raise SystemExit(2)
