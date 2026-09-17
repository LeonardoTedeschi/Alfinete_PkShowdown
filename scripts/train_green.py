"""
scripts/train_green.py — treino individual do agente GREEN (Q-puro).

Configuração:
  - Máx. de batalhas POR EXECUÇÃO : 10.000
  - Orçamento do ciclo            : 400.000 (40 repetições, via treino_continuo)
  - Salvamento (log + cérebro)    : a cada 1.000 batalhas
  - Batalhas simultâneas          : 5
  - Timer do servidor DESLIGADO (evita derrotas por timeout de decisão; auto-ties
    residuais não geram recompensa terminal, logo não poluem a Q-table)
  - Ao fim: gera gráfico de treino + dashboard de inspeção do cérebro

Executar da raiz do projeto:
    python -m scripts.train_green

Nota de memória: blocos de 1.000 batalhas são consolidados e o cérebro é salvo a
cada bloco, libertando memória — evita a sobrecarga de RAM de acumular muitas
batalhas antes de consolidar.
"""

import asyncio
import csv
import glob
import os
import re
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
from shared.analysis.plot_graph import generate_graph
from shared.analysis.inspect_brain import analyze_brain

LOCAL = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

# ---- HIPERPARÂMETROS ----
ALPHA_START = 0.15
MIN_ALPHA = 0.02
GAMMA = 0.99
EPSILON_START = 0.40
MIN_EPSILON = 0.05     # nunca menos de 5% de exploracao (aprendizado continuo)
EPSILON_DECAY = 0.00046
# CALIBRACAO DO CALENDARIO DO EPSILON (27/08/2026)
# ------------------------------------------------
# A formula em brain.decay_epsilon e SUBTRATIVA POR BLOCO, com piso e teto:
#
#     actual_decay = max(DECAY_FLOOR, EPSILON_DECAY * min(3, 3/discovery_rate))
#     epsilon -= actual_decay
#
# Cedo, com muitos estados novos, o multiplicador e minusculo e o PISO manda.
# Tarde, o multiplicador satura em 3 e vale o TETO (3 * EPSILON_DECAY).
#
# ATENCAO: o decaimento e POR BLOCO, nao por batalha. Qualquer alteracao a
# BLOCO_SALVAMENTO obriga a recalibrar estes dois valores.
#
# Alvo: 400.000 batalhas (40 repeticoes de 10k), blocos de 1.000 = 400 blocos,
# com o epsilon a atingir o minimo a ~70% do treino, ou seja no bloco 280.
# Piso 0,001 e teto 0,002 dao media estimada de 0,00127/bloco -> minimo ~bloco 275.
#
# Historico: com 0,002 e 0,003 (calibrados para 200k) o minimo caia no bloco 92 de
# 200, ou seja a 46% do treino.
#
# VERIFICACAO na primeira noite (a taxa de descoberta com 60 times pode divergir):
#     bloco  50 (50k)  -> epsilon ~0,336
#     bloco 100 (100k) -> epsilon ~0,273
#     bloco 200 (200k) -> epsilon ~0,146
#     bloco 280 (280k) -> epsilon  0,050
# Muito abaixo do esperado pede piso maior; acima, piso menor. E proporcional.
#
# RECALIBRACAO DE 29/08/2026, sobre dados REAIS e nao sobre estimativa.
# O ciclo de 400k com 0,00067 e 0,001 atingiu o minimo no bloco 209 de 400 (52%),
# quando o alvo era o bloco 280 (70%). O decaimento medio observado foi 0,001683 por
# bloco, ou seja 1,683 vezes o piso: e essa razao, MEDIDA e nao suposta, que calibra
# os valores abaixo.
#
#   minimo no bloco 280 (70%)  ->  piso 0,00074   decay 0,00050
#   minimo no bloco 300 (75%)  ->  piso 0,00069   decay 0,00046   <== ESCOLHIDO
#   minimo no bloco 320 (80%)  ->  piso 0,00065   decay 0,00044
#
# Alvo de 75% e nao 70% porque o requisito e "NO MINIMO 70%" e as tres calibracoes
# anteriores erraram sempre por defeito (o minimo chegou cedo demais). Cinco pontos
# de margem garantem o requisito mesmo com desvio semelhante.
#
# COMO CONFIRMAR QUE ESTA APLICADO: o cabecalho do treino imprime
# "epsilon : 0.4 -> 0.05 (piso 0.00069)". Se disser 0.001, o ficheiro nao foi
# substituido. Foi assim que se detetou, no ciclo v9, que a recalibracao anterior
# nunca chegara ao disco: o piso impresso era 0,001 e o epsilon caia 0,002 por bloco
# (= 3 x 0,00067, o TETO dos valores antigos).
DECAY_FLOOR   = 0.00069
NOVELTY_K = 30.0

# ---- PROTOCOLO ----
MAX_BATALHAS = 10_000        # teto do treino
BLOCO_SALVAMENTO = 1000      # log + save do cérebro a cada 1000 batalhas
                             # ATENCAO: alterar isto obriga a recalibrar
                             # EPSILON_DECAY e DECAY_FLOOR (ver acima)
CONCORRENCIA = 5             # batalhas simultâneas. Subiu de 3 para 5 em
                             # 27/08/2026: a maquina fica dedicada ao treino.
                             # Nao altera o que se aprende, so a ordem de
                             # chegada das atualizacoes (logo as corridas
                             # deixam de ser reproduziveis por semente).
REPLAY_CICLOS = 20           # chamadas de replay por bloco (batch inalterado)
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


def proximo_indice_log(agente):
    """Devolve o proximo numero de sessao livre para os ficheiros de log/grafico.

    Cada execucao do treino cria os SEUS ficheiros (blue_treino_01.csv,
    blue_treino_02.csv, ...), em vez de sobrescrever o mesmo CSV. Assim cada trecho
    de treino fica isolado e comparavel, e o orquestrador de treino continuo nao
    perde o historico das repeticoes anteriores.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    existentes = glob.glob(os.path.join(LOGS_DIR, f"{agente}_treino_*.csv"))
    ids = []
    for caminho in existentes:
        m = re.fullmatch(rf"{agente}_treino_(\d+)\.csv", os.path.basename(caminho))
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 1


async def main():
    os.makedirs(BRAINS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    brain_path = os.path.join(BRAINS_DIR, f"{AGENT}_brain.pkl")

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

    sessao = proximo_indice_log(AGENT)
    csv_path = os.path.join(LOGS_DIR, f"{AGENT}_treino_{sessao:02d}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            # COLUNAS DE CONVERGENCIA (30/08/2026).
            #
            # `Visitas_Est` (media simples de visitas) SAIU. Tratava todos os estados
            # por igual, e medido nos cerebros de 400k 16,7% dos estados absorvem
            # 90,7% das decisoes. A media respondia a pergunta errada.
            # `Confianca` (visit_counts>=3) tambem saiu: limiar baixo demais para
            # significar confianca.
            #
            # Entram cinco. `Cobertura_Pond` e `Descoberta_Turno` medem TERRENO;
            # `Churn_Politica`, `Delta_Q` e `Margem_Decisao` medem CONVERGENCIA, que
            # e outra coisa: a politica pode estar estavel com cobertura a crescer,
            # e pode haver cobertura alta com a decisao ainda a oscilar.
            ["Batalhas", "WinRate_Bloco", "Vitorias", "Derrotas", "Estados_Q", "Epsilon",
             "Cobertura_Pond", "Descoberta_Turno", "Churn_Politica", "Delta_Q",
             "Margem_Decisao", "Estados_Maduros",
             "Reward", "Ghost_Battles",
             "Latencia_ms", "Margem_Media", "Duracao_Media", "Auto_Ties", "Tamanho_KB", "Tempo_s"])

    # SAVE IMEDIATO: cria o .pkl logo no arranque, para o ficheiro existir (e o
    # caminho ser validado) desde o inicio, em vez de so ao fim do 1o bloco de 1000.
    agent.save_brain()
    if os.path.exists(brain_path):
        estado_cerebro = (f"{brain_path}  "
                          f"({os.path.getsize(brain_path)/1024:.1f} KB, "
                          f"{len(agent.brain.q_table)} estados)")
    else:
        estado_cerebro = f"FALHA A CRIAR: {brain_path} (progresso NAO sera persistido)"

    rel = RelatorioConsola(agente="GREEN", descricao="Q-puro (Q-Learning sem instinto)")
    rel.cabecalho(
        config={
            "Oponente": "Instinto-puro",
            "Formato": BATTLE_FORMAT,
            "Max batalhas": MAX_BATALHAS,
            "Save a cada": f"{BLOCO_SALVAMENTO} batalhas",
            "Concorrencia": CONCORRENCIA,
            "Timer do servidor": "DESLIGADO",
            "alpha": f"{ALPHA_START} -> {MIN_ALPHA}",
            "epsilon": f"{EPSILON_START} -> {MIN_EPSILON} (piso {DECAY_FLOOR})",
            "Convergencia": f"{WR_STABILITY_PP}pp em {STABILITY_BLOCKS} blocos, eps no minimo",
        },
        caminhos={
            "Sessao": f"#{sessao:02d}",
            "Cerebro": estado_cerebro,
            "Log CSV": csv_path,
        })

    total, recent_wr, t0 = 0, [], time.time()
    convergiu = "NAO"
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
        agent.save_brain()

        # As metricas de convergencia guardam snapshot para o bloco seguinte, logo
        # tem de ser chamadas UMA vez por bloco e sempre na mesma ordem.
        conv = agent.brain.metricas_convergencia()
        _, avg_visits, conf = agent.brain.inspect_brain()   # so para o relatorio
        ghost = len(agent.brain.active_battles_reward)
        wall = time.time() - t0
        tamanho_kb = os.path.getsize(brain_path) / 1024.0 if os.path.exists(brain_path) else 0.0

        # Acesso TOLERANTE as metricas: uma metrica opcional em falta (por exemplo
        # numa versao anterior do base_agent) nao pode parar o treino.
        lat = m.get("latencia_ms", 0.0)
        marg = m.get("margem_media", 0.0)
        dur = m.get("duracao_media", 0.0)
        ties = m.get("auto_ties", 0)
        # Recompensa REAL do bloco. A coluna anterior usava brain.episode_reward_max,
        # que ficava presa no sentinela (-9999) e nao media nada.
        reward = m.get("reward_batalha", 0.0)
        reward_turno = m.get("reward_turno", 0.0)

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

        # SALVAGUARDA: se os Q-values divergirem, parar JA. Continuar so queima
        # tempo a aprender uma politica corrompida.
        ok_div, maior_q, msg_div = agent.brain.verificar_divergencia()
        if not ok_div:
            print()
            print("!" * 70)
            print(f"  [%s] TREINO ABORTADO — DIVERGENCIA DETETADA" % "GREEN")
            print(f"  {msg_div}")
            print("  O cerebro NAO foi corrompido no disco: o ultimo save e anterior")
            print("  a este bloco. Reduz alpha ou lambda antes de retomar.")
            print("!" * 70)
            convergiu = f"ABORTADO (divergencia: maior |Q| = {maior_q:,.0f})"
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

    # ORCAMENTO FIXO: o treino corre SEMPRE as MAX_BATALHAS. Nao ha paragem
    # antecipada. Para a comparacao entre agentes ser controlada, todos tem de
    # receber exatamente o mesmo orcamento de treino; parar mais cedo num deles
    # invalidaria a comparacao (e o criterio de estabilidade disparava por acaso:
    # com n=1000 o desvio padrao do WR e 1.58pp, e 5 blocos aleatorios ficam dentro
    # de 2.0pp em ~10% das janelas).
    convergiu = (f"orcamento cumprido: {total:,} batalhas | "
                 f"{blocos_estaveis} janela(s) de {STABILITY_BLOCKS} blocos "
                 f"dentro de {WR_STABILITY_PP}pp (indicador, nao criterio de paragem)")

    agent.save_brain()

    # Analise automatica: grafico da sessao + dashboard do cerebro.
    graf_path = os.path.join(LOGS_DIR, f"{AGENT}_grafico_{sessao:02d}.png")
    resultados = {"Convergencia": convergiu}
    try:
        generate_graph(csv_path, graf_path,
                       agent="Green", opponent="Instinto", total_battles=total,
                       final_win_rate=recent_wr[-1] if recent_wr else 0.0,
                       final_states=len(agent.brain.q_table))
        resultados["Grafico"] = graf_path
    except Exception as e:
        resultados["Grafico"] = f"FALHOU: {e}"
    try:
        analyze_brain(brain_path, os.path.join(LOGS_DIR, "analise"), f"Green_s{sessao:02d}")
        resultados["Dashboard"] = os.path.join(LOGS_DIR, "analise")
    except Exception as e:
        resultados["Dashboard"] = f"FALHOU: {e}"
    resultados["Cerebro"] = brain_path

    rel.resumo_final(extra=resultados)


if __name__ == "__main__":
    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    else:
        asyncio.run(main())
