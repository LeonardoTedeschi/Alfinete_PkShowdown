"""
BlueBrain — o motor de Q-Learning tabular do projeto ALFINETE.

Partilhado pelos agentes Green (Q-puro) e Blue (Híbrido). O que os distingue não é o
cérebro, é COMO o alimentam: o Green passa todas as ações legais e ignora o ranking
do instinto; o Blue passa o ranking do instinto como prior de exploração.

CORREÇÕES aplicadas nesta versão (decididas ao longo do projeto):
  (1) replay_experience: REMOVIDO o corpo duplicado que reprocessava o batch e
      injetava ruído (o método rodava os updates duas vezes).
  (2) DPRS dinamico REMOVIDO do PBRS. A funcao historica de novidade
      +k/sqrt(visits(state)) permanece no ficheiro apenas para compatibilidade, mas
      nao possui chamadores no treino actual e NAO entra em reward/PBRS/policy.
  (3) DESFIBRILADOR removido: apply_epsilon_shock_if_stagnant era um no-op preso numa
      docstring; retirado para simplificar a atribuição de causa nos experimentos.
  (4) TIPO uniformizado: estados inicializam sempre com np.zeros (antes decide_action
      usava lista e _apply_q_update usava np.zeros — coexistência frágil).
"""

import numpy as np
import pickle
import os
import random
import threading
from collections import deque


class BlueBrain:
    def __init__(self, alpha=0.2, gamma=0.99, epsilon=0.40, min_epsilon=0.05, decay=0.005,
                 novelty_k=30.0, decay_floor=0.01,
                 memory_maxlen=400000, elite_maxlen=20000, elite_quota=0.10):
        self.initial_alpha = alpha
        self.min_alpha = 0.005
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        # Piso de decaimento por bloco (ver decay_epsilon). Baixo, para no início o
        # epsilon cair de forma constante mesmo com muita descoberta de estados.
        self.decay_floor = decay_floor
        # Gate de conhecimento: nº mínimo de estados na Q-table antes de o replay
        # começar a correr (evita consolidar ruído numa tabela quase vazia).
        self.replay_min_states = 2000
        # --- ELIGIBILITY TRACES (replacing, Watkins Q-lambda com corte off-policy) ---
        # lambda controla o DECAIMENTO do credito, nao um numero fixo de turnos:
        # o horizonte efetivo e ~1/(1-lambda). Com o corte por exploracao (epsilon),
        # o horizonte REAL fica ~1/(1 - lambda*(1-epsilon)).
        #   lambda=0.933 -> horizonte nominal ~15 t, real ~8.8 t (com eps=0.05)
        #   lambda=0.967 -> horizonte nominal ~30 t, real ~12.3 t, mas VARIANCIA 2x
        # Escolhemos 0.933: fica no joelho da curva (ganho de credito por unidade de
        # ruido). Subir para 0.967 pagaria o dobro do ruido por +3.5 turnos.
        self.trace_lambda = 0.933
        self.use_traces = True
        # Traces por batalha: {battle_key: {(abs_state, action_idx): elegibilidade}}
        # Tem de ser POR BATALHA porque ha varias batalhas concorrentes.
        self._traces = {}
        # Abaixo deste valor a elegibilidade e descartada (poupa memoria e tempo).
        self.trace_min = 0.01
        # --- Recompensa terminal ---
        # Reduzida de 5000 para 1000: com 5000 a variancia era ~278x a dos abates e
        # afogava os sinais densos (ver ESTADO_DO_PROJETO 6.9). 1000 mantem o terminal
        # como o maior sinal individual (10x um abate) sem dominar por ruido.
        self.terminal_reward = 1000.0
        # Lambda MAIOR para o update terminal: o desfecho deve chegar as jogadas de
        # setup (hazards, status) que aconteceram no inicio da batalha, nao so as
        # ultimas. 0.98 -> horizonte ~50 turnos, cobre a batalha inteira (~43t).
        self.trace_lambda_terminal = 0.98
        # Traco de EPISODIO: acumulado ao longo de toda a batalha e NAO cortado por
        # jogadas exploratorias. Usado APENAS no update terminal. O corte off-policy
        # existe porque o bootstrap assume continuacao gulosa; no terminal nao ha
        # bootstrap (Q futuro = 0), logo a razao para cortar nao se aplica.
        self._episode_traces = {}
        # Pesos do potencial para setup (hazards e status).
        self.peso_hazard = 10.0
        self.peso_status = 8.0

        # --- CONTROLO DO VIES DE MAXIMIZACAO (ambos DESLIGADOS por omissao) ---
        #
        # PORQUE ESTAO DESLIGADOS: a sobrestimacao dos Q-values, por si so, NAO
        # prejudica a politica. A decisao e argmax_a Q(s,a) DENTRO de cada estado: se
        # todas as acoes de um estado sao inflacionadas de forma semelhante, a escolha
        # nao muda. So a sobrestimacao DIFERENCIAL faz mal.
        #
        # Evidencia empirica: o Blue v4 (sem terminal, teto 600) tinha Q ate 1145 —
        # 1.9x o teto — e era a versao com a melhor distribuicao de estados maduros e
        # comportamento saudavel. Sobrestimar nao o impediu de funcionar.
        #
        # E cada correcao tem um custo real:
        #   - o encolhimento penaliza acoes que levam a estados pouco visitados, ou
        #     seja, DESENCORAJA explorar regioes novas (pessimismo sistematico);
        #   - a trava, se atuar em varias acoes do mesmo estado, empata-as no teto e
        #     o argmax passa a decidir por ordem de indice, nao por valor.
        #
        # Ficam disponiveis para experimentacao A/B, mas o comportamento por omissao
        # e o do v4: sem interferencia.
        self.usar_encolhimento = False
        self.shrink_k = 20.0

        # TRAVA DE SEGURANCA, nao de correcao. Fica MUITO acima do teto teorico
        # (10x) para nunca interferir na operacao normal: existe apenas para impedir
        # uma explosao numerica como a de 996.758 que corrompeu um treino de 200k.
        self.q_clip = None            # None = 10x o teto teorico
        self.clip_events = 0
        self.max_q_pre_clip = 0.0
        # Flag: a ultima decisao foi exploratoria? (usada para o corte do traco)
        self.ultima_foi_exploratoria = False
        # Limiar de alarme para deteção de divergencia dos Q-values.
        # Limiar de alarme: 5x o teto teorico do episodio. Fica ABAIXO do teto de
        # seguranca (10x), para o detetor disparar ANTES de a trava atuar. Se fosse
        # ao contrario, a trava mascarava a divergencia.
        self.limiar_divergencia = 5.0 * (1000.0 + 600.0 + 400.0)

        # Peso da priorização por TD-error no replay (0 = desligado, 1 = só surpresa).
        self.td_priority_alpha = 0.6

        self.q_table = {}
        self.visit_counts = {}

        # MEDICAO N(s,a) POS-MATURIDADE (17/09/2026).
        # Objetivo: distinguir "estado maduro" de "acoes realmente experimentadas
        # dentro desse estado". Para nao duplicar a Q-table de 1,7M estados, so
        # passamos a persistir contagens quando o estado ja atingiu LIMIAR_MADURO.
        # Assim medimos exatamente a politica no nucleo que concentra ~70% das
        # decisoes, que e a pergunta causal atual, com custo de memoria controlado.
        self.action_choice_counts = {}   # estado -> np.uint32[len(actions)]
        self.action_valid_masks = {}     # estado -> bitmask das acoes vistas como validas

        self._qtable_lock = threading.Lock()

        # ==================================================================
        # PROPOSTA A (11/09/2026): MEMORIA DIMENSIONADA AO ORCAMENTO
        # ==================================================================
        # HISTORIA. 15.000 cobria ~380 batalhas e o replay reciclava quase so o
        # passado imediato — o contrario da razao de existir do replay, que e quebrar
        # a correlacao entre amostras. Subiu para 40.000 (~1.000 batalhas) no ciclo
        # v9/v6, quando o treino era de 400k e a Q-table tinha ~500 mil estados.
        #
        # O QUE MUDOU. O ciclo v10/v7 correu 500k batalhas e produziu 1,7 MILHOES de
        # estados (6.49 A). Com `maxlen=40000` o buffer passou a cobrir 0,2% do
        # treino: o replay via uma janela de 1.000 batalhas e reprocessava-a, sem
        # nunca voltar a tocar em 99,8% da experiencia.
        #
        # A CORRECCAO E ESCALAR, NAO FIXAR. `memory_maxlen` passa a ser parametro do
        # construtor e o treino calcula-o a partir do orcamento:
        #
        #     maxlen = batalhas_previstas * turnos_por_batalha * fraccao_de_cobertura
        #
        # Com 500k batalhas, ~30 decisoes por batalha e 2% de cobertura, da ~300.000.
        # O default abaixo (400.000, ~2% de um ciclo de 660k) cobre o orcamento que a
        # projeccao de cobertura sugere para o proximo ciclo (6.49 E).
        #
        # CUSTO. Uma transicao e `(state, action_str, reward, next_state)`: dois
        # tuplos de 16 strings, uma string e um float. A ~40 bytes uteis por
        # transicao mais overhead do deque, 400.000 ficam na ordem dos 16 a 60 MB de
        # RAM. Contra os 653 MB que o `.pkl` ja ocupa aos 50k, e barato.
        #
        # O QUE ISTO NAO RESOLVE. Um buffer maior nao torna o replay mais inteligente:
        # so lhe da de onde escolher. A escolha e das propostas B e C.
        self.memory = deque(maxlen=memory_maxlen)
        self.memory_maxlen = memory_maxlen
        self.batch_size = 512

        self.wr_history = []

        # Rastreadores de recorde (mantidos p/ Hall da Fama e diagnóstico)
        self.episode_reward_max = -9999.0
        self.episode_reward_min = 9999.0
        self.active_battles_reward = {}

        # Parametro historico da funcao de novidade. A funcao esta INATIVA na
        # configuracao actual (sem chamadores); manter apenas por compatibilidade.
        self.novelty_k = novelty_k

        # ------------------------------------------------------------------
        # Hall da Fama (memória de elite)
        # ------------------------------------------------------------------
        # CRITÉRIO REVISTO EM 25/08/2026. O anterior era
        #     if final_ep_reward > self.episode_reward_max and > 100
        # ou seja: só entrava quem BATESSE O RECORDE ABSOLUTO de sempre. Como a
        # barreira só sobe, a memória enchia nas primeiras centenas de batalhas e
        # congelava. Medido no ciclo v7/v4: 198 transições ao fim de 200.000
        # batalhas, contra um maxlen de 2000.
        #
        # Isso não era inofensivo: 25% de cada lote de replay é sorteado da elite
        # (ver replay()), logo um quarto da capacidade de consolidação passou 200 mil
        # batalhas a reprocessar as mesmas ~198 transições — e transições do INÍCIO
        # do treino, quando epsilon estava a 0,40 e a política era quase aleatória.
        # O nome sugeria jogadas geniais; o conteúdo eram os primeiros episódios que
        # por acaso bateram um recorde ainda baixo.
        #
        # Agora a barreira é um PERCENTIL MÓVEL das recompensas recentes: acompanha a
        # melhoria do agente, a memória renova-se, e o maxlen passa a fazer sentido.
        # ==================================================================
        # PROPOSTA C (11/09/2026): CAPACIDADE E QUOTA DA ELITE
        # ==================================================================
        # O CRITERIO (percentil movel) ESTA CERTO e resolveu o congelamento de 25/08.
        # O problema que resta e de CAPACIDADE e de PESO, e sao dois.
        #
        # CAPACIDADE. Com `maxlen=2000` e percentil 0,95 entram ~5% dos episodios. Em
        # 500 mil batalhas sao ~25.000 candidatos para 2.000 lugares: a memoria e
        # substituida ~12 vezes e no fim so contem material das ultimas ~40 mil
        # batalhas. Isso faz o Hall da Fama convergir para "o que o agente ja faz bem
        # AGORA", que e precisamente o que ele menos precisa de reforcar. A
        # diversidade desaparece, e com ela o valor do replay de elite para material
        # inedito.
        #
        # PESO. A quota era 25% FIXOS do lote. Com 512 de lote sao 128 transicoes de
        # elite por chamada, 20 chamadas por bloco, 500 blocos: os mesmos episodios
        # reprocessados mais de um milhao de vezes. E a mesma armadilha que o
        # comentario de 25/08 descreve (198 transicoes recicladas durante 200 mil
        # batalhas), so que com 2.000 em vez de 198 — o numero subiu, a PROPORCAO
        # continuou errada.
        #
        # A CORRECCAO E DUPLA:
        #   1. `elite_maxlen` sobe para 20.000 (~1 MB) e passa a ser parametro.
        #   2. `elite_quota` desce de 0,25 para 0,10, e no replay e ainda limitada
        #      pelo PESO REAL da elite na memoria (ver `replay_experience`), para a
        #      quota nunca sobre-representar um conjunto pequeno.
        self.elite_memory = deque(maxlen=elite_maxlen)
        self.elite_maxlen = elite_maxlen
        self.elite_quota = elite_quota
        self.active_battles_transitions = {}
        self.recompensas_recentes = deque(maxlen=1000)
        self.elite_percentil = 0.95      # entra quem estiver no topo dos 5% recentes
        self.elite_min_amostras = 100    # abaixo disto não há percentil fiável
        self.elite_reward_minimo = 100.0 # piso absoluto, herdado do critério antigo

        # ==================================================================
        # PROPOSTA B (11/09/2026): LIMIAR DE FRESCURA RELATIVO
        # ==================================================================
        # O `replay_experience` escolhe entre dois modos conforme a fraccao de
        # transicoes em estados de 1 visita (`single_visit_ratio`):
        #
        #     ratio > 0,30  -> PRIORIZAR FRESCOS   (70% do lote no balde de 1 visita)
        #     caso contrario -> CONSOLIDAR MADUROS (60% do lote no balde de 2 a 4)
        #
        # O limiar 0,30 foi calibrado com ~500 mil estados. Com os 1,7 MILHOES do
        # v10/v7, 48% dos estados tem uma so visita e o ratio fica quase sempre acima
        # de 0,30: **o modo de CONSOLIDAR quase nunca corre**. O replay passou o
        # treino inteiro a perseguir novidade e nunca a fixar o que ja tinha sido
        # visto, o que e consistente com a cobertura ter estagnado nos 70%.
        #
        # A CORRECCAO E TORNAR O LIMIAR RELATIVO AO PROPRIO TREINO. Guarda-se a media
        # movel do ratio e compara-se o valor actual com ela:
        #
        #     ratio acima da media  -> a novidade esta a ACELERAR   -> priorizar frescos
        #     ratio abaixo da media -> a novidade esta a ABRANDAR   -> consolidar
        #
        # Assim os dois modos correm em QUALQUER escala de Q-table, porque o criterio
        # deixa de ser um valor absoluto e passa a ser a DERIVADA do proprio processo.
        #
        # O limiar absoluto fica como rede para o arranque: enquanto nao houver
        # amostras suficientes para uma media fiavel, usa-se o comportamento antigo.
        # Isso preserva o inicio do treino, que e onde o 0,30 estava bem calibrado.
        self.single_visit_ratio_hist = deque(maxlen=50)
        self.single_visit_ratio_min_amostras = 10
        self.single_visit_ratio_limiar_absoluto = 0.30

        # Ações: base + variante _MEC (mecânica) para as não-switch
        self.base_actions = [
            "ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH",
            "BUFF", "STATUS", "HEAL", "CLEAN_HAZARD",
            "PROTECT", "DEBUFF", "DISRUPTION", "STAT_CLEAN", "HEAL_STATUS", "PHAZE",
            "FIELD_CONTROL", "HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
            "BARRIER",
        ]
        self.actions = []
        for act in self.base_actions:
            self.actions.append(act)
            if "SWITCH" not in act:
                self.actions.append(f"{act}_MEC")

    # ======================================================================
    # DIAGNÓSTICO
    # ======================================================================

    # ==================================================================
    # METRICAS DE CONVERGENCIA (30/08/2026)
    # ==================================================================
    # PORQUE ESTE BLOCO EXISTE. A metrica de cobertura usada ate aqui era "visitas
    # por estado" (media simples). Ela trata todos os estados como igualmente
    # importantes, e NAO SAO: medido nos cerebros de 400k, 16,7% dos estados
    # absorvem 90,7% de todas as decisoes, e a mediana de visitas e 3 enquanto o
    # estado mais visitado tem 61.967.
    #
    # A media simples respondia "quantas visitas tem um estado tipico". A pergunta
    # que decide e outra: "que fracao dos TURNOS o agente joga em terreno que ja
    # conhece". Sao numeros muito diferentes e levam a decisoes opostas sobre
    # orcamento.
    #
    # E cobertura, por si so, nao e convergencia. Q-Learning converge quando a
    # POLITICA (o argmax) para de mudar, nao quando os valores param de se mexer:
    # a tabela pode continuar a somar valor a todas as accoes sem que a decisao
    # mude nunca, e pode ter valores quase estaveis com a decisao a oscilar entre
    # duas accoes empatadas. Por isso ha snapshot de politica.

    LIMIAR_MADURO = 20          # visitas a partir das quais um estado "conta"

    def inspect_brain(self):
        """(total_visitas, media_simples, taxa_confianca). MANTIDO POR COMPATIBILIDADE.

        As duas ultimas medidas estao DESPROMOVIDAS e nao devem guiar decisoes:

          media_simples   trata todos os estados por igual; ver o bloco acima
          taxa_confianca  e so `visit_counts >= 3`, um limiar tao baixo que mede
                          cobertura rasa e nao confianca nenhuma. O nome prometia
                          mais do que entregava (56,1% no Blue de 400k).

        Use `metricas_convergencia()`.
        """
        total_states = len(self.q_table)
        if total_states == 0:
            return 0, 0.0, 0.0
        total_visits = 0
        confident_states = 0
        for state in self.q_table:
            state_visits = self.visit_counts.get(state, 1)
            total_visits += state_visits
            if state_visits >= 3:
                confident_states += 1
        return total_visits, total_visits / total_states, (confident_states / total_states) * 100.0

    def _snapshot_politica(self):
        """{estado_maduro: (idx_argmax, q_do_argmax)}. Base do churn e do delta.

        SO ESTADOS MADUROS, de proposito e por duas razoes. A primeira e memoria:
        guardar a tabela inteira duplicaria o .pkl (165 MB no Blue de 400k); os
        maduros sao 16,7% dos estados e um tuplo pequeno cada. A segunda e sentido:
        um estado com duas visitas TEM de mudar de argmax, isso e aprendizado e nao
        instabilidade. Contar essas mudancas afogaria o sinal em ruido esperado.
        """
        snap = {}
        for estado, q in self.q_table.items():
            if self.visit_counts.get(estado, 0) >= self.LIMIAR_MADURO:
                idx = int(np.argmax(q))
                snap[estado] = (idx, float(q[idx]))
        return snap

    def metricas_convergencia(self, guardar_snapshot=True):
        """Metricas de convergencia do bloco. Devolve dict.

          cobertura_ponderada   % das DECISOES tomadas em estados maduros.
                                Substitui "visitas por estado". Medido: 90,7% no
                                Blue de 400k, com so 16,7% dos estados maduros.
          descoberta_por_turno  estados novos por DECISAO desde a chamada anterior.
                                E a unidade certa: uma batalha sao ~40 decisoes,
                                logo "500 novos por 1000 batalhas" sao 0,0125 por
                                turno, ou seja 98,75% dos turnos caem em terreno
                                conhecido.
          churn_politica        % da MASSA DE DECISAO cujo argmax mudou desde o
                                bloco anterior. E O CRITERIO DE CONVERGENCIA:
                                ponderado por visitas, porque mudar a decisao num
                                estado de 60.000 visitas nao vale o mesmo que num
                                de 21.
          delta_q               |dQ| medio do argmax, ponderado por visitas e
                                normalizado pelo teto. Complementa o churn: mostra
                                se os valores ainda se movem quando a decisao ja
                                nao muda.
          margem_decisao        distancia media entre a melhor e a segunda melhor
                                accao nos estados maduros, ponderada. Margem larga
                                significa que o ruido que resta nao chega para
                                virar a politica.
          estados_maduros       contagem absoluta.

        No primeiro bloco de uma sessao nao ha snapshot anterior, e `churn_politica`
        e `delta_q` saem a -1.0 para nao serem confundidos com zero (que seria
        "convergiu").
        """
        n = len(self.q_table)
        if n == 0:
            return {"cobertura_ponderada": 0.0, "descoberta_por_turno": 0.0,
                    "churn_politica": -1.0, "delta_q": -1.0,
                    "margem_decisao": 0.0, "estados_maduros": 0}

        total_visitas = 0
        massa_madura = 0
        maduros = 0
        soma_margem = 0.0
        peso_margem = 0
        for estado, q in self.q_table.items():
            v = self.visit_counts.get(estado, 0)
            total_visitas += v
            if v >= self.LIMIAR_MADURO:
                maduros += 1
                massa_madura += v
                if len(q) >= 2:
                    ordenado = np.partition(q, -2)
                    soma_margem += (float(ordenado[-1]) - float(ordenado[-2])) * v
                    peso_margem += v

        # Descoberta por DECISAO, e nao por batalha nem por bloco.
        anterior_estados = getattr(self, "_conv_estados_ant", None)
        anterior_visitas = getattr(self, "_conv_visitas_ant", None)
        if anterior_estados is None or anterior_visitas is None:
            descoberta = -1.0
        else:
            d_visitas = total_visitas - anterior_visitas
            descoberta = (n - anterior_estados) / d_visitas if d_visitas > 0 else 0.0

        snap = self._snapshot_politica()
        antigo = getattr(self, "_conv_snapshot", None)
        if not antigo:
            churn, delta = -1.0, -1.0
        else:
            peso, mudou, soma_delta = 0, 0, 0.0
            teto = max(1.0, float(self._teto_q()))
            for estado, (idx, qv) in snap.items():
                anterior = antigo.get(estado)
                if anterior is None:
                    continue          # estado que amadureceu agora: sem termo de comparacao
                v = self.visit_counts.get(estado, 0)
                peso += v
                if anterior[0] != idx:
                    mudou += v
                soma_delta += abs(qv - anterior[1]) * v
            churn = (mudou / peso * 100.0) if peso else -1.0
            delta = (soma_delta / peso / teto * 100.0) if peso else -1.0

        if guardar_snapshot:
            self._conv_snapshot = snap
            self._conv_estados_ant = n
            self._conv_visitas_ant = total_visitas

        return {
            "cobertura_ponderada": (massa_madura / total_visitas * 100.0) if total_visitas else 0.0,
            "descoberta_por_turno": descoberta,
            "churn_politica": churn,
            "delta_q": delta,
            "margem_decisao": (soma_margem / peso_margem) if peso_margem else 0.0,
            "estados_maduros": maduros,
        }

    def histograma_visitas(self, limiares=(1, 2, 3, 5, 10, 20, 30, 57, 100, 200, 500, 1000)):
        """Massa de decisao acumulada por limiar de visitas. Para o dashboard.

        Substitui os quatro baldes fixos, que escondiam a forma real da cauda: a
        mediana e 3 visitas e o topo tem 61.967, quatro ordens de grandeza.
        """
        if not self.visit_counts:
            return {}
        v = np.fromiter(self.visit_counts.values(), dtype=np.int64)
        tot = max(1, int(v.sum()))
        return {lim: {"estados": int((v >= lim).sum()),
                      "massa_pct": float(v[v >= lim].sum()) / tot * 100.0}
                for lim in limiares}

    # ======================================================================
    # FASES DE CURRÍCULO (mantido; o treino atual usa só "instinct")
    # ======================================================================

    # REMOVIDO EM 25/08/2026: `enter_phase()` e `current_phase`.
    # Resíduo do curriculum learning (maxdamage -> instinct -> selfplay), abandonado
    # há muito. O método nunca era chamado e `current_phase` ficava fixo em
    # "maxdamage", o que aparecia no print de carregamento do cérebro como
    # "Fase: MAXDAMAGE" e confundia quem lia o terminal.

    def _limiar_elite(self):
        """Barreira de entrada no Hall da Fama: percentil movel das recompensas
        recentes, nunca abaixo do piso absoluto.

        SUBSTITUI o criterio antigo `final_ep_reward > self.episode_reward_max`, que
        exigia BATER O RECORDE ABSOLUTO de sempre. Como essa barreira so subia, a
        memoria de elite enchia nas primeiras centenas de batalhas e congelava:
        medido nos cerebros v7/v4, 198 transicoes ao fim de 200.000 batalhas, contra
        um maxlen de 2000. E 25% de cada lote de replay era sorteado dessas 198,
        quase todas do inicio do treino, com epsilon a 0,40 e politica quase
        aleatoria.

        Com menos de `elite_min_amostras` batalhas registadas ainda nao ha
        distribuicao para calcular percentil, e usa-se so o piso. Isso deixa a
        memoria arrancar no inicio do treino sem ficar presa depois.
        """
        if len(self.recompensas_recentes) < self.elite_min_amostras:
            return self.elite_reward_minimo
        ordenadas = sorted(self.recompensas_recentes)
        idx = min(int(len(ordenadas) * self.elite_percentil), len(ordenadas) - 1)
        return max(ordenadas[idx], self.elite_reward_minimo)

    def _update_global_records(self, final_reward):
        if final_reward > self.episode_reward_max:
            self.episode_reward_max = final_reward
        if final_reward < self.episode_reward_min:
            self.episode_reward_min = final_reward

    # ======================================================================
    # POTENCIAL (shaping estático) + FUNCAO DE NOVIDADE INATIVA
    # ======================================================================

    def _get_novelty_bonus(self, state):
        """Funcao historica de novidade, actualmente SEM chamadores no treino.

        Mantida apenas para compatibilidade/experimentos. Nao entra em PBRS, reward
        nem na policy da configuracao v11/v8. Qualquer futura exploracao N(s,a) deve
        ser introduzida explicitamente como experimento separado.
        """
        visits = self.visit_counts.get(self._get_abstract_state(state), 0)
        return self.novelty_k / np.sqrt(visits + 1.0)

    def _get_adapted_potential(self, battle, state):
        """Potencial de shaping = APENAS o potencial tático estático.

        CORREÇÃO CRÍTICA: o bónus de novidade foi REMOVIDO daqui. Somá-lo ao phi
        quebrava o teorema do PBRS (Ng, Harada & Russell 1999), que exige que phi
        seja uma função ESTÁVEL do estado para não alterar a política óptima. O bónus
        k/sqrt(visits) muda a cada visita, logo não é estável — e, pior, no shaping
        phi(s')-phi(s) ele premiava transições para estados POUCO visitados, que no
        meio de uma batalha costumam ser estados MAUS (situações raras = situações
        de sarilho). Isso empurrava o Win Rate para BAIXO ao longo do treino.

        A exploração por novidade pertence à POLÍTICA (decide_action), não à
        recompensa. Aqui o shaping volta a ser PBRS puro e neutro: só orienta com o
        conhecimento tático estável, sem distorcer o que é uma jogada boa ou má.
        """
        return self._calculate_potential(battle, state)

    def _calculate_potential(self, battle, state):
        """Potencial tático estático Phi = Phi_guerra (macro) + Phi_batalha (micro)."""
        # 15 e o MINIMO historico, nao a dimensao atual (16 desde 30/08/2026). Fica
        # como piso porque todos os indices lidos aqui vao ate 14: a dimensao de banco
        # foi acrescentada no FIM justamente para nao deslocar nenhum deles. Se
        # alguma vez se inserir uma dimensao a MEIO, este numero e estes indices tem
        # de ser revistos em conjunto.
        if not state or len(state) < 15:
            return 0.0
        phi_guerra = 0.0
        phi_batalha = 0.0

        matchup_vals = {
            "DOMINANT": 40.0, "OFFENSIVE_ADV": 20.0, "DEFENSIVE_ADV": 10.0,
            "NEUTRAL": 0.0, "VOLATILE": -5.0, "STALEMATE": 0.0,
            "OFFENSIVE_DIS": -10.0, "DEFENSIVE_DIS": -20.0, "CRITICAL_DIS": -40.0,
        }
        phi_batalha += matchup_vals.get(str(state[2]).upper(), 0.0)

        context_vals = {"DOMINATING": 30.0, "RECOVERING": -30.0}
        phi_guerra += context_vals.get(str(state[14]).upper(), 0.0)

        my_total_hp = sum(m.current_hp_fraction for m in battle.team.values())
        opp_total_hp = sum(m.current_hp_fraction for m in battle.opponent_team.values())
        phi_guerra += (my_total_hp - opp_total_hp) * 40.0

        # BUG CORRIGIDO: os indices estavam errados E o sinal invertido.
        #   state[11] = opp_BOOST (valores NEUTRAL/BUFFED/DEBUFF) -> nunca "SET",
        #               logo a penalizacao era CODIGO MORTO.
        #   state[12] = MY_hazard -> o agente era RECOMPENSADO (+10) por ter hazards
        #               no PROPRIO campo.
        # Indices corretos: 12 = my_hazard (mau), 13 = opp_hazard (bom).
        # Consequencia do bug: montar hazards no adversario — a jogada de setup mais
        # importante do jogo — nunca foi recompensada pelo shaping.
        if state[12] == "SET":
            phi_guerra -= self.peso_hazard      # hazards no MEU campo: mau
        if state[13] == "SET":
            phi_guerra += self.peso_hazard      # hazards no campo INIMIGO: bom


        # ATUALIZADO 30/08/2026 para os SEIS baldes de `get_weather_state` (a
        # primeira versao tinha cinco; `FIELD_OURS` foi separado em `_OFF` e
        # `_DEF` na mesma data, ver 6.44 do ESTADO_DO_PROJETO.md). Os
        # nomes antigos (FIELD_POWER, FIELD_SPEED, FIELD_DEFENSE, FIELD_HOSTILE,
        # FIELD_NEUTRAL) DEIXARAM DE EXISTIR: se ficassem aqui, o `.get` devolvia
        # sempre 0.0 e o shaping de campo tornava-se codigo morto sem dar erro
        # nenhum. E o mesmo padrao de falha silenciosa que ja custou caro no projeto.
        #
        # Com 6 baldes (revisao de 30/08/2026) cada nome antigo tem herdeiro directo:
        #   FIELD_OURS_OFF  <- os +15 de FIELD_POWER e FIELD_SPEED
        #   FIELD_OURS_DEF  <- os +10 de FIELD_DEFENSE
        #   FIELD_THEIRS    <- os -20 de FIELD_HOSTILE, e passa a cobrir tambem a
        #                      vantagem do adversario, que ate aqui valia 0.0
        #   FIELD_SHARED    <- 0.0 por definicao: serve aos dois, ou a nenhum
        field_vals = {"FIELD_SWEEP": 20.0, "FIELD_OURS_OFF": 15.0,
                      "FIELD_OURS_DEF": 10.0, "FIELD_THEIRS": -20.0}
        phi_guerra += field_vals.get(str(state[5]).upper(), 0.0)

        my_role = str(state[0]).upper()
        speed_tier = str(state[6]).upper()
        if speed_tier == "FASTER":
            phi_batalha += 20.0 if my_role == "SWEEPER" else 5.0
        else:
            phi_batalha -= 20.0 if my_role == "SWEEPER" else 0.0

        # BUG CORRIGIDO: todos estes indices estavam deslocados uma posicao para tras.
        # Ordem real do estado (ver shared/state.py):
        #   8=my_status  9=opp_status  10=my_boost  11=opp_boost  7=mechanic
        # O codigo lia my_status<-mechanic (nunca AFFLICTED, penalizacao morta),
        # opp_status<-my_status (premiava o proprio status), e mec_state<-opp_hazard.
        my_boost = str(state[10]).upper()
        opp_boost = str(state[11]).upper()
        my_status = str(state[8]).upper()
        opp_status = str(state[9]).upper()
        mec_state = str(state[7]).upper()

        if "BUFFED" in my_boost:
            phi_batalha += 10.0
        if "DEBUFF" in my_boost:
            phi_batalha -= 10.0
        if my_status == "AFFLICTED":
            phi_batalha -= 15.0
        if opp_status == "AFFLICTED":
            phi_batalha += 15.0
        if "BUFFED" in opp_boost:
            phi_batalha -= 10.0
        if "DEBUFF" in opp_boost:
            phi_batalha += 10.0
        if mec_state == "MEC_AVAIL":
            phi_batalha += 10.0

        return phi_guerra + phi_batalha

    # ======================================================================
    # RECOMPENSA (PBRS não-descontado + faints + Hall da Fama)
    # ======================================================================

    def calculate_reward(self, battle, history, current_state=None):
        ext_reward = 0.0
        if battle.won:
            ext_reward += self.terminal_reward
        elif battle.lost:
            ext_reward -= self.terminal_reward

        current_my_fainted = len([m for m in battle.team.values() if m.fainted])
        current_opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        if current_my_fainted > history.get('my_fainted', 0):
            ext_reward -= 100.0
        if current_opp_fainted > history.get('opp_fainted', 0):
            ext_reward += 100.0

        tag = battle.battle_tag
        if tag not in self.active_battles_reward:
            self.active_battles_reward[tag] = 0.0
            self.active_battles_transitions[tag] = []
        self.active_battles_reward[tag] += ext_reward

        # Shaping: diferença de potencial (sem gamma no shaping = PBRS não-descontado)
        phi_current = self._get_adapted_potential(battle, current_state)
        # CORREÇÃO: last_phi pode estar PRESENTE com valor None (primeiro turno da
        # batalha). O default de .get() só cobre a chave ausente. Se for None, usamos
        # phi_current -> shaping 0 no primeiro turno (não há transição anterior).
        phi_prev = history.get('last_phi')
        if phi_prev is None:
            phi_prev = phi_current
        # PBRS CORRETO (Ng, Harada & Russell 1999): F = gamma*Phi(s') - Phi(s).
        # A versao anterior omitia o gamma, o que dava um bonus extra de (1-gamma)*Phi
        # POR TURNO passado num estado bom — um incentivo residual a acomodar-se.
        # Com o gamma reposto:
        #   ficar no MESMO estado bom (Phi=40)  -> -0.40  (custo infimo, mas negativo)
        #   MELHORAR de Phi=40 para Phi=60      -> +19.40 (so progredir compensa)
        # Isto satisfaz o requisito: permanecer paga quase nada, avancar paga.
        shaping_reward = self.gamma * phi_current - phi_prev
        history['last_phi'] = phi_current
        final_turn_reward = ext_reward + shaping_reward

        # Grava a transição (para o Hall da Fama)
        last_state = history.get('state')
        last_action_tuple = history.get('last_action')
        if last_state and last_action_tuple:
            base_action, mechanic = last_action_tuple
            action_str = f"{base_action}_MEC" if mechanic else base_action
            actual_next_state = current_state
            if actual_next_state is None:
                actual_next_state = ("TERMINAL_WIN",) if battle.won else ("TERMINAL_LOSS",)
            self.active_battles_transitions[tag].append((last_state, action_str, final_turn_reward, actual_next_state))

        # Fim de batalha: julgamento para o Hall da Fama
        if battle.won or battle.lost:
            final_ep_reward = self.active_battles_reward.pop(tag, 0.0)
            # A recompensa entra na distribuição ANTES do julgamento, para o percentil
            # refletir também esta batalha.
            self.recompensas_recentes.append(final_ep_reward)
            if final_ep_reward >= self._limiar_elite():
                for t in self.active_battles_transitions.get(tag, []):
                    self.elite_memory.append(t)
            self.active_battles_transitions.pop(tag, None)
            self._update_global_records(final_ep_reward)

        # Coletor de lixo para batalhas-fantasma
        if len(self.active_battles_reward) > 20:
            for k in list(self.active_battles_reward.keys())[:-10]:
                self.active_battles_reward.pop(k, None)
                self.active_battles_transitions.pop(k, None)

        return final_turn_reward

    # ======================================================================
    # ESTADO ABSTRATO + UPDATE DA Q-TABLE
    # ======================================================================

    def _aplicar_teto(self, valor):
        """Trava o valor no teto, REGISTANDO a ocorrencia.

        O registo e essencial: sem ele a trava mascara a divergencia. Com ele, um
        cerebro doente e detetavel por clip_events alto e max_q_pre_clip enorme,
        mesmo que nenhum Q armazenado passe do teto.
        """
        mag = abs(float(valor))
        if mag > self.max_q_pre_clip:
            self.max_q_pre_clip = mag
        teto = self._teto_q()
        if mag > teto:
            self.clip_events += 1
            return teto if valor > 0 else -teto
        return valor

    def _teto_q(self):
        """Teto teorico do valor de um par (estado, acao).

        Retorno maximo de um episodio = recompensa terminal + abates + shaping.
        Os abates dao +100 por turno em que um Pokemon inimigo cai (max 6 por
        batalha). O shaping telescopa, mas damos margem para o seu intervalo.
        """
        if self.q_clip is not None:
            return self.q_clip
        # 10x o retorno maximo teorico: nunca atua em operacao normal, so trava
        # explosao numerica.
        return 10.0 * (self.terminal_reward + 6 * 100.0 + 400.0)

    def _valor_encolhido(self, abs_state):
        """max_a Q(s,a) encolhido pela incerteza do estado.

        Multiplica por n/(n+k): estados pouco visitados contribuem pouco para o
        bootstrap, o que remove a maior parte do vies de maximizacao sem duplicar a
        memoria (como exigiria o Double Q-learning).
        """
        if abs_state not in self.q_table:
            return 0.0
        bruto = float(np.max(self.q_table[abs_state]))
        if not self.usar_encolhimento:
            return bruto          # comportamento padrao (v4): sem interferencia
        n = float(self.visit_counts.get(abs_state, 0))
        return bruto * (n / (n + self.shrink_k))

    def _get_abstract_state(self, state):
        """Passthrough: a Q-table aprende sobre o estado completo. Mantido como
        ponto único de abstração caso se queira comprimir o estado no futuro."""
        return state

    def update_feedback(self, current_state, last_state, last_action_tuple, reward,
                        battle_key=None, was_exploratory=False):
        """Feedback do mundo real: grava na memoria e faz o update imediato.

        battle_key      : identificador da batalha (battle_tag). Necessario porque os
                          eligibility traces sao POR BATALHA e ha 3 batalhas em
                          paralelo — sem isto os tracos misturavam-se.
        was_exploratory : True se a acao anterior foi escolhida por exploracao. No
                          Q(lambda) de Watkins o traco tem de ser CORTADO nessas
                          jogadas, senao credita estados por resultados produzidos
                          ao acaso.
        """
        if last_state is None or last_action_tuple is None:
            return
        base_action, mechanic = last_action_tuple
        action_str = f"{base_action}_MEC" if mechanic else base_action
        if action_str not in self.actions:
            return

        self.memory.append((last_state, action_str, reward, current_state))

        if self.use_traces and battle_key is not None:
            self._apply_q_update_traced(last_state, action_str, reward, current_state,
                                        battle_key, was_exploratory)
        else:
            self._apply_q_update(last_state, action_str, reward, current_state, is_replay=False)

    # ------------------------------------------------------------------
    # ELIGIBILITY TRACES (replacing) — Watkins Q(lambda) com corte off-policy
    # ------------------------------------------------------------------

    def _apply_q_update_traced(self, state, action_str, reward, next_state,
                               battle_key, was_exploratory):
        """Update com eligibility traces: o erro TD e propagado por TODA a trajetoria
        recente, com peso decrescente, em vez de so pelo ultimo par (estado,acao).

        Porque isto importa aqui: a recompensa e dominada pelo terminal (+/-5000) e a
        batalha tem ~39 turnos. Com TD(0) o resultado so recua UM passo por update,
        logo as decisoes de inicio/meio de batalha quase nunca aprendem se levaram a
        vitoria. Com traces, uma unica batalha propaga o desfecho para tras.

        REPLACING trace (Singh & Sutton 1996): ao revisitar um par, a elegibilidade e
        REPOSTA a 1 em vez de somada. O traco acumulativo cresceria sem limite em
        estados que se repetem dentro da batalha (1.0, 1.9, 2.7, 3.4...), inflacionando
        o credito desses estados. O replacing satura em 1 e e nao-enviesado.
        """
        abs_state = self._get_abstract_state(state)
        abs_next = self._get_abstract_state(next_state)
        action_idx = self.actions.index(action_str)

        with self._qtable_lock:
            if abs_state not in self.q_table:
                self.q_table[abs_state] = np.zeros(len(self.actions))
                self.visit_counts[abs_state] = 0
            self.visit_counts[abs_state] = self.visit_counts.get(abs_state, 0) + 1
            visits = self.visit_counts.get(abs_state, 1)

            if abs_next not in self.q_table:
                self.q_table[abs_next] = np.zeros(len(self.actions))
                self.visit_counts[abs_next] = 0

            # Erro TD (delta), igual ao do Q-learning normal.
            q_sa = self.q_table[abs_state][action_idx]
            if abs_next in [("TERMINAL_WIN",), ("TERMINAL_LOSS",)]:
                next_max = 0.0
            else:
                # Encolhido pela incerteza: remove o vies de maximizacao.
                next_max = self._valor_encolhido(abs_next)
            delta = reward + self.gamma * next_max - q_sa

            # Alpha efetivo (mesma regra do update normal).
            effective_alpha = max(self.min_alpha, self.alpha / (1.0 + 0.05 * visits))

            traces = self._traces.setdefault(battle_key, {})
            # REPLACING: repoe a 1 (nao soma).
            traces[(abs_state, action_idx)] = 1.0

            # TRACO DE EPISODIO (usado so no update terminal): decai mais devagar
            # (lambda_terminal) e NAO e cortado por exploracao, para o desfecho poder
            # chegar as jogadas de setup do inicio da batalha.
            ep = self._episode_traces.setdefault(battle_key, {})
            for k in list(ep.keys()):
                ep[k] *= self.gamma * self.trace_lambda_terminal
                if ep[k] < self.trace_min:
                    del ep[k]
            ep[(abs_state, action_idx)] = 1.0

            # ---------------------------------------------------------------
            # ESCALAMENTO DINAMICO DO PASSO (correcao critica)
            # ---------------------------------------------------------------
            # O mesmo delta e aplicado a TODOS os pares com elegibilidade viva.
            # Sem escalar, o passo efetivo e alpha * soma(elegibilidades), que com
            # lambda=0.933 e epsilon baixo chega a ~13 -> passo efetivo ~2.0.
            # Um passo acima de 1 ULTRAPASSA o alvo, o erro seguinte cresce na
            # direcao oposta e a serie DIVERGE. Foi o que inflacionou os Q-values
            # (996.758 em vez de ~2.360) e levou o agente a stalling com
            # PROTECT/HEAL: nesses estados a acao nao muda o estado abstrato
            # (s' == s), logo o Q faz bootstrap sobre si proprio e a realimentacao
            # positiva explode primeiro.
            #
            # Dividimos pela soma REAL das elegibilidades vivas (nao por uma
            # constante): assim, quando o corte off-policy encurta o traco, o passo
            # nao e penalizado desnecessariamente, e quando o traco e longo o passo
            # e travado na medida exata. Garante passo efetivo <= alpha, sempre.
            soma_elegibilidades = sum(traces.values())
            passo = effective_alpha / max(1.0, soma_elegibilidades)

            # Aplica o delta a TODOS os pares com elegibilidade viva.
            mortos = []
            for (s_k, a_k), e in traces.items():
                if s_k not in self.q_table:
                    mortos.append((s_k, a_k))
                    continue
                self.q_table[s_k][a_k] = self._aplicar_teto(
                    self.q_table[s_k][a_k] + passo * delta * e)
                # Decaimento para o proximo passo.
                novo_e = e * self.gamma * self.trace_lambda
                if novo_e < self.trace_min:
                    mortos.append((s_k, a_k))
                else:
                    traces[(s_k, a_k)] = novo_e
            for k in mortos:
                traces.pop(k, None)

            # CORTE OFF-POLICY (Watkins): se a acao tomada foi exploratoria, a
            # trajetoria seguinte deixa de refletir a politica gulosa, logo o traco
            # e zerado. Sem isto, estados seriam creditados por resultados de
            # jogadas aleatorias.
            if was_exploratory:
                traces.clear()

    def verificar_divergencia(self, limiar=None):
        """Deteta explosao dos Q-values (divergencia numerica).

        Devolve (ok, maior_q, mensagem). Chamado pelo script uma vez por bloco: se
        detetar divergencia, o treino deve PARAR — continuar so queima tempo a
        aprender uma politica corrompida (foi o que aconteceu no treino de 200k,
        onde os Q-values chegaram a 996.758 e o agente convergiu para stalling).

        O limiar por omissao e generoso: a escala sa dos Q-values e da ordem das
        dezenas de milhar (a recompensa terminal e 5000 e o fator de desconto 0.99
        da um teto teorico na ordem de 5000/(1-0.99) = 500.000 para um estado que
        vencesse sempre; usamos 200.000 como sinal de alarme precoce).
        """
        if limiar is None:
            limiar = self.limiar_divergencia
        if not self.q_table:
            return True, 0.0, "Q-table vazia"

        # CRITICO: usar o maior |Q| ANTES da trava. Se usassemos os valores
        # armazenados, a trava garantiria que nada passa do teto e o detetor
        # nunca dispararia — mascarando exatamente o que deve detetar.
        if self.max_q_pre_clip > limiar:
            return False, self.max_q_pre_clip, (
                f"maior |Q| ANTES da trava = {self.max_q_pre_clip:,.0f} acima do limiar "
                f"{limiar:,.0f}: divergencia mascarada pela trava.")
        if self.clip_events > 0:
            # Nao e falha, mas e sinal de alarme: num cerebro saudavel a trava
            # quase nunca deve atuar.
            pass
        maior = 0.0
        with self._qtable_lock:
            for v in self.q_table.values():
                m = float(np.max(np.abs(v)))
                if m > maior:
                    maior = m
        if not np.isfinite(maior):
            return False, maior, "Q-values NAO FINITOS (inf/nan): divergencia grave."
        if maior > limiar:
            return False, maior, (f"Maior |Q| = {maior:,.0f} acima do limiar "
                                  f"{limiar:,.0f}: possivel divergencia.")
        return True, maior, "escala dos Q-values dentro do esperado"

    def aplicar_update_terminal(self, state, action_str, reward, battle_key, venceu):
        """Update do desfecho, propagado pelo TRACO DE EPISODIO (mais largo).

        Diferencas face ao update normal:
          - usa `trace_lambda_terminal` (0.98 -> horizonte ~50 turnos, cobre a batalha
            inteira), enquanto o normal usa 0.933 (~15 turnos);
          - o traco de episodio NAO e cortado por jogadas exploratorias, porque no
            terminal nao ha bootstrap (Q futuro = 0) e a razao para cortar — o alvo
            assumir continuacao gulosa — nao se aplica;
          - Q futuro = 0 por definicao (estado terminal).

        Objetivo: dar valor as jogadas de SETUP (hazards, status) que acontecem no
        inicio da batalha e cujo retorno so se materializa no desfecho. Sem isto o
        agente so aprende o que da dano imediato.
        """
        abs_state = self._get_abstract_state(state)
        action_idx = self.actions.index(action_str)

        with self._qtable_lock:
            if abs_state not in self.q_table:
                self.q_table[abs_state] = np.zeros(len(self.actions))
                self.visit_counts[abs_state] = 0
            self.visit_counts[abs_state] = self.visit_counts.get(abs_state, 0) + 1
            visits = self.visit_counts.get(abs_state, 1)

            # Estado terminal: nao ha proximo estado, logo o alvo e so a recompensa.
            delta = reward - self.q_table[abs_state][action_idx]
            effective_alpha = max(self.min_alpha, self.alpha / (1.0 + 0.05 * visits))

            ep = self._episode_traces.setdefault(battle_key, {})
            ep[(abs_state, action_idx)] = 1.0

            # ESCALAMENTO DINAMICO obrigatorio: o traco de episodio e mais longo que o
            # normal, logo a soma das elegibilidades e maior. Sem dividir, o passo
            # efetivo ultrapassaria 1 e divergiria (ver 6.9 / 7 do ESTADO_DO_PROJETO).
            soma = sum(ep.values())
            passo = effective_alpha / max(1.0, soma)

            for (s_k, a_k), e in ep.items():
                if s_k in self.q_table:
                    self.q_table[s_k][a_k] = self._aplicar_teto(
                        self.q_table[s_k][a_k] + passo * delta * e)
                teto = self._teto_q()
                if abs(self.q_table[s_k][a_k]) > teto:
                    self.q_table[s_k][a_k] = float(np.clip(self.q_table[s_k][a_k], -teto, teto))

    def limpar_traces(self, battle_key):
        """Descarta os traces de uma batalha terminada (evita fuga de memoria)."""
        self._traces.pop(battle_key, None)
        self._episode_traces.pop(battle_key, None)

    def _apply_q_update(self, state, action_str, reward, next_state, is_replay=False):
        abs_state = self._get_abstract_state(state)
        abs_next = self._get_abstract_state(next_state)

        with self._qtable_lock:
            # CORREÇÃO (4): inicialização SEMPRE com np.zeros (nunca lista).
            if abs_state not in self.q_table:
                self.q_table[abs_state] = np.zeros(len(self.actions))
                self.visit_counts[abs_state] = 0

            # O replay ("sonho") não envelhece o estado: só visitas reais contam.
            if not is_replay:
                self.visit_counts[abs_state] = self.visit_counts.get(abs_state, 0) + 1

            visits = self.visit_counts.get(abs_state, 1)

            # Alpha efetivo por visitas: estados novos aprendem mais rápido.
            if visits <= 1:
                effective_alpha = min(0.5, self.alpha * 2.0)
            else:
                effective_alpha = max(self.alpha, self.alpha / (1 + 0.05 * visits))
            if is_replay:
                effective_alpha *= 0.2

            action_idx = self.actions.index(action_str)
            old_val = self.q_table[abs_state][action_idx]

            if abs_next in [("TERMINAL_WIN",), ("TERMINAL_LOSS",)]:
                next_max = 0.0
            else:
                if abs_next not in self.q_table:
                    self.q_table[abs_next] = np.zeros(len(self.actions))
                    self.visit_counts[abs_next] = 0
                next_max = self._valor_encolhido(abs_next)

            new_val = (1 - effective_alpha) * old_val + effective_alpha * (reward + self.gamma * next_max)
            # TETO (registado): ver _aplicar_teto.
            new_val = self._aplicar_teto(new_val)
            self.q_table[abs_state][action_idx] = new_val

    # ======================================================================
    # REPLAY ("sonho") — CORREÇÃO (1): sem corpo duplicado
    # ======================================================================

    def replay_experience(self):
        """Consolida a memória. Reserva 25% do lote para a Memória de Elite (PER
        injetado) e distribui os 75% restantes por buckets de visitas/impacto.

        CORREÇÃO (1): o monólito tinha este método com o corpo DUPLICADO — após o
        primeiro loop de updates, redefinia sample_bucket, re-anexava buckets ao
        batch já processado e rodava um SEGUNDO loop, injetando ruído e enviesando
        a Q-table. Aqui o método termina no primeiro (e único) loop de updates.
        """
        if len(self.memory) < self.batch_size:
            return
        # GATE DE CONHECIMENTO: só consolida quando a Q-table tem base razoável.
        # Replaying sobre uma tabela quase vazia só propaga ruído (ver PER, Schaul
        # et al. 2016 — transições sem sinal útil não devem dominar as atualizações).
        if len(self.q_table) < self.replay_min_states:
            return

        # ==============================================================
        # PROPOSTA C: QUOTA DA ELITE, LIMITADA PELO PESO REAL
        # ==============================================================
        # Era `int(self.batch_size * 0.25)` fixo. Ver a nota extensa no construtor:
        # 25% de um conjunto pequeno significa reprocessar os mesmos episodios
        # milhoes de vezes.
        #
        # Agora ha DOIS tectos e vale o MENOR:
        #   1. `elite_quota` (default 0,10), o tecto de desenho
        #   2. o PESO REAL da elite na memoria, `len(elite)/len(memory)`, multiplicado
        #      por um factor de sobre-amostragem
        #
        # O segundo e o que impede a sobre-representacao de forma automatica: se a
        # elite for 0,5% da memoria, ela nao pode ocupar 25% do lote so porque a
        # constante o permitia. O factor 4 da-lhe ainda um peso QUATRO VEZES superior
        # ao que teria por amostragem uniforme — continua a ser PER injetado, e nao
        # amostragem neutra — mas passa a ser um multiplo do peso real em vez de um
        # valor arbitrario.
        FACTOR_SOBREAMOSTRAGEM_ELITE = 4.0
        elite_sample = []
        if len(self.elite_memory) > 0 and len(self.memory) > 0:
            peso_real = len(self.elite_memory) / float(len(self.memory))
            quota = min(self.elite_quota, peso_real * FACTOR_SOBREAMOSTRAGEM_ELITE)
            target_elite = int(self.batch_size * quota)
            take_elite = min(target_elite, len(self.elite_memory))
            if take_elite > 0:
                elite_sample = random.sample(list(self.elite_memory), take_elite)

        target_normal = self.batch_size - len(elite_sample)

        sample_size = min(4000, len(self.memory))
        memory_list = random.sample(list(self.memory), sample_size)

        single_visit_count = 0
        bucket_1_visit = []
        bucket_2_to_4 = []
        bucket_high_reward = []

        for m in memory_list:
            state, action_str, reward, next_state = m
            abs_state = self._get_abstract_state(state)
            visits = self.visit_counts.get(abs_state, 0)
            if visits <= 1:
                single_visit_count += 1
                bucket_1_visit.append(m)
            elif 2 <= visits <= 4:
                bucket_2_to_4.append(m)
            if abs(reward) >= 60.0:
                bucket_high_reward.append(m)

        single_visit_ratio = single_visit_count / len(memory_list)

        # ==============================================================
        # PROPOSTA B: O LIMIAR E RELATIVO AO PROPRIO TREINO
        # ==============================================================
        # Ver a nota extensa no construtor. Em resumo: o limiar absoluto de 0,30 foi
        # calibrado com ~500 mil estados e, com 1,7 milhoes, o ratio fica quase
        # sempre acima dele — o modo de CONSOLIDAR nunca corria.
        #
        # Compara-se agora o ratio com a MEDIA MOVEL dos ultimos 50 valores. Enquanto
        # nao houver amostras que cheguem, usa-se o limiar absoluto, que preserva o
        # arranque do treino (onde o 0,30 estava bem calibrado).
        #
        # O registo e feito DEPOIS da comparacao, para o valor actual nao contaminar
        # a media contra a qual esta a ser comparado.
        if len(self.single_visit_ratio_hist) >= self.single_visit_ratio_min_amostras:
            media = sum(self.single_visit_ratio_hist) / len(self.single_visit_ratio_hist)
            priorizar_frescos = single_visit_ratio > media
        else:
            priorizar_frescos = single_visit_ratio > self.single_visit_ratio_limiar_absoluto
        self.single_visit_ratio_hist.append(single_visit_ratio)

        # Se a novidade esta a ACELERAR, persegue-a; se esta a ABRANDAR, consolida.
        if priorizar_frescos:
            target_1v = int(target_normal * 0.70)
            target_2to4 = int(target_normal * 0.20)
        else:
            target_1v = int(target_normal * 0.20)
            target_2to4 = int(target_normal * 0.60)
        target_high = target_normal - target_1v - target_2to4

        def td_error(m):
            """|δ| = |r + γ·max Q(s') − Q(s,a)| — a 'surpresa' da transição.
            Transições de maior TD-error são as de que o agente mais tem a aprender
            (Schaul et al. 2016). É o sinal que tira o agente de subótimos: uma
            jogada que rendeu muito mais (ou menos) que o esperado empurra a política.
            """
            state, action_str, reward, next_state = m
            abs_s = self._get_abstract_state(state)
            if abs_s not in self.q_table or action_str not in self.actions:
                return 1.0  # transição nova -> prioridade máxima (garante que é vista)
            q_sa = self.q_table[abs_s][self.actions.index(action_str)]
            abs_n = self._get_abstract_state(next_state)
            if abs_n in [("TERMINAL_WIN",), ("TERMINAL_LOSS",)] or abs_n not in self.q_table:
                next_max = 0.0
            else:
                next_max = self._valor_encolhido(abs_n)
            return abs(reward + self.gamma * next_max - q_sa)

        def sample_bucket(bucket, target_size):
            """Amostra do bucket priorizando por TD-error (PER). Mantém estocástico
            (todos têm probabilidade > 0) para não colapsar a diversidade."""
            if not bucket:
                return []
            k = min(target_size, len(bucket))
            if self.td_priority_alpha <= 0.0 or k >= len(bucket):
                return random.sample(bucket, k)
            # prioridade p_i = (|δ_i| + ε)^α ; probabilidade ∝ p_i
            eps = 0.01
            errors = np.array([td_error(m) for m in bucket], dtype=float)
            prios = np.power(errors + eps, self.td_priority_alpha)
            probs = prios / prios.sum()
            idxs = np.random.choice(len(bucket), size=k, replace=False, p=probs)
            return [bucket[i] for i in idxs]

        batch = []
        batch.extend(sample_bucket(bucket_1_visit, target_1v))
        batch.extend(sample_bucket(bucket_2_to_4, target_2to4))
        batch.extend(sample_bucket(bucket_high_reward, target_high))

        missing = target_normal - len(batch)
        if missing > 0:
            batch.extend(random.sample(memory_list, min(missing, len(memory_list))))

        # Funde a memória normal com a elite e embaralha (evita viés de ordem).
        batch.extend(elite_sample)
        random.shuffle(batch)

        # Loop ÚNICO de updates (o segundo, duplicado, foi removido).
        for state, action_str, reward, next_state in batch:
            self._apply_q_update(state, action_str, reward, next_state, is_replay=True)

    # ======================================================================
    # DECISÃO (política epsilon-greedy com prior do instinto + herança _MEC)
    # ======================================================================

    def decide_action(self, state, valid_actions, ranking_list):
        # Reposta a cada decisao; o agente le-a logo a seguir (ver update_feedback).
        self.ultima_foi_exploratoria = False

        # REMOVIDO EM 24/08/2026: aqui existia `is_mec_avail = (state[13] ==
        # "MEC_AVAIL")`, atribuido e NUNCA LIDO. Alem de morto, tinha o indice
        # errado: em shared/state.py o campo de mecanica e o INDICE 7; o 13 e
        # `opp_hazards`, cujos valores sao SET e CLEAR. Se voltar a ser preciso:
        # state[7] == "MEC_AVAIL".

        abs_state = self._get_abstract_state(state)

        # CORREÇÃO (4): np.zeros, consistente com _apply_q_update.
        if abs_state not in self.q_table:
            self.q_table[abs_state] = np.zeros(len(self.actions))
            self.visit_counts[abs_state] = 0

        # Herança de mecânica: a variante _MEC herda o valor da base se ainda for 0.
        for i, act in enumerate(self.actions):
            if act.endswith("_MEC"):
                base = act.replace("_MEC", "")
                if base in self.actions:
                    base_idx = self.actions.index(base)
                    if self.q_table[abs_state][i] == 0.0 and self.q_table[abs_state][base_idx] != 0.0:
                        self.q_table[abs_state][i] = self.q_table[abs_state][base_idx]

        valid_indices = [self.actions.index(a) for a in valid_actions]
        valid_q_values = {idx: self.q_table[abs_state][idx] for idx in valid_indices}

        best_action_idx = max(valid_q_values, key=valid_q_values.get)
        worst_action_idx = min(valid_q_values, key=valid_q_values.get)
        best_q_value = valid_q_values[best_action_idx]
        worst_q_value = valid_q_values[worst_action_idx]

        visits = self.visit_counts.get(abs_state, 0)

        # Expande o ranking do instinto para incluir as variantes _MEC.
        valid_ranked = []
        for intent in ranking_list:
            if intent in valid_actions and intent not in valid_ranked:
                valid_ranked.append(intent)
            mec_intent = f"{intent}_MEC"
            if mec_intent in valid_actions and mec_intent not in valid_ranked:
                valid_ranked.append(mec_intent)
        if not valid_ranked:
            valid_ranked = valid_actions

        # Estado virgem -> segue o mestre (primeira intenção do instinto).
        if visits == 0 or (best_q_value == 0.0 and worst_q_value == 0.0):
            action_idx = self.actions.index(valid_ranked[0])
        else:
            # Epsilon por estado, modulado pela incerteza (spread dos Q-values).
            if visits < 5:
                state_epsilon = min(0.30, self.epsilon * 0.8)
            elif visits < 20:
                state_epsilon = self.epsilon
            else:
                spread = best_q_value - worst_q_value
                max_abs_q = max(abs(best_q_value), abs(worst_q_value), 1.0)
                normalized_spread = spread / max_abs_q
                if best_q_value < 0 and normalized_spread < 0.1:
                    uncertainty_factor = 1.0
                elif best_q_value < 0:
                    uncertainty_factor = 0.5
                else:
                    uncertainty_factor = max(0.1, np.exp(-normalized_spread * 10.0))
                state_epsilon = min(0.5, self.epsilon * uncertainty_factor)

            # Exploração por buckets de ranking (favorece o topo do instinto).
            # A flag ultima_foi_exploratoria e lida pelo agente IMEDIATAMENTE apos
            # esta chamada (sem await pelo meio), para saber se deve cortar o traco.
            if random.random() < state_epsilon:
                self.ultima_foi_exploratoria = True
                r = random.random()
                if r < 0.50 and len(valid_ranked) >= 2:
                    chosen_action = valid_ranked[0] if random.random() < 0.60 else valid_ranked[1]
                elif r < 0.85 and len(valid_ranked) >= 4:
                    chosen_action = valid_ranked[2] if random.random() < 0.60 else valid_ranked[3]
                else:
                    remaining = valid_ranked[4:] if len(valid_ranked) >= 5 else valid_ranked
                    chosen_action = random.choice(remaining) if remaining else valid_ranked[0]
                action_idx = self.actions.index(chosen_action)
            else:
                self.ultima_foi_exploratoria = False
                action_idx = best_action_idx

        # Diagnostico puro: mede a distribuicao de escolhas nos estados maduros.
        self._record_action_choice(abs_state, action_idx, valid_actions)

        chosen_action_str = self.actions[action_idx]
        if chosen_action_str.endswith("_MEC"):
            return (chosen_action_str.replace("_MEC", ""), "ACTIVATE")
        return (chosen_action_str, None)


    def _record_action_choice(self, abs_state, action_idx, valid_actions):
        """Regista N(s,a) apenas depois de o estado estar maduro.

        A medicao NAO participa de reward, PBRS, Q-update, replay ou escolha de acao.
        E diagnostico puro. `action_valid_masks` acumula quais acoes chegaram a ser
        legais no mesmo estado abstrato, permitindo distinguir repertorio disponivel
        de repertorio efetivamente escolhido.
        """
        if self.visit_counts.get(abs_state, 0) < self.LIMIAR_MADURO:
            return
        # `decide_action` pode correr em varias batalhas concorrentes. As contagens
        # diagnosticas usam o mesmo lock da Q-table para nao perder incrementos.
        with self._qtable_lock:
            counts = self.action_choice_counts.get(abs_state)
            if counts is None:
                counts = np.zeros(len(self.actions), dtype=np.uint32)
                self.action_choice_counts[abs_state] = counts
            counts[action_idx] += 1

            mask = self.action_valid_masks.get(abs_state, 0)
            for action in valid_actions:
                try:
                    idx = self.actions.index(action)
                except ValueError:
                    continue
                mask |= (1 << idx)
            self.action_valid_masks[abs_state] = mask

    def metricas_acoes_por_estado(self):
        """Resumo ponderado da diversidade de decisoes em estados maduros medidos."""
        total_dec = 0
        soma_dominante = 0
        soma_entropia = 0.0
        soma_efetivas = 0.0
        soma_cobertura = 0.0
        peso_cobertura = 0
        estados = 0
        for estado, counts in self.action_choice_counts.items():
            n = int(np.sum(counts))
            if n <= 0:
                continue
            estados += 1
            total_dec += n
            nz = counts[counts > 0].astype(float)
            soma_dominante += int(np.max(counts))
            p = nz / n
            h = float(-np.sum(p * np.log(p))) if len(p) else 0.0
            soma_entropia += h * n
            soma_efetivas += float(np.exp(h)) * n

            valid_mask = int(self.action_valid_masks.get(estado, 0))
            validas = valid_mask.bit_count()
            escolhidas = int(np.count_nonzero(counts))
            if validas > 0:
                soma_cobertura += (escolhidas / validas) * n
                peso_cobertura += n

        if total_dec <= 0:
            return {
                "estados_medidos": 0, "decisoes_medidas": 0,
                "acao_dominante_pct": 0.0, "entropia_media": 0.0,
                "acoes_efetivas": 0.0, "cobertura_acoes_pct": 0.0,
            }
        return {
            "estados_medidos": estados,
            "decisoes_medidas": total_dec,
            "acao_dominante_pct": soma_dominante / total_dec * 100.0,
            "entropia_media": soma_entropia / total_dec,
            "acoes_efetivas": soma_efetivas / total_dec,
            "cobertura_acoes_pct": (soma_cobertura / peso_cobertura * 100.0) if peso_cobertura else 0.0,
        }

    # ======================================================================
    # DECAIMENTO DE EPSILON/ALPHA
    # ======================================================================

    def decay_epsilon(self, new_states=0, battles_in_block=500):
        if not self.visit_counts:
            return
        discovery_rate = new_states / max(1, battles_in_block)
        decay_multiplier = min(3.0, 3.0 / max(discovery_rate, 0.1))
        actual_decay = self.epsilon_decay * decay_multiplier

        # PISO DE DECAIMENTO: garante que o epsilon cai pelo menos DECAY_FLOOR por
        # bloco, mesmo quando a descoberta de estados é alta (início do treino). Sem
        # isto, enquanto há muitos estados novos o decay_multiplier fica pequeno e o
        # epsilon quase não desce — o agente fica preso em exploração alta e o WR
        # medido reflete jogo semi-aleatório em vez da política aprendida.
        actual_decay = max(actual_decay, self.decay_floor)

        self.epsilon = max(self.min_epsilon, self.epsilon - actual_decay)

        initial_eps = getattr(self, 'initial_epsilon', 0.40)
        if self.epsilon > self.min_epsilon:
            progress = (self.epsilon - self.min_epsilon) / max(0.01, (initial_eps - self.min_epsilon))
        else:
            progress = 0.0
        self.alpha = max(self.min_alpha, self.min_alpha + (self.initial_alpha - self.min_alpha) * progress)

    # ======================================================================
    # PERSISTÊNCIA
    # ======================================================================

    def _get_root_path(self, filename):
        # Caminhos absolutos sao usados pelos scripts de treino para os snapshots
        # oficial/emergencia em artefatos/brains. Mantemos caminhos relativos
        # compativeis com o comportamento historico (relativos a qlearning/).
        if os.path.isabs(filename):
            return filename
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, filename)

    def save_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        temp_filepath = filepath + ".tmp"
        data = {
            "q_table": self.q_table,
            "visit_counts": self.visit_counts,
            "actions": self.actions,
            "action_choice_counts": self.action_choice_counts,
            "action_valid_masks": self.action_valid_masks,
            # SNAPSHOT DE POLITICA (30/08/2026). Cada sessao de 10k e um PROCESSO
            # novo; sem persistir, o churn saia sempre a -1.0 no primeiro bloco de
            # cada sessao e perdiam-se 40 pontos de medida por ciclo. Sao ~72k
            # tuplos no Blue de 400k, uns poucos MB face aos 165 MB da Q-table.
            "conv_snapshot": getattr(self, "_conv_snapshot", None),
            "conv_estados_ant": getattr(self, "_conv_estados_ant", None),
            "conv_visitas_ant": getattr(self, "_conv_visitas_ant", None),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "episode_reward_max": self.episode_reward_max,
            "episode_reward_min": self.episode_reward_min,
            "elite_memory": self.elite_memory,
            # Distribuicao recente: sem ela, ao retomar uma sessao o percentil da
            # elite recomecava do zero e as primeiras 100 batalhas voltavam a usar
            # so o piso absoluto, poluindo a memoria com episodios medianos.
            "recompensas_recentes": self.recompensas_recentes,
        }
        # Persistencia atomica. O codigo antigo engolia qualquer Exception e o
        # chamador podia acreditar que um checkpoint existia quando a gravacao tinha
        # falhado. Agora o save devolve True/False, imprime a causa e nunca substitui
        # o ultimo ficheiro valido por uma escrita parcial.
        try:
            pasta = os.path.dirname(os.path.abspath(filepath))
            os.makedirs(pasta, exist_ok=True)
            with open(temp_filepath, "wb") as f:
                pickle.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_filepath, filepath)
            return True
        except Exception as e:
            try:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            except OSError:
                pass
            print(f"[CÉREBRO] Erro ao salvar em {filepath}: {e}")
            return False

    def load_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                self.q_table = data.get("q_table", {})
                self.visit_counts = data.get("visit_counts", {})
                # Compatibilidade: cerebros anteriores a 17/09/2026 nao possuem
                # N(s,a); a medicao comeca vazia e nao tenta reconstruir historia.
                self.action_choice_counts = data.get("action_choice_counts", {})
                self.action_valid_masks = data.get("action_valid_masks", {})
                # `.get` com None: um .pkl anterior a 30/08/2026 nao tem estas
                # chaves, e nesse caso o churn arranca a -1.0 no primeiro bloco,
                # que e o comportamento correcto (nao ha com que comparar).
                self._conv_snapshot = data.get("conv_snapshot", None)
                self._conv_estados_ant = data.get("conv_estados_ant", None)
                self._conv_visitas_ant = data.get("conv_visitas_ant", None)
                # ARMADILHA REMOVIDA EM 25/08/2026: aqui comparava-se `current_phase`
                # com a fase gravada e, se diferissem, o epsilon e o alpha guardados
                # eram DESCARTADOS silenciosamente. Um .pkl com outra fase gravada
                # fazia o treino recomecar com epsilon a 0,40 sobre uma tabela madura,
                # arruinando a corrida sem qualquer erro visivel. O epsilon e o alpha
                # sao agora sempre restaurados.
                self.epsilon = data.get("epsilon", self.epsilon)
                self.alpha = data.get("alpha", self.alpha)
                self.episode_reward_max = data.get("episode_reward_max", -9999.0)
                self.episode_reward_min = data.get("episode_reward_min", 9999.0)
                # ==========================================================
                # COMPATIBILIDADE COM .pkl ANTERIORES A 11/09/2026
                # ==========================================================
                # O `deque` guardado no ficheiro traz o `maxlen` COM QUE FOI CRIADO.
                # Atribui-lo directamente faria o Hall da Fama continuar preso em
                # 2.000 mesmo depois de `elite_maxlen` subir para 20.000 — a
                # alteracao existiria no codigo e NAO correria, que e o modo de
                # falha dominante deste projecto.
                #
                # Por isso reconstroi-se sempre o deque com o maxlen ACTUAL,
                # preservando o conteudo. Se o maxlen novo for menor, o `deque`
                # descarta os mais antigos, que e o comportamento correcto.
                _elite = data.get("elite_memory", [])
                self.elite_memory = deque(_elite, maxlen=self.elite_maxlen)
                _rec = data.get("recompensas_recentes", [])
                self.recompensas_recentes = deque(_rec, maxlen=1000)
                # `memory` nao e guardada no .pkl (e volatil por desenho), logo o
                # `memory_maxlen` novo aplica-se sozinho a partir do proximo bloco.
                print(f"[CÉREBRO] Carregado. Estados: {len(self.q_table)} | "
                      f"Elite: {len(self.elite_memory)} | "
                      f"Limiar elite: {self._limiar_elite():.0f}")
                return True
            except Exception as e:
                print(f"[CÉREBRO] Erro ao carregar: {e}")
        print("[CÉREBRO] Iniciando do zero.")
        return False
