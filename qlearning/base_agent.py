"""
qlearning/base_agent.py — base comum aos agentes tabulares (Blue e Green).

Contém TODO o ciclo partilhado por ambos, para garantir que a única diferença entre
o híbrido e o Q-puro é o uso (ou não) do instinto — nunca o estado, o cérebro, o
reward ou o espaço de ações. Essa paridade é o que torna a comparação Blue-vs-Green
cientificamente válida: se divergissem noutra coisa, mediríamos "dois agentes
diferentes", não "o efeito do instinto".

Partilhado (nesta base):
  - StateParser  -> mesma tupla de estado (16 dims)
  - BlueBrain    -> mesma Q-table, mesmo update, mesmo reward
  - InstinctExecutor -> mesma tradução intenção->golpe concreto
  - mesmo espaço de 36 ações abstratas (19 intenções-base; 17 têm variante _MEC)

Diferença (definida nas subclasses via _get_actions_and_ranking):
  - Blue  (HybridAgent): usa o instinto para podar (mask) e ranquear (prior).
  - Green (PureAgent)  : ignora o instinto; todas as ações legais, sem ranking.

As subclasses implementam APENAS `_get_actions_and_ranking(battle, hist)`.
"""

import time

import numpy as np
from poke_env.player import Player

from instinct import build_instinct
from qlearning.brain import BlueBrain

from shared.mechanics import marcar_uso, mega_valido, z_move_valido



class TabularAgent(Player):
    """Base para agentes de Q-Learning tabular. Não usar diretamente — subclassear."""

    codename = "Base"

    def __init__(self, *args, brain_file="brain.pkl",
                 alpha=0.2, gamma=0.99, epsilon=0.40, min_epsilon=0.05, decay=0.005,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.instinct = build_instinct()
        # Executor substituivel: o Ash troca-o por um ExecutorMinimo. Os agentes que
        # nao o substituem usam o do instinto (comportamento anterior inalterado).
        self.executor = self.instinct.executor
        self.brain = BlueBrain(alpha=alpha, gamma=gamma, epsilon=epsilon,
                               min_epsilon=min_epsilon, decay=decay)
        self.brain_file = brain_file
        self.brain.load_model(brain_file)
        self._history = {}

        # --- Métricas de treino (acumuladas por bloco, lidas e zeradas pelo script) ---
        # Latência de decisão: soma de tempos e nº de decisões, para média por bloco.
        self._decision_time_sum = 0.0
        self._decision_count = 0
        # Margem de vitória (Pokémon do VENCEDOR que sobreviveram) e duração (turnos),
        # registadas no fim de cada batalha. Listas consumidas e limpas por bloco.
        self._win_margins = []
        self._battle_durations = []
        # Batalhas já contabilizadas (para detetar o fim de cada batalha uma só vez).
        self._counted_battles = set()
        # Contador de auto-ties (batalhas terminadas sem vencedor) do bloco atual.
        self._auto_ties = 0
        # Recompensa REAL acumulada no bloco. A coluna Reward antiga usava
        # brain.episode_reward_max, que ficava presa no valor sentinela (-9999) e nao
        # media nada. Aqui somamos as recompensas efetivamente atribuidas por turno.
        self._reward_sum = 0.0
        self._reward_turns = 0

    # ------------------------------------------------------------------
    # A ÚNICA diferença entre Blue e Green vive aqui (subclasses implementam).
    # ------------------------------------------------------------------

    def _get_actions_and_ranking(self, battle, hist):
        """Devolve (valid_actions, ranking_list) para este turno.
        Subclasses definem: Blue usa o instinto; Green usa tudo legal sem ranking."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # ciclo partilhado (idêntico para ambos)
    # ------------------------------------------------------------------

    def _get_history(self, battle):
        tag = battle.battle_tag
        if tag not in self._history:
            self._history[tag] = {
                'state': None, 'last_action': None, 'prev_action': None,
                'last_was_exploratory': False, 'buffs_consecutivos': 0,
                # Chaves de 04/09/2026: ver o bloco de paridade no `choose_move`.
                'wish_turno': None, 'diferido_turno': None,
                'predicao_turno': None, 'predicao_alvo': None,
                'predicao_alvo_anterior': None, 'predicoes_falhadas': 0,
                'last_active_id': None, 'last_opponent_id': None,
                'last_opp_hp': None, 'last_my_hp': None,
                'last_action_was_damage': False,
                'last_phi': None, 'my_fainted': 0, 'opp_fainted': 0,
                'weather_start_turn': 0, 'weather_active_prev': False,
            }
        return self._history[tag]

    def teampreview(self, battle):
        # Ambos usam o mesmo lead: faz parte do "ambiente", não do instinto tático
        # de combate. Manter igual evita enviesar a comparação pela ordem de time.
        return self.executor.get_best_lead(battle)

    def choose_move(self, battle):
        try:
            hist = self._get_history(battle)

            if battle.force_switch or (battle.active_pokemon and battle.active_pokemon.fainted):
                # ==========================================================
                # BUG CORRIGIDO 30/08/2026 — TRANSICAO APRENDIDA A DOBRAR
                # ==========================================================
                # Aqui existia `self._learn_from_previous(battle, hist,
                # current_state=None)`. Este ramo devolve SEM escrever no `hist`, e o
                # `_learn_from_previous` tambem nao limpa `hist['state']` nem
                # `hist['last_action']` (no caminho normal eles sao sobrescritos logo
                # a seguir pela decisao nova; aqui nao ha decisao nova).
                #
                # Resultado: a MESMA transicao era aprendida DUAS VEZES.
                #
                #   turno T    decidimos; hist guarda (s_T, a_T)
                #   turno T+1  o nosso Pokemon desmaiou -> troca forcada
                #              aprende (s_T, a_T) e incrementa visit_counts[s_T]
                #              devolve sem tocar no hist
                #   turno T+1  decisao real, depois da troca
                #              aprende (s_T, a_T) OUTRA VEZ, incrementa OUTRA VEZ
                #
                # MEDIDO nos cerebros de 400k: 15.836.687 visitas contra 13.525.600
                # turnos reais. A diferenca sao 5,78 por batalha, das quais 1 e o
                # update terminal legitimo e 4,78 sao este bug — exatamente o numero
                # de Pokemon nossos que desmaiam numa batalha tipica. O Green da 4,79.
                #
                # O DANO NAO ERA O CONTADOR. A transicao duplicada e precisamente
                # aquela em que o nosso Pokemon morreu, ou seja a de recompensa mais
                # negativa da tabela. Aplicar o update duas vezes equivale a DOBRAR o
                # alfa no sinal mais forte do sistema, o que enviesa os dois agentes
                # para jogo defensivo. As duas recompensas nem sequer eram iguais: a
                # primeira chamada passava `current_state=None`, logo sem valor
                # futuro, e a segunda passava o estado real.
                #
                # Efeito colateral: `_get_abstract_state(None)` criava uma entrada
                # `None` na Q-table (confirmada nos dois .pkl, 36 zeros, 0 visitas).
                # Inofensiva, mas inflacionava `Estados_Q` em 1.
                #
                # PORQUE NAO APRENDER AQUI, EM VEZ DE APRENDER UMA SO VEZ. A proxima
                # decisao real aprende a mesma transicao com o estado que DE FACTO
                # resultou, em vez de com `None`, e a recompensa ja inclui o faint. E
                # a formulacao correcta do TD: um alvo com valor futuro real em vez de
                # zero. Aprender aqui com `next_state=None` deitaria fora o
                # bootstrapping da transicao mais informativa da batalha.
                #
                # A recompensa TERMINAL nao se perde: e aplicada por
                # `_aplicar_update_terminal`, chamado do `_battle_finished_callback`.
                switch = self.executor.get_post_faint_switch(battle)
                return self.create_order(switch) if switch else self.choose_random_move(battle)

            if not battle.active_pokemon or not battle.opponent_active_pokemon:
                return self.choose_random_move(battle)

            state = self.instinct.parser.get_state(battle)
            self._learn_from_previous(battle, hist, current_state=state)

            # --- Latência de decisão: mede o tempo de decidir + traduzir a ação ---
            _t_dec = time.perf_counter()

            valid_actions, ranking_list = self._get_actions_and_ranking(battle, hist)

            # A guarda mecanica anti-ciclo que aqui existia foi REMOVIDA (ver 6.16
            # do ESTADO_DO_PROJETO.md): tratava o sintoma, nao a causa. O ciclo de
            # trocas e resolvido a montante, pela quarentena por matchup em
            # instinct/execution.py. O modulo shared/anti_loop.py NUNCA EXISTIU no
            # projeto: o import estava comentado e o construtor rebentava com
            # `NameError: name 'AntiLoop' is not defined`, abortando qualquer treino
            # a primeira repeticao do orquestrador.
            if not valid_actions:
                return self.choose_random_move(battle)

            action_tuple = self.brain.decide_action(state, valid_actions, ranking_list)
            # Lido IMEDIATAMENTE a seguir (sem await pelo meio, logo e seguro mesmo
            # com batalhas concorrentes): indica se esta decisao foi exploratoria.
            foi_exploratoria = getattr(self.brain, "ultima_foi_exploratoria", False)
            base_action, mechanic = action_tuple

            # `atalho_de_pivo=False`: SWITCH SIGNIFICA SWITCH (03/09/2026).
            #
            # O atalho de pivo do executor devolvia um U-turn ou Volt Switch
            # quando a intencao era `SWITCH_*`. Para o cerebro isso e trocar de
            # accao pelas costas da Q-table: a recompensa ia para `SWITCH_*` num
            # turno em que a accao executada foi na pratica um `ATTACK_PIVOT`, e
            # quando o pivo era imune (Volt Switch contra Terra) nem troca havia
            # — dano zero, sem sair de campo, e a penalizacao registada nas duas
            # accoes de troca. Esteve activo em todos os ciclos ate ao v9/v6.
            #
            # `ATTACK_PIVOT` E UMA ACCAO DO ESPACO, e `physics.classify_move`
            # devolve-a para os cinco golpes de pivo (Teleport incluido). Se sair
            # de campo atacando for a jogada certa, o cerebro aprende a escolhe-la
            # pelo rotulo dela, com a recompensa no sitio certo. Nao se perde
            # repertorio, ganha-se a distincao entre duas accoes.
            #
            # O InstinctBot mantem o atalho (nao tem Q-table para corromper) e
            # por isso o default do parametro e True: nenhum outro chamador muda.
            obj = self.executor.get_best_execution_object(
                base_action, battle, hist, atalho_de_pivo=False)

            self._decision_time_sum += (time.perf_counter() - _t_dec)
            self._decision_count += 1

            # `prev_action` guarda a acao de DOIS turnos atras (last_action antes de
            # ser sobrescrita). Escrita aqui, e ATUALMENTE SEM NENHUM LEITOR.
            #
            # HISTORIA (30/08/2026). Esta chave foi criada para a regra anti-Protect
            # do masking, que lia `prev_action` OU `last_action`. Essa leitura estava
            # ERRADA: banía o Protect por dois turnos em vez de um, e por isso
            # proibia a sequencia Protect -> ataque -> Protect, que tem 100% de
            # sucesso porque o contador zera ao usar outro golpe. Corrigido no
            # instinto v17 (ver 6.38): o masking passou a ler so `last_action`.
            #
            # A chave FICA, de proposito. Remove-la obrigaria a mexer em tres
            # ficheiros (`base_agent`, `instinct_player`, `masking`) e a paridade
            # exata das chaves entre o TabularAgent e o InstinctBot e o que garante
            # que a regua e os agentes tem o MESMO comportamento (6.32). Custa uma
            # atribuicao por turno e evita esse risco.
            #
            # SE ALGUEM VOLTAR A LE-LA: confirmar primeiro que a semantica pretendida
            # e mesmo "dois turnos atras" e nao "o turno anterior". Foi essa confusao
            # que produziu o bug original.
            hist['prev_action'] = hist.get('last_action')
            hist['state'] = state
            hist['last_action'] = action_tuple
            hist['last_was_exploratory'] = foi_exploratoria
            # Contador para a regra da corrida de buffs (ver instinct/policy.py).
            if str(base_action).replace("_MEC", "") == "BUFF":
                hist['buffs_consecutivos'] = int(hist.get('buffs_consecutivos', 0)) + 1
            else:
                hist['buffs_consecutivos'] = 0

            # ==========================================================
            # WISH E PREDITIVO (04/09/2026) — PARIDADE OBRIGATORIA
            # ==========================================================
            # ESTE BLOCO TEM DE SER IDENTICO AO DO `instinct_player`. Duas regras
            # novas leem estas chaves do historico:
            #
            #   `masking`, filtro do Wish  -> `wish_turno`
            #   `policy`, Regra Global 11  -> `predicoes_falhadas`
            #
            # O poke-env NAO expoe o Wish (verificado em 04/09: a string nao
            # existe no `pokemon.py` nem nas `SideCondition`), logo o dado tem de
            # ser nosso. Se so o InstinctBot o escrevesse, o Blue e o Green
            # ficariam com a poda do Wish e a penalizacao da previsao DESLIGADAS,
            # e isso seria uma ASSIMETRIA NAO DECLARADA entre a regua e os
            # agentes — exactamente o que a 6.32 diz que invalida a comparacao.
            # As tres assimetrias legitimas do projeto sao escolhidas por
            # PARAMETRO no executor, nunca por esquecimento no historico.
            try:
                _acao = str(base_action).replace("_MEC", "")
                _opp = getattr(battle.opponent_active_pokemon, "species", None)

                if _acao in ("STATUS", "HEAL") and any(
                        getattr(m, "id", "") == "wish"
                        for m in (getattr(battle, "available_moves", None) or [])):
                    hist['wish_turno'] = battle.turn

                if _acao == "ATTACK_PREDICTIVE":
                    hist['predicao_turno'] = battle.turn
                    hist['predicao_alvo'] = _opp
                    if hist.get('predicao_alvo_anterior') == _opp:
                        hist['predicoes_falhadas'] = int(hist.get('predicoes_falhadas', 0)) + 1
                    else:
                        hist['predicoes_falhadas'] = 0
                    hist['predicao_alvo_anterior'] = _opp
                elif hist.get('predicao_alvo_anterior') != _opp:
                    hist['predicoes_falhadas'] = 0
                    hist['predicao_alvo_anterior'] = None
            except Exception:
                pass

            # GOLPES DE EFEITO DIFERIDO (04/09/2026). Registados pelo GOLPE
            # EXECUTADO e nao pela intencao: a intencao diz o que se queria, o
            # `obj` diz o que saiu. O registo do Wish por intencao (acima) fica,
            # como rede; este e o preciso.
            #
            # PORQUE TEM DE SER NOSSO: Wish, Future Sight e Doom Desire sao SLOT
            # conditions. O poke-env nao as expoe nem em `side_conditions` nem em
            # `effects` (verificado em 04/09: a string "WISH" nao existe no
            # `pokemon.py`). Sem memoria propria, nenhum filtro os pode ver.
            #
            # Observado: `futuresight` em quatro turnos seguidos, com
            # "But it failed!" nos dois ultimos.
            try:
                _gid = getattr(obj, "id", None)
                if _gid == "wish":
                    hist['wish_turno'] = battle.turn
                elif _gid in ("futuresight", "doomdesire"):
                    hist['diferido_turno'] = battle.turn
            except Exception:
                pass

            # --- Instrumentacao para LETALIDADE POR DANO OBSERVADO ---
            # Guarda quem estava em campo e o HP do oponente ANTES da acao. No turno
            # seguinte, se os MESMOS dois Pokemon continuarem em campo e a acao tiver
            # sido de dano, a diferenca de HP e o dano REAL — muito mais fiavel que a
            # estimativa (que desconhece EVs, IVs, item e nature do adversario).
            try:
                hist['last_active_id'] = getattr(battle.active_pokemon, 'species', None)
                hist['last_opponent_id'] = getattr(battle.opponent_active_pokemon, 'species', None)
                hist['last_opp_hp'] = battle.opponent_active_pokemon.current_hp_fraction
                hist['last_my_hp'] = battle.active_pokemon.current_hp_fraction
                hist['last_action_was_damage'] = str(base_action).startswith("ATTACK")
            except Exception:
                hist['last_action_was_damage'] = False
            hist['my_fainted'] = len([m for m in battle.team.values() if m.fainted])
            hist['opp_fainted'] = len([m for m in battle.opponent_team.values() if m.fainted])

            # Regista o turno em que um clima começou (a heurística de troca para
            # abusadores de clima no executor lê 'weather_start_turn').
            if battle.weather:
                if not hist.get('weather_active_prev'):
                    hist['weather_start_turn'] = battle.turn
                hist['weather_active_prev'] = True
            else:
                hist['weather_active_prev'] = False

            if obj:
                if mechanic == "ACTIVATE":
                    return self._order_with_mechanic(obj, battle)
                return self.create_order(obj)
            return self.choose_random_move(battle)

        except Exception:
            # DIAGNÓSTICO: imprime o erro real UMA vez por tipo, para não spammar o log
            # mas também não esconder o problema. Sem isto, um bug aqui manifesta-se
            # como "estados=0 / WR baixo" sem pista da causa.
            self._log_choose_error()
            return self.choose_random_move(battle)

    _seen_errors = None

    def _log_choose_error(self):
        import traceback
        if self._seen_errors is None:
            self._seen_errors = set()
        tb = traceback.format_exc()
        # chave = última linha do traceback (o tipo/mensagem do erro)
        key = tb.strip().splitlines()[-1] if tb.strip() else "?"
        if key not in self._seen_errors:
            self._seen_errors.add(key)
            print("\n[ERRO em choose_move — primeira ocorrência deste tipo]:")
            print(tb)

    def _aplicar_update_terminal(self, battle, tag):
        """Aplica o update da ULTIMA transicao da batalha, com o resultado final.

        Sem isto a recompensa terminal nunca entra na Q-table (ver nota em
        _capture_battle_end_metrics). Corre uma vez por batalha, ja com o resultado
        conhecido.
        """
        hist = self._history.get(tag)
        if not hist:
            return
        if hist.get('state') is None or hist.get('last_action') is None:
            return
        try:
            # Estado terminal convencional do cerebro: Q futuro = 0 e Phi = 0,
            # o que fecha o telescopio do PBRS com F = -Phi(s_anterior).
            if battle.won:
                estado_terminal = ("TERMINAL_WIN",)
            elif battle.lost:
                estado_terminal = ("TERMINAL_LOSS",)
            else:
                # auto-tie: nao ha desfecho, logo nao ha recompensa terminal.
                return

            reward = self.brain.calculate_reward(battle, hist, estado_terminal)

            # Usa o update terminal DEDICADO (traco de episodio, lambda maior, sem
            # corte por exploracao) para o desfecho chegar as jogadas de setup do
            # inicio da batalha. Cai no update normal se o cerebro for uma versao
            # anterior sem esse metodo.
            base_action, mechanic = hist['last_action']
            action_str = f"{base_action}_MEC" if mechanic else base_action
            if hasattr(self.brain, "aplicar_update_terminal") and action_str in self.brain.actions:
                self.brain.aplicar_update_terminal(
                    hist['state'], action_str, reward, tag, venceu=bool(battle.won))
            else:
                self.brain.update_feedback(
                    estado_terminal, hist['state'], hist['last_action'], reward,
                    battle_key=tag,
                    was_exploratory=hist.get('last_was_exploratory', False))

            # Contabiliza na metrica de recompensa do bloco.
            try:
                self._reward_sum += float(reward)
                self._reward_turns += 1
            except (TypeError, ValueError):
                pass

            # Impede que a mesma transicao seja reaplicada se algo voltar aqui.
            hist['last_action'] = None
        except Exception:
            self._log_choose_error()

    def _capture_battle_end_metrics(self, battle):
        """No fim de uma batalha, regista margem de vitória e duração. Chamado uma só
        vez por batalha (guardado por battle_tag). Auto-tie residual (sem vencedor)
        não conta para margem nem para vitória/derrota — apenas duração."""
        tag = battle.battle_tag
        if tag in self._counted_battles:
            return
        self._counted_battles.add(tag)

        # ------------------------------------------------------------------
        # UPDATE TERMINAL (correcao critica)
        # ------------------------------------------------------------------
        # A recompensa de +5000/-5000 era ATRIBUIDA pelo calculate_reward mas NUNCA
        # chegava a Q-table: quem o chama e o _learn_from_previous, que corre dentro
        # do choose_move, e o poke-env deixa de chamar choose_move assim que a
        # batalha acaba. A ultima transicao — precisamente a jogada que ganhou ou
        # perdeu o jogo — ficava sem update.
        #
        # Consequencia medida: a recompensa media por batalha era -173 quando a
        # aritmetica previa ~+2080. O agente aprendeu apenas com +100/-100 por abate
        # e com o shaping; o sinal mais forte do sistema estava desligado.
        #
        # Aqui aplicamos esse update final. O proximo estado e o terminal
        # convencional do cerebro (Q futuro = 0), e com os traces ligados o
        # resultado propaga-se por toda a trajetoria final da batalha.
        self._aplicar_update_terminal(battle, tag)

        # Só DEPOIS do update terminal se descartam os traces desta batalha (o update
        # precisa deles para propagar o desfecho para tras).
        try:
            self.brain.limpar_traces(tag)
        except AttributeError:
            pass

        # Liberta o historico desta batalha (senao o dicionario cresce 1 entrada por
        # batalha e nunca encolhe).
        self._history.pop(tag, None)
        try:
            self.executor.limpar_saidas(tag)
        except AttributeError:
            pass
        try:
            # REGRA 7 (impasse): o historico de posicoes desta batalha. Sem esta
            # limpeza o dicionario cresce uma entrada por batalha e nunca encolhe —
            # em 400.000 batalhas seria fuga de memoria.
            self.instinct.policy.limpar_posicoes(tag)
        except AttributeError:
            pass

        # Duração: nº de turnos até a decisão.
        self._battle_durations.append(getattr(battle, "turn", 0))

        # Margem: Pokémon do VENCEDOR que sobreviveram (só se houve vencedor claro).
        if battle.won:
            survivors = len([m for m in battle.team.values() if not m.fainted])
            self._win_margins.append(survivors)
        elif battle.lost:
            survivors = len([m for m in battle.opponent_team.values() if not m.fainted])
            self._win_margins.append(survivors)
        else:
            # auto-tie (finished mas nem won nem lost): não gera recompensa terminal
            # nem margem. Só contamos quantas ocorreram, para reportar no log.
            self._auto_ties += 1

    def _varrer_batalhas_terminadas(self):
        """Percorre as batalhas conhecidas e captura as metricas das que terminaram.

        PORQUE E NECESSARIO: quando uma batalha termina, o poke-env deixa de chamar
        choose_move, logo uma captura feita dentro do ciclo de decisao nunca ve o
        estado final (era o bug: margem=0.0 e dur=0t em todos os blocos). Sobrepor
        _battle_finished_callback tambem nao serve: interfere com o ciclo de vida da
        batalha e degradava a win rate.

        Aqui e seguro: o script chama pop_block_metrics() SEMPRE antes de
        reset_battles(), portanto neste momento as batalhas do bloco ainda existem e
        ja estao terminadas.
        """
        batalhas = getattr(self, "battles", None)
        if batalhas is None:
            batalhas = getattr(self, "_battles", {}) or {}
        try:
            for battle in list(batalhas.values()):
                if getattr(battle, "finished", False):
                    self._capture_battle_end_metrics(battle)
        except Exception:
            pass

    def pop_block_metrics(self):
        """Devolve as métricas acumuladas desde a última chamada e ZERA os
        acumuladores. O script de treino chama isto uma vez por bloco (1k batalhas).

        Retorna dict com: latencia_ms (média), margem_media, duracao_media,
        n_margens (nº de batalhas com vencedor claro no bloco)."""
        # Captura as metricas das batalhas TERMINADAS antes de calcular as medias.
        self._varrer_batalhas_terminadas()

        lat_ms = (self._decision_time_sum / self._decision_count * 1000.0) if self._decision_count else 0.0
        margem = (sum(self._win_margins) / len(self._win_margins)) if self._win_margins else 0.0
        duracao = (sum(self._battle_durations) / len(self._battle_durations)) if self._battle_durations else 0.0
        n_margens = len(self._win_margins)
        auto_ties = self._auto_ties
        # Recompensa media por TURNO e por BATALHA no bloco.
        reward_turno = (self._reward_sum / self._reward_turns) if self._reward_turns else 0.0
        n_batalhas = len(self._battle_durations)
        reward_batalha = (self._reward_sum / n_batalhas) if n_batalhas else 0.0

        # Zera para o próximo bloco.
        self._decision_time_sum = 0.0
        self._decision_count = 0
        self._win_margins = []
        self._battle_durations = []
        self._counted_battles = set()
        self._auto_ties = 0
        self._reward_sum = 0.0
        self._reward_turns = 0

        return {"latencia_ms": lat_ms, "margem_media": margem,
                "duracao_media": duracao, "n_margens": n_margens,
                "auto_ties": auto_ties, "reward_turno": reward_turno,
                "reward_batalha": reward_batalha}

    def _learn_from_previous(self, battle, hist, current_state):
        if hist.get('state') is not None and hist.get('last_action') is not None:
            reward = self.brain.calculate_reward(battle, hist, current_state)
            # battle_key: os eligibility traces sao POR BATALHA (ha 3 concorrentes).
            # was_exploratory: lida do cerebro logo apos decide_action, para o corte
            # off-policy do Q(lambda) de Watkins.
            self.brain.update_feedback(
                current_state, hist['state'], hist['last_action'], reward,
                battle_key=battle.battle_tag,
                was_exploratory=hist.get('last_was_exploratory', False))
            # Acumula a recompensa real do turno para a metrica do bloco.
            try:
                self._reward_sum += float(reward)
                self._reward_turns += 1
            except (TypeError, ValueError):
                pass
        # Se a batalha terminou, captura as métricas de fim (uma vez por tag). Feito
        # aqui, no fluxo normal, em vez de sobrepor o _battle_finished_callback do
        # poke-env — sobrepor esse hook interferia com o ciclo de vida da batalha.
        if getattr(battle, "finished", False):
            self._capture_battle_end_metrics(battle)

    # ------------------------------------------------------------------
    # helpers de ação partilhados
    # ------------------------------------------------------------------

    # FASE 2 (24/08/2026): mecânicas LIGADAS. Ver 6.19 do ESTADO_DO_PROJETO.md.
    #
    # Cobre MEGA EVOLUÇÃO e Z-MOVE. A Terastalização está BANIDA no gen9nationaldex
    # por Terastal Clause, logo `battle.can_tera` é sempre falso e o ramo tera do
    # _order_with_mechanic nunca dispara neste formato.
    #
    # Porque se liga agora: sem mecânica, a validação cobria um subconjunto do
    # formato. A Mega altera tipagem, stats, velocidade e por vezes o eixo
    # físico/especial, tocando em 5 das 15 dimensões do estado. Testar a abstração
    # sem ela era testá-la em condições mais fáceis do que a realidade do formato.
    #
    # Custo assumido: o espaço de ações passa de 19 para 36 quando há mecânica
    # disponível. Mitigado pela HERANÇA _MEC em brain.decide_action, onde a variante
    # _MEC arranca com o valor já aprendido pela ação base em vez de zero.
    #
    # A fragilidade original do Z-move continua tratada: o _order_with_mechanic
    # verifica golpe a golpe se existe versão Z e, na dúvida, joga sem mecânica.
    ENABLE_MECHANICS = True

    def _expand_with_mechanic(self, categories, battle):
        """Filtra as categorias para o espaço de ações do cérebro.

        Com ENABLE_MECHANICS=False, NUNCA adiciona variantes _MEC — o cérebro não
        vê a mecânica, logo não pode escolher uma jogada que o servidor rejeitaria.
        """
        valid = []
        for cat in categories:
            if cat in self.brain.actions and cat not in valid:
                valid.append(cat)
            if self.ENABLE_MECHANICS:
                mec_avail = (self.instinct.parser.get_mechanic_state(battle) == "MEC_AVAIL")
                if mec_avail and "SWITCH" not in cat:
                    mec = f"{cat}_MEC"
                    if mec in self.brain.actions and mec not in valid:
                        valid.append(mec)
        return valid

    def _order_with_mechanic(self, obj, battle):
        """Ativa uma mecânica APENAS se o golpe concreto a suporta.

        Não basta a mecânica estar disponível na batalha (battle.can_z_move etc.): o
        golpe específico tem de ser compatível. Ex.: Landorus pode ter Z-move
        disponível, mas Swords Dance só é Z-válido se o item Z for do tipo certo — o
        servidor rejeita "Can't use X as a Z-move". Verificamos golpe a golpe e, se a
        mecânica não servir para este golpe, jogamo-lo normalmente (sem mecânica).

        A verificação de compatibilidade usa as listas que o poke-env fornece por
        mecânica quando disponíveis; se a lista não existir, cai no comportamento
        seguro (jogar sem mecânica) em vez de arriscar uma ordem inválida.
        """
        # Só faz sentido para objetos que são Move (trocas não usam mecânica).
        is_move = hasattr(obj, "id") and hasattr(obj, "base_power")
        if not is_move:
            return self.create_order(obj)

        def move_in(list_name):
            lst = getattr(battle, list_name, None)
            if not lst:
                return False
            try:
                return obj in lst or any(getattr(m, "id", None) == obj.id for m in lst)
            except TypeError:
                return False

        try:
            # Terastallize: aplica-se a qualquer golpe quando disponível.
            if getattr(battle, "can_tera", False):
                return self.create_order(obj, terastallize=True)

            # Mega evolução: propriedade do Pokémon, não do golpe — disponível = ok.
            if mega_valido(battle):
                # marcar ANTES de devolver: desde poke-env 0.12.0 o item permanece
                # depois do uso, logo deixou de servir de guarda de uso unico.
                marcar_uso(battle, "mega")
                return self.create_order(obj, mega=True)

            # Z-move: SÓ se este Pokémon carregar o cristal do tipo certo.
            # CORRIGIDO 24/08/2026: usava obj.can_z_move, que é propriedade do golpe
            # nos dados e não da situação. Produzia "[Invalid choice]" e custava o
            # turno. Ver z_move_valido no topo do ficheiro.
            if z_move_valido(obj, battle):
                marcar_uso(battle, "z")
                return self.create_order(obj, z_move=True)

            # Dynamax: aplica-se a qualquer golpe quando disponível.
            if getattr(battle, "can_dynamax", False):
                return self.create_order(obj, dynamax=True)
        except Exception:
            pass

        # Fallback seguro: joga o golpe sem mecânica.
        return self.create_order(obj)

    # ------------------------------------------------------------------
    # persistência
    # ------------------------------------------------------------------

    def save_brain(self):
        """Persiste o brain e devolve True/False para o chamador poder validar I/O."""
        return self.brain.save_model(self.brain_file)

    def replay(self):
        self.brain.replay_experience()
