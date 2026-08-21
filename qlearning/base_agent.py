"""
qlearning/base_agent.py — base comum aos agentes tabulares (Blue e Green).

Contém TODO o ciclo partilhado por ambos, para garantir que a única diferença entre
o híbrido e o Q-puro é o uso (ou não) do instinto — nunca o estado, o cérebro, o
reward ou o espaço de ações. Essa paridade é o que torna a comparação Blue-vs-Green
cientificamente válida: se divergissem noutra coisa, mediríamos "dois agentes
diferentes", não "o efeito do instinto".

Partilhado (nesta base):
  - StateParser  -> mesma tupla de estado (15 dims)
  - BlueBrain    -> mesma Q-table, mesmo update, mesmo reward
  - InstinctExecutor -> mesma tradução intenção->golpe concreto
  - mesmo espaço de 37 ações abstratas

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


class TabularAgent(Player):
    """Base para agentes de Q-Learning tabular. Não usar diretamente — subclassear."""

    codename = "Base"

    def __init__(self, *args, brain_file="brain.pkl",
                 alpha=0.2, gamma=0.99, epsilon=0.40, min_epsilon=0.05, decay=0.005,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.instinct = build_instinct()
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
                'last_was_exploratory': False,
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
        return self.instinct.executor.get_best_lead(battle)

    def choose_move(self, battle):
        try:
            hist = self._get_history(battle)

            if battle.force_switch or (battle.active_pokemon and battle.active_pokemon.fainted):
                switch = self.instinct.executor.get_post_faint_switch(battle)
                self._learn_from_previous(battle, hist, current_state=None)
                return self.create_order(switch) if switch else self.choose_random_move(battle)

            if not battle.active_pokemon or not battle.opponent_active_pokemon:
                return self.choose_random_move(battle)

            state = self.instinct.parser.get_state(battle)
            self._learn_from_previous(battle, hist, current_state=state)

            # --- Latência de decisão: mede o tempo de decidir + traduzir a ação ---
            _t_dec = time.perf_counter()

            valid_actions, ranking_list = self._get_actions_and_ranking(battle, hist)
            if not valid_actions:
                return self.choose_random_move(battle)

            action_tuple = self.brain.decide_action(state, valid_actions, ranking_list)
            # Lido IMEDIATAMENTE a seguir (sem await pelo meio, logo e seguro mesmo
            # com batalhas concorrentes): indica se esta decisao foi exploratoria.
            foi_exploratoria = getattr(self.brain, "ultima_foi_exploratoria", False)
            base_action, mechanic = action_tuple

            obj = self.instinct.executor.get_best_execution_object(base_action, battle, hist)

            self._decision_time_sum += (time.perf_counter() - _t_dec)
            self._decision_count += 1

            # Antes de sobrescrever last_action, preserva-a como prev_action — a regra
            # anti-Protect-consecutivo no masking lê 'prev_action'. Sem isto, essa
            # regra ficava inerte (a chave nunca era escrita na refatoração).
            hist['prev_action'] = hist.get('last_action')
            hist['state'] = state
            hist['last_action'] = action_tuple
            hist['last_was_exploratory'] = foi_exploratoria

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

    # FASE 1: mecânicas (tera/mega/z/dynamax) DESATIVADAS.
    # Motivo: a validade de Z-move depende do golpe concreto E do item Z que o Pokémon
    # carrega, e determiná-la de forma fiável pela API do poke-env é frágil (o servidor
    # rejeita ordens como "Swords Dance como Z-move"). Para a Fase 1 (eficiência de
    # treino) a mecânica não é essencial e a sua ausência é IGUAL para todos os agentes,
    # mantendo a comparação justa. Reintroduzir mecânica é trabalho para uma fase later.
    ENABLE_MECHANICS = False

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
            if getattr(battle, "can_mega_evolve", False):
                return self.create_order(obj, mega=True)

            # Z-move: SÓ se este golpe tiver versão Z válida.
            if getattr(battle, "can_z_move", False):
                move_supports_z = getattr(obj, "can_z_move", None)
                if move_supports_z is None:
                    # fallback: consulta a lista de golpes Z disponíveis, se existir
                    move_supports_z = move_in("available_z_moves")
                if move_supports_z:
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
        self.brain.save_model(self.brain_file)

    def replay(self):
        self.brain.replay_experience()
