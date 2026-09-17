"""
Camada de Execução do Instinto (InstinctExecutor).

Separada da política de propósito: a POLICY decide a INTENÇÃO (uma categoria como
"ATTACK_STRONG"); o EXECUTOR converte essa intenção num OBJETO concreto de golpe ou
troca do poke-env, aplicando heurísticas de desempate dentro da categoria escolhida.

Divisão de responsabilidades:
- policy.py    : "o que quero fazer?"  -> string de categoria
- execution.py : "com que golpe/troca concreta?" -> objeto do poke-env

Responsabilidades deste módulo:
- get_best_lead: ordem de time inicial (/team ...).
- get_defensive_switch / get_offensive_switch / get_post_faint_switch: escolhe o
  melhor Pokémon do banco por pontuação de sobrevivência/ofensiva.
- _select_best_move_in_category: desempate entre golpes da mesma categoria.
- get_best_execution_object: o ponto de entrada — recebe a intenção e devolve o
  objeto final, com toda a lógica de ataque (recoil, self-drop, predict, prioridade).

DEPENDÊNCIAS (por injeção): GamePhysics, StateParser, ActionMasker.
"""

from shared import diagnostico
from shared.definitions import Role, MatchupState, MoveCategory


class InstinctExecutor:
    """Converte intenções táticas em objetos concretos de ação do poke-env."""

    def __init__(self, physics, parser, masker):
        self.physics = physics
        self.parser = parser
        self.masker = masker

    # ======================================================================
    # LEAD: ordem de time inicial
    # ======================================================================

    # ==================================================================
    # CLIMA NO TEAM PREVIEW (29/08/2026)
    # ==================================================================
    # Habilidades que poem clima A ENTRAR EM CAMPO. Sand Spit e Ice Face ficam de
    # fora: ativam ao SER ATINGIDO, nao a entrada, logo nao contam para a guerra de
    # clima do turno 1.
    HABILIDADES_DE_CLIMA = {
        'drought': 'sol', 'drizzle': 'chuva', 'sandstream': 'areia',
        'snowwarning': 'granizo', 'orichalcumpulse': 'sol',
        'desolateland': 'sol', 'primordialsea': 'chuva',
    }

    # Especies que poem clima. NECESSARIO porque no TEAM PREVIEW as HABILIDADES do
    # adversario sao desconhecidas — mas as ESPECIES sao visiveis. Sem esta lista, o
    # `opp_has_weather` era quase sempre falso mesmo contra times de clima, e a
    # arvore do clima nunca disparava.
    #
    # Inclui as formas Mega que ganham a habilidade ao evoluir (Charizard-Y sol,
    # Abomasnow granizo, Tyranitar areia).
    ESPECIES_DE_CLIMA = {
        'torkoal', 'ninetales', 'charizard', 'charizardmegay', 'groudon',
        'politoed', 'pelipper', 'kyogre',
        'tyranitar', 'tyranitarmega', 'hippowdon', 'gigalith', 'hippopotas',
        'ninetalesalola', 'abomasnow', 'abomasnowmega', 'vanilluxe', 'snover',
    }

    @staticmethod
    def _velocidade_base(mon):
        try:
            return mon.base_stats.get('spe', 0)
        except Exception:
            return 0

    def _setters_de_clima(self, equipa, por_especie=False):
        """Membros da equipa que poem clima ao entrar."""
        out = []
        for m in equipa:
            if por_especie:
                nome = str(getattr(m, "species", "")).lower().replace("-", "").replace(" ", "")
                if nome in self.ESPECIES_DE_CLIMA:
                    out.append(m)
            else:
                if str(getattr(m, "ability", "")).lower() in self.HABILIDADES_DE_CLIMA:
                    out.append(m)
        return out

    def _equipa_abusa_de_clima(self, equipa):
        """A equipa tem quem TIRE PARTIDO do clima? E isto que justifica leva-lo."""
        abusadores = {'swiftswim', 'chlorophyll', 'sandrush', 'slushrush',
                      'sandforce', 'solarpower', 'leafguard', 'icebody',
                      'raindish', 'dryskin', 'hydration', 'sandveil', 'snowcloak'}
        for m in equipa:
            if str(getattr(m, "ability", "")).lower() in abusadores:
                return True
            for mv in getattr(m, "moves", {}).values():
                if getattr(mv, "id", "") in ('solarbeam', 'solarblade', 'weatherball',
                                             'auroraveil', 'thunder', 'hurricane',
                                             'blizzard', 'growth'):
                    return True
        return False

    # ======================================================================
    # O LEAD SOBREVIVE A EQUIPA DO OUTRO LADO? (03/09/2026)
    # ======================================================================
    # O `get_best_lead` tinha cinco arvores e NENHUMA olhava para o adversario,
    # tirando a comparacao de velocidade da guerra de clima. No formato usado os
    # DOIS times sao revelados no arranque: a informacao existia e estava a ser
    # deitada fora.
    #
    # E ha uma tensao interna que isto arbitra. A Arvore 1 escolhe o setter MAIS
    # LENTO de proposito, porque na entrada simultanea as habilidades activam por
    # velocidade decrescente e o clima do mais lento e o que fica. Mas ser o mais
    # lento e exactamente o perfil que COME O PRIMEIRO GOLPE SEM RESPONDER.
    # A regra que maximiza a hipotese de ganhar a guerra de clima maximiza a
    # hipotese de perder o setter no turno 1.
    #
    # A REGRA DO SETTER MAIS LENTO MANTEM-SE E TEM PRECEDENCIA. Esta verificacao
    # e SUBORDINADA: nao escolhe o lead, so DESQUALIFICA um candidato que morre
    # de vez. Se desqualificar, cai-se para a arvore seguinte.
    #
    # O criterio e grosseiro de proposito. No team preview nao ha item, EVs,
    # nature nem habilidade do adversario, e os `moves` vem quase sempre vazios,
    # logo `physics.sobrevive_a` nao tem dados e devolveria None. Usa-se o que
    # EXISTE no preview: efectividade de TIPO da equipa adversaria contra o
    # candidato. Um candidato que leva 4x de metade da equipa deles nao lidera.
    LIMIAR_LEAD_INVIAVEL = 0.50
    MULT_LEAD_PERIGOSO = 3.9      # 4x com folga para erro de virgula flutuante

    def _lead_e_viavel(self, candidato, opp_team):
        """Fraccao da equipa adversaria que bate no candidato com 4x ou mais."""
        try:
            if not candidato or not opp_team:
                return True
            perigosos = 0
            contados = 0
            for adversario in opp_team:
                tipos = [t for t in getattr(adversario, "types", []) or [] if t]
                if not tipos:
                    continue
                contados += 1
                # Melhor multiplicador que ele consegue so com STAB de tipo.
                pior = max((candidato.damage_multiplier(t) for t in tipos),
                           default=1.0)
                if pior >= self.MULT_LEAD_PERIGOSO:
                    perigosos += 1
            if not contados:
                return True
            return (perigosos / contados) < self.LIMIAR_LEAD_INVIAVEL
        except Exception:
            # Na duvida o lead e viavel: esta verificacao so DESQUALIFICA, e
            # falhar nela nao pode custar a heuristica inteira.
            return True

    def get_best_lead(self, battle):
        try:
            my_team = list(battle.team.values())
            opp_team = list(battle.opponent_team.values())
            if not opp_team:
                return "/team 123456"
            best_lead = None

            # ==========================================================
            # Arvore 0: TRICK ROOM  (29/08/2026)
            # ==========================================================
            # Passa a frente do clima porque a dependencia e maior: NAO HA HABILIDADE
            # que ative Trick Room. Um time de Trick Room sem Trick Room e so um time
            # lento, e o setter tem de entrar em campo e gastar um turno para o por.
            # Lidera-lo e a forma de o ter ativo desde o turno 1.
            #
            # Entre varios setters escolhe-se o MAIS LENTO: e ele que mais beneficia
            # da inversao de velocidade, e o que menos perde se o Trick Room acabar.
            setters_tr = [m for m in my_team if self._sabe_trick_room(m)]
            if setters_tr:
                # VIABILIDADE (03/09/2026): entre os setters de Trick Room
                # prefere-se o mais lento, mas nao um que morra de vez ao lead
                # provavel. Se nenhum for viavel, mantem-se o mais lento e a
                # heuristica antiga: e melhor um Trick Room arriscado do que
                # nenhum, porque sem ele a equipa lenta joga sem plano.
                viaveis_tr = [m for m in setters_tr
                              if self._lead_e_viavel(m, opp_team)]
                best_lead = min(viaveis_tr or setters_tr, key=self._velocidade_base)

            # ==========================================================
            # Arvore 1: CLIMA  (reescrita em 29/08/2026)
            # ==========================================================
            # A versao anterior exigia `my_weather_setter AND opp_has_weather`, e
            # tinha dois defeitos que se somavam:
            #
            #   1. O CRITERIO ESTAVA INVERTIDO. O que justifica levar o setter a
            #      frente nao e o adversario ter clima — e a NOSSA equipa saber
            #      aproveitar o nosso. Com um Ninetales e dois abusadores de sol,
            #      lidera-lo e certo mesmo contra um time sem clima.
            #   2. O `opp_has_weather` lia `m.ability` do ADVERSARIO, que no team
            #      preview e quase sempre desconhecida. Era quase sempre falso, mesmo
            #      em espelhos de clima.
            #
            # Resultado observado: um time com Ninetales liderou com Infernape (pela
            # Arvore 3, pivo rapido) e usou U-turn no turno 1 em vez de aproveitar o
            # sol.
            #
            # GUERRA DE CLIMA — quem e mais LENTO ganha. Na entrada simultanea do
            # turno 1 as habilidades ativam por ordem de velocidade DECRESCENTE, logo
            # o setter mais lento ativa POR ULTIMO e o clima dele e o que fica. Por
            # isso escolhe-se o NOSSO setter mais lento, e desiste-se da arvore se o
            # adversario tiver um setter ainda mais lento sem que tenhamos abusadores
            # que justifiquem gastar o lead na mesma.
            meus_setters = self._setters_de_clima(my_team)
            if not best_lead and meus_setters:
                # Adversario: por ESPECIE (a habilidade nao e conhecida aqui).
                setters_adv = self._setters_de_clima(opp_team, por_especie=True)
                abusamos = self._equipa_abusa_de_clima(my_team)

                # O MAIS LENTO DOS NOSSOS VENCE A GUERRA — regra mantida, e tem
                # precedencia. A viabilidade so DESQUALIFICA: entre os nossos
                # setters escolhe-se o mais lento DE ENTRE OS QUE SOBREVIVEM ao
                # que o outro lado mostrou no preview. Se nenhum sobreviver,
                # volta-se ao mais lento e deixa-se a decisao as arvores
                # seguintes atraves do `perdemos_a_guerra`.
                setters_viaveis = [m for m in meus_setters
                                   if self._lead_e_viavel(m, opp_team)]
                candidato = min(setters_viaveis or meus_setters,
                                key=self._velocidade_base)
                # A guerra de clima compara-se com o candidato JA escolhido, para
                # a desqualificacao nao inverter a leitura de quem ganha o clima.
                perdemos_a_guerra = bool(setters_adv) and any(
                    self._velocidade_base(o) < self._velocidade_base(candidato)
                    for o in setters_adv)
                # Sem nenhum setter viavel, nao se gasta o lead na guerra de
                # clima: cai-se para as arvores de hazard e de pivo.
                if not setters_viaveis:
                    meus_setters = []

                if meus_setters and (abusamos or setters_adv):
                    # Perder a guerra so desqualifica se nao houver abusadores: com
                    # abusadores, por o clima e util mesmo que seja sobreposto depois.
                    if not (perdemos_a_guerra and not abusamos):
                        best_lead = candidato

            # Árvore 2: hazard setter dedicado (sash + hazard ou speed alta)
            if not best_lead:
                for m in my_team:
                    has_hazard = any(mv.id in ['stealthrock', 'spikes', 'stickyweb'] for mv in m.moves.values())
                    fast_or_sash = str(m.item) == 'focussash' or m.base_stats.get('spe', 0) > 105
                    if has_hazard and fast_or_sash:
                        best_lead = m
                        break

            # Árvore 3: fast pivot (momentum no turno 1)
            if not best_lead:
                pivots = [m for m in my_team if any(mv.id in ['uturn', 'voltswitch', 'flipturn'] for mv in m.moves.values())]
                if pivots:
                    best_lead = max(pivots, key=lambda m: m.base_stats.get('spe', 0))

            # Árvore 4: fallback por arquétipo do time
            if not best_lead:
                avg_speed = sum(m.base_stats.get('spe', 50) for m in my_team) / len(my_team)
                if avg_speed > 85:
                    best_lead = max(my_team, key=lambda m: m.base_stats.get('spe', 50))
                else:
                    best_lead = max(my_team, key=lambda m: m.base_stats.get('hp', 50) + m.base_stats.get('def', 50) + m.base_stats.get('spd', 50))

            try:
                lead_index = my_team.index(best_lead) + 1
            except ValueError:
                lead_index = 1
            rest_indices = [str(i + 1) for i in range(len(my_team)) if i + 1 != lead_index]
            team_order = str(lead_index) + "".join(rest_indices)
            return f"/team {team_order}"
        except Exception:
            return "/team 123456"

    # ======================================================================
    # SWITCHES: escolha do melhor Pokémon do banco
    # ======================================================================

    # ==================================================================
    # ANTI-CICLO DE TROCAS: quarentena por MATCHUP
    # ==================================================================
    # Guardado DENTRO do executor e nao no historico do agente porque o
    # InstinctBot chama o executor sem passar historico. Assim os quatro
    # (Blue, Green, Ash e InstinctBot) ficam protegidos pelo mesmo mecanismo.
    #
    # O QUE SE REGISTA: o PAR (quem saiu, contra quem estava). Nao basta marcar o
    # Pokemon que saiu, porque se o adversario tambem trocou o confronto e OUTRO e
    # regressar passa a ser legitimo — pode ate ser a jogada certa.
    #
    #   Sylveon sai de campo contra Mantine   -> marca o par (Sylveon, Mantine)
    #   turno seguinte, ainda Mantine          -> voltar ao Sylveon e o CICLO: penaliza
    #   turno seguinte, agora Garchomp         -> par diferente: PERMITE
    #
    # Bloqueia a repeticao do confronto, nao a movimentacao do Pokemon.

    TURNOS_DE_QUARENTENA = 2
    # Penalizacao por recriar um confronto recente. Tem de ser GRANDE, nao um
    # desconto: com epsilon=0 a politica e determinista, logo uma penalizacao
    # pequena continuaria a permitir o ciclo se o candidato fosse o melhor por
    # margem suficiente. A melhor pontuacao possivel sem penalizacao ronda os 1.700.
    PENALIZACAO_REGRESSO = 5000.0

    @staticmethod
    def _especie(mon):
        return getattr(mon, "species", None) if mon is not None else None

    def _registar_saida(self, battle):
        """Marca o PAR (quem sai, contra quem) e o turno em que saiu."""
        if not hasattr(self, "_saidas"):
            self._saidas = {}
        minha = self._especie(getattr(battle, "active_pokemon", None))
        adversaria = self._especie(getattr(battle, "opponent_active_pokemon", None))
        if minha is None:
            return
        tag = getattr(battle, "battle_tag", None)
        self._saidas.setdefault(tag, {})[(minha, adversaria)] = battle.turn
        # SONDA [QUAR] (03/09/2026) — TEMPORARIA. Ver o par no
        # `_penalizacao_quarentena`. Hipoteses que estas duas linhas separam:
        # `battle.turn` diferente do assumido, `_especie` a divergir entre o
        # activo e o mesmo Pokemon na lista de trocas (Ninetales-Alola e o
        # candidato obvio), `tag` a mudar entre turnos, ou a troca a nao passar
        # por aqui de todo.
        diagnostico.log("QUAR", f"SAIDA turno={battle.turn} tag={tag} "
                                f"par=({minha!r},{adversaria!r}) "
                                f"registo={self._saidas.get(tag)!r}")


    # ==================================================================
    # TRICK ROOM NA ESCOLHA DE TROCA (29/08/2026)
    # ==================================================================
    # Sem Trick Room ativo, uma equipa de Trick Room e apenas uma equipa lenta: cada
    # turno sem ele e um turno em desvantagem. E como NAO HA HABILIDADE que o ative,
    # o setter tem de entrar em campo e gastar um turno — ninguem o faz por ele.
    #
    # Por isso quem sabe Trick Room ganha preferencia na troca enquanto ele nao
    # estiver ativo, e perde-a assim que estiver (ai o setter ja fez o seu trabalho e
    # a troca deve seguir os criterios normais).

    BONUS_SETTER_TRICK_ROOM = 400

    @staticmethod
    def _sabe_trick_room(mon):
        for mv in getattr(mon, "moves", {}).values():
            if getattr(mv, "id", "") == "trickroom":
                return True
        return False

    def _bonus_trick_room(self, candidate, battle):
        try:
            # MIGRADO PARA `physics.tem` EM 03/09/2026: o Trick Room nunca era
            # detectado, logo o bonus ao setter era dado mesmo com ele JA activo.
            if self.physics.tem(battle.fields, "TRICK_ROOM"):
                return 0
            equipa_tr = any(self._sabe_trick_room(m)
                            for m in battle.team.values()
                            if not getattr(m, "fainted", False))
            if equipa_tr and self._sabe_trick_room(candidate):
                return self.BONUS_SETTER_TRICK_ROOM
        except Exception:
            pass
        return 0

    def _penalizacao_quarentena(self, candidate, battle):
        """Desconto se entrar este candidato RECRIA um confronto recente.

        CAUSA QUE ISTO CORRIGE: a pontuacao de troca avaliava so o estado ATUAL
        (HP, papel, resistencias). O Pokemon que acabou de sair continuava a ser o
        melhor pontuado, porque nada mudou nele — e o agente trocava de volta. Com
        A->B->A o estado abstrato repete-se e nem o instinto nem o Q-Learning
        conseguiam ver que estavam presos. Batalhas chegavam ao turno 1000.
        """
        tag = getattr(battle, "battle_tag", None)
        saidas = getattr(self, "_saidas", {}).get(tag, {})
        if not saidas:
            return 0.0
        adversaria_agora = self._especie(getattr(battle, "opponent_active_pokemon", None))
        par = (self._especie(candidate), adversaria_agora)
        turno_saida = saidas.get(par)
        # SONDA [QUAR] (03/09/2026) — TEMPORARIA. O par que aqui se PROCURA tem de
        # ser identico ao que foi REGISTADO em `_registar_saida`. Se as duas
        # linhas do log mostrarem pares diferentes para o mesmo Pokemon, a causa
        # e a normalizacao de especie; se mostrarem o mesmo par e a penalizacao
        # nao aparecer, a causa e a aritmetica dos turnos ou o `tag`.
        diagnostico.log("QUAR", f"CONSULTA turno={battle.turn} tag={tag} "
                                f"par={par!r} turno_saida={turno_saida!r} "
                                f"registo={saidas!r}")
        if turno_saida is None:
            return 0.0          # confronto diferente: regressar e legitimo
        if (battle.turn - turno_saida) <= self.TURNOS_DE_QUARENTENA:
            return self.PENALIZACAO_REGRESSO
        return 0.0

    def limpar_saidas(self, battle_tag):
        """Descarta o registo de uma batalha terminada."""
        if hasattr(self, "_saidas"):
            self._saidas.pop(battle_tag, None)

    def get_defensive_switch(self, battle, history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not candidates:
            return None
        opp_types_obj = [t for t in opponent.types if t] if opponent else []
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0] if opponent else []
        opp_is_physical = self.physics._is_physical(opponent) if opponent else True

        def get_score(candidate):
            hazard_dmg = self.physics_get_hazard_damage(candidate, battle)
            hp_frac = candidate.current_hp_fraction
            if hp_frac <= hazard_dmg + 0.05:
                return -9999
            # ANTI-CICLO: penaliza regressar a um Pokemon que saiu de campo ha
            # <= TURNOS_DE_QUARENTENA turnos (ver _penalizacao_quarentena).
            score = -self._penalizacao_quarentena(candidate, battle)
            score += self._bonus_trick_room(candidate, battle)
            if hp_frac >= 0.7:
                score += 200
            elif hp_frac >= 0.35:
                score += 100
            else:
                score += 50
            if self.physics.get_role(candidate) == Role.TANK:
                score += 100
            if opponent:
                # MIGRADO PARA `physics.mais_rapido` EM 30/08/2026: a comparacao crua
                # ignorava clima, terreno, Tailwind e a INVERSAO do Trick Room.
                opp_e_mais_rapido = self.physics.mais_rapido(opponent, candidate, battle)
                cand_e_mais_rapido = self.physics.mais_rapido(candidate, opponent, battle)
                has_weakness = False
                for opp_type in opp_types_obj:
                    mult = candidate.damage_multiplier(opp_type)
                    if mult > 1.0:
                        score -= 150 * mult
                        has_weakness = True
                    elif mult < 1.0:
                        score += min(300, 50 / max(mult, 0.1))
                for move in known_opp_moves:
                    mult = candidate.damage_multiplier(move)
                    if mult > 1.0:
                        score -= 200 * mult
                        has_weakness = True
                    elif mult < 1.0:
                        score += min(300, 75 / max(mult, 0.1))
                if opp_e_mais_rapido and has_weakness:
                    score -= 300
                cand_abi = str(candidate.ability).lower() if candidate.ability else ""
                type_absorb_map = {
                    'WATER': ['waterabsorb', 'dryskin', 'stormdrain'], 'GROUND': ['levitate'],
                    'GRASS': ['sapsipper'], 'FIRE': ['flashfire'], 'ELECTRIC': ['voltabsorb', 'lightningrod'],
                }
                for t_str in [t.name for t in opp_types_obj]:
                    if t_str in type_absorb_map and cand_abi in type_absorb_map[t_str]:
                        score += 500
                cand_matchup = self.parser.get_matchup_state(candidate, opponent)
                if cand_matchup == MatchupState.DOMINANT:
                    score += 300
                elif cand_matchup == MatchupState.DEFENSIVE_ADV:
                    score += 200
                elif cand_matchup == MatchupState.STALEMATE:
                    score += 100
                elif cand_matchup == MatchupState.NEUTRAL:
                    score += 50
                elif cand_matchup == MatchupState.DEFENSIVE_DIS:
                    score -= 150
                elif cand_matchup == MatchupState.CRITICAL_DIS:
                    score -= 300
                cand_def = candidate.base_stats.get('def', 0)
                cand_spd = candidate.base_stats.get('spd', 0)
                if opp_is_physical and cand_def > cand_spd:
                    score += 100
                elif not opp_is_physical and cand_spd > cand_def:
                    score += 100
            return score

        escolhido = max(candidates, key=get_score)
        # Marca quem esta a SAIR de campo, para o proximo turno saber que nao deve
        # regressar imediatamente. E aqui que o ciclo A->B->A e quebrado na raiz.
        if escolhido is not None:
            self._registar_saida(battle)
        return escolhido

    def get_offensive_switch(self, battle, history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not candidates:
            return None
        active_weather = battle.weather
        opp_types_obj = [t for t in opponent.types if t] if opponent else []
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0] if opponent else []

        def get_score(candidate):
            hazard_dmg = self.physics_get_hazard_damage(candidate, battle)
            hp_frac = candidate.current_hp_fraction
            if hp_frac <= hazard_dmg + 0.05:
                return -9999
            # ANTI-CICLO: penaliza regressar a um Pokemon que saiu de campo ha
            # <= TURNOS_DE_QUARENTENA turnos (ver _penalizacao_quarentena).
            score = -self._penalizacao_quarentena(candidate, battle)
            score += self._bonus_trick_room(candidate, battle)
            if hp_frac >= 0.7:
                score += 200
            elif hp_frac >= 0.35:
                score += 100
            else:
                score += 50
            if self.physics.get_role(candidate) == Role.SWEEPER:
                score += 100
            cand_abi = str(candidate.ability).lower() if candidate.ability else ""
            weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration']
            if active_weather:
                weather_start = history.get('weather_start_turn', battle.turn) if history and 'weather_start_turn' in history else battle.turn
                if cand_abi in weather_abusers and (battle.turn - weather_start) < 4:
                    score += 200
            if opponent:
                # MIGRADO PARA `physics.mais_rapido` EM 30/08/2026: a comparacao crua
                # ignorava clima, terreno, Tailwind e a INVERSAO do Trick Room.
                opp_e_mais_rapido = self.physics.mais_rapido(opponent, candidate, battle)
                cand_e_mais_rapido = self.physics.mais_rapido(candidate, opponent, battle)
                has_weakness = False
                for opp_type in opp_types_obj:
                    mult = candidate.damage_multiplier(opp_type)
                    if mult > 1.0:
                        score -= 150 * mult
                        has_weakness = True
                    elif mult < 1.0:
                        score += min(150, 25 / max(mult, 0.1))
                for move in known_opp_moves:
                    mult = candidate.damage_multiplier(move)
                    if mult > 1.0:
                        score -= 200 * mult
                        has_weakness = True
                    elif mult < 1.0:
                        score += min(150, 50 / max(mult, 0.1))
                if opp_e_mais_rapido and has_weakness:
                    score -= 300
                if cand_e_mais_rapido:
                    score += 150
                has_se_move = False
                for m in candidate.moves.values():
                    if m.base_power > 0:
                        mult = opponent.damage_multiplier(m)
                        if mult > 1.0:
                            score += 100 * mult
                            has_se_move = True
                if has_se_move:
                    score += 150
            return score

        escolhido = max(candidates, key=get_score)
        # Marca quem esta a SAIR de campo, para o proximo turno saber que nao deve
        # regressar imediatamente. E aqui que o ciclo A->B->A e quebrado na raiz.
        if escolhido is not None:
            self._registar_saida(battle)
        return escolhido

    def get_post_faint_switch(self, battle, history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not opponent or not candidates:
            return None
        opp_types_obj = [t for t in opponent.types if t]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]

        def get_general_score(cand):
            score = 0.0
            # MIGRADO 30/08/2026: ver nota em get_defensive_switch. `battle` vem do
            # fecho de `get_post_faint_switch`.
            cand_e_mais_rapido = self.physics.mais_rapido(cand, opponent, battle)
            hp_frac = cand.current_hp_fraction
            if hp_frac >= 0.7:
                score += 150
            elif hp_frac >= 0.4:
                score += 50
            else:
                score -= 100
            has_weakness = False
            for opp_type in opp_types_obj:
                mult = cand.damage_multiplier(opp_type)
                if mult > 1.0:
                    score -= 100 * mult
                    has_weakness = True
                elif mult < 1.0:
                    score += 50 / max(mult, 0.1)
            for move in known_opp_moves:
                mult = cand.damage_multiplier(move)
                if mult > 1.0:
                    score -= 150 * mult
                    has_weakness = True
                elif mult < 1.0:
                    score += 75 / max(mult, 0.1)
            if cand_e_mais_rapido:
                score += 100
                has_se_move = any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in cand.moves.values())
                if has_se_move:
                    score += 150
            else:
                if has_weakness:
                    score -= 200
            matchup = self.parser.get_matchup_state(cand, opponent)
            if matchup == MatchupState.DOMINANT:
                score += 200
            elif matchup == MatchupState.DEFENSIVE_ADV:
                score += 100
            elif matchup == MatchupState.CRITICAL_DIS:
                score -= 300
            return score

        return max(candidates, key=get_general_score)

    # ======================================================================
    # SELEÇÃO DE GOLPE DENTRO DE UMA CATEGORIA (desempate)
    # ======================================================================

    def _select_best_move_in_category(self, candidates, cat, active, opponent, battle):
        if not candidates:
            return None

        if cat == MoveCategory.HAZARD:
            priority = {'stealthrock': 4, 'stickyweb': 3, 'spikes': 2, 'toxicspikes': 1}
            return max(candidates, key=lambda m: priority.get(m.id, 0))

        if cat == MoveCategory.STATUS:
            def status_score(m):
                # Precisao EFETIVA: o Blizzard nunca erra no granizo.
                s = self.physics.precisao_efetiva(m, battle)
                if m.id in ['spore', 'sleeppowder', 'yawn']:
                    s += 50
                elif m.id in ['willowisp', 'thunderwave', 'glare']:
                    s += 30
                elif m.id in ['toxic']:
                    s += 20
                return s
            return max(candidates, key=status_score)

        if cat == MoveCategory.ATTACK_TECH:
            # ==============================================================
            # BASE POR DANO ESTIMADO, NAO POR POTENCIA BRUTA (30/08/2026)
            # ==============================================================
            # `float(m.base_power)` ignora tipo, STAB, terreno, clima, itens e
            # atributos. Os bonus (+200 a +500) foram calibrados contra potencias de
            # 40 a 120, logo QUALQUER bonus dominava QUALQUER diferenca de tipo.
            #
            # Observado em batalha manual: Rillaboom (Grass Surge, com Grassy Terrain
            # em campo) usou Knock Off, de 65 de potencia e NEUTRO, contra um Mega
            # Swampert (Agua/Terra) que leva 4x de Grama e ainda com STAB e terreno a
            # somar. O +200 do Knock Off venceu 4x de efetividade.
            #
            # `estimate_damage_percent` devolve fracao do HP do alvo. Multiplicado por
            # 100 fica numa escala de 0 a ~100, comparavel a antiga de potencia, mas
            # com metade da amplitude tipica (um golpe forte tira 40-60%, nao 120).
            # Por isso os bonus sao REESCALADOS para MEIO dos valores antigos, e
            # passam a ler-se como "quanto dano extra vale este efeito":
            #
            #   Rapid Spin com hazards no nosso lado   250  vale mais que qualquer golpe
            #   status em alvo limpo                   150  vale ~1,5 golpes fortes
            #   Knock Off com item derrubavel          100  vale ~1 golpe forte
            #   golpe de prisao                        125  ~5 turnos de residual
            def tech_score(m):
                try:
                    s = self.physics.estimate_damage_percent(m, active, opponent, battle) * 100.0
                except Exception:
                    s = float(m.base_power) * 0.5     # fallback na escala nova
                if m.id in ['rapidspin', 'mortalspin'] and self.parser.get_hazard_state(battle.side_conditions) == "SET":
                    s += 250
                elif m.id in ['nuzzle', 'scald', 'discharge', 'lavaplume'] and self.parser.get_status_state(opponent) == "CLEAN":
                    s += 150
                elif m.id in self.physics.REMOVEDORES_DE_ITEM and self.physics.item_e_removivel(opponent):
                    # CORRIGIDO 28/08/2026: o bonus era incondicional. Knock Off contra
                    # um alvo SEM ITEM ganhava +200 na mesma e vencia golpes tech que
                    # ainda tinham efeito por aplicar. O bonus e pelo item derrubado,
                    # logo so existe enquanto houver item.
                    # CORRIGIDO 30/08/2026: ter item nao chega, o item tem de ser
                    # REMOVIVEL. Mega Stones, Z-Crystals e Orbes Primais nao saem com
                    # Knock Off. Ver `item_e_removivel` na fisica.
                    s += 100
                elif m.id in getattr(self.physics, "GOLPES_DE_PRISAO", set()):
                    # Prender vale mais que os 15 a 35 de potencia destes golpes: sao
                    # ~5 turnos em que o alvo nao sai e leva dano residual.
                    s += 125
                return s
            return max(candidates, key=tech_score)

        if cat == MoveCategory.ATTACK_PIVOT:
            # Mesma razao do TECH: dano estimado em vez de potencia x STAB manual.
            def pivot_score(m):
                try:
                    return self.physics.estimate_damage_percent(m, active, opponent, battle)
                except Exception:
                    return m.base_power * (1.5 if m.type in active.types else 1.0) * 0.005
            return max(candidates, key=pivot_score)

        return candidates[0]

    # ======================================================================
    # SOBREVIVENCIA AO PROXIMO GOLPE (03/09/2026)
    # ======================================================================
    # PORQUE EXISTE. Um pivo tem DUAS leituras opostas conforme a ordem de acao,
    # e o codigo antigo so conhecia uma delas (`is_faster`):
    #
    #   MAIS RAPIDOS   bate-se (ou debuffa-se) e sai-se ANTES do golpe adversario.
    #                  O substituto entra limpo e nos tambem nao levamos nada.
    #
    #   MAIS LENTOS    o adversario ataca PRIMEIRO, nos levamos o golpe, e so
    #                  depois o pivo resolve. O substituto entra limpo na mesma.
    #                  A jogada continua boa: o que muda e que quem esta em campo
    #                  PAGA A CONTA. So compensa se aguentar.
    #
    # Numa TROCA CRUA a conta e paga por quem ENTRA. O pivo troca quem paga. Por
    # isso a pergunta certa quando somos lentos nao e "sou mais rapido?", e
    # "aguento o proximo golpe?".
    #
    # O TELEPORT e o caso extremo desta mesma regra: prioridade -6, sai SEMPRE
    # por ultimo, logo `is_faster` nunca lhe diz nada e esta e a UNICA pergunta
    # que se lhe aplica. Ver o bloco de switches.
    #
    # A MARGEM de 1,20 existe porque `estimate_damage_percent` nao conhece EVs,
    # IVs, nature nem item do adversario, e nao modela critico. Sobreviver por
    # 20% de folga e o minimo para a jogada nao depender de a estimativa estar
    # certa.
    MARGEM_SOBREVIVENCIA_PIVO = 1.20

    def _sobrevive_ao_proximo_golpe(self, active, opponent, battle):
        """Aguentamos o melhor golpe CONHECIDO do adversario contra nos?

        DELEGA NO `physics.sobrevive_a` (03/09/2026). A aritmetica vivia aqui em
        copia propria e a `policy` precisava da mesma resposta: duas copias da
        mesma regra sao o padrao que ja produziu tres bugs de divergencia
        silenciosa neste projecto. A regra fica num sitio so.

        A politica de DESCONHECIDO e que continua a ser local, porque e uma
        decisao de execucao e nao de fisica: sem golpe revelado exige-se o PERFIL
        para o qual a jogada existe, um `Role.TANK`.
        """
        try:
            veredicto = self.physics.sobrevive_a(
                active, opponent, battle, margem=self.MARGEM_SOBREVIVENCIA_PIVO)
            if veredicto is None:
                return self.physics.get_role(active, battle) == Role.TANK
            return bool(veredicto)
        except Exception:
            # Na duvida nao se arrisca: falhar aqui custa o Pokemon, e a
            # alternativa (troca crua) nunca e catastrofica.
            return False

    # ======================================================================
    # PREDITIVO: COBERTURA DO BANCO, e nao so dano medio (03/09/2026)
    # ======================================================================
    # A media simples de dano no banco nao distingue estes dois casos:
    #
    #   golpe A   100% num alvo, 0% nos outros tres     media 25%
    #   golpe B    25% nos quatro alvos                 media 25%
    #
    # Para uma jogada cuja premissa e NAO SABER quem entra, B e melhor que A.
    # A cobertura mede isso: em quantos Pokemon do banco este golpe e o que
    # MAXIMIZA o dano, contado sobre o repertorio ofensivo inteiro.
    #
    # CALIBRACAO, declarada. A diferenca tipica de dano medio entre dois
    # candidatos anda por 0,15 em fracao de HP. Com 0,30, cobrir METADE do banco
    # vale 0,15, ou seja a mesma ordem de grandeza: nem a cobertura esmaga a
    # magnitude, nem o contrario. Primeira calibracao, revisivel por medicao.
    PESO_COBERTURA_PREDITIVA = 0.30

    # ======================================================================
    # COMPARACAO ENTRE CATEGORIAS DE ATAQUE — SO PARA O InstinctBot
    # ======================================================================
    # PORQUE ISTO EXISTE. O ranking escolhe uma CATEGORIA e o executor escolhe o
    # melhor golpe DENTRO dela. Se ATTACK_TECH vier antes de ATTACK_STRONG, o golpe
    # de STRONG nunca entra na comparacao, por melhor que seja.
    #
    # A solucao 1 (dano estimado no `tech_score`) corrige a escolha DENTRO da
    # categoria. Esta corrige a escolha ENTRE categorias.
    #
    # PORQUE SO O INSTINTO, e a razao e a mesma de sempre neste ficheiro: se o
    # executor devolvesse um golpe de STRONG quando o CEREBRO pediu TECH, a Q-table
    # registaria a recompensa na ACAO ERRADA. Isso corrompe a representacao, que e
    # pior do que qualquer jogada ma — ja esta documentado no masking a proposito do
    # mesmo tipo de desalinhamento. Para o Blue e o Green a separacao entre
    # categorias de ataque E o conhecimento que eles estao a aprender, e tem de ser
    # respeitada mesmo quando produz uma jogada pior.
    MARGEM_TROCA_DE_CATEGORIA = 1.5

    def _preferir_ataque_muito_melhor(self, escolhido, active, opponent, battle,
                                      history=None):
        """Troca o TECH escolhido por um golpe claramente mais forte, se existir.

        A margem de 1.5 le-se como: o efeito tatico do TECH (derrubar item, aplicar
        status, limpar hazards) vale ate 50% de dano extra. Acima disso, o dano
        ganha. Nao e 1.0 de proposito — com margem zero o instinto deixaria de usar
        TECH sempre que houvesse um golpe marginalmente mais forte, e perderia os
        efeitos que justificam a categoria existir.
        """
        try:
            dano = lambda m: self.physics.estimate_damage_percent(m, active, opponent, battle)
            base = dano(escolhido)
            alternativas = [
                m for m in battle.available_moves
                if m.base_power > 0 and m.id != escolhido.id
                and not self.masker.is_move_useless(m, opponent, battle, history)
            ]
            if not alternativas:
                return escolhido
            melhor = max(alternativas, key=dano)
            if dano(melhor) >= base * self.MARGEM_TROCA_DE_CATEGORIA:
                return melhor
        except Exception:
            pass
        return escolhido

    # ======================================================================
    # PONTO DE ENTRADA: intenção -> objeto concreto
    # ======================================================================

    def get_best_execution_object(self, base_action, battle, history=None,
                                  fallback_obediencia=True, comparar_ataques=False,
                                  atalho_de_pivo=True):
        if isinstance(base_action, list):
            base_action = base_action[0]

        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon

        # ==================================================================
        # `SWITCH` GENERICO -> ACCAO EXPLICITA (03/09/2026)
        # ==================================================================
        # VERIFICADO EM 03/09: nenhum produtor de `"SWITCH"` existe no projecto.
        # A `policy.py` emite sempre `SWITCH_DEFENSIVE` ou `SWITCH_OFFENSIVE` (39
        # ocorrencias, zero genericas), `brain.base_actions` so tem as duas
        # variantes, e `MoveCategory` nao tem o valor generico.
        #
        # O ramo estava vivo por um caminho perverso: `MoveCategory["SWITCH"]`
        # levanta KeyError, o `except KeyError: pass` mais abaixo engole-o, e o
        # fluxo chegava ao bloco de switches onde o `else` o encaminhava SEMPRE
        # para troca OFENSIVA. Um pedido sem intencao declarada virava a variante
        # ofensiva em silencio — o mesmo padrao de falha calada que domina o
        # historico do projecto.
        #
        # NAO SE REMOVE (fica como guarda contra chamador externo), mas passa a
        # NORMALIZAR para uma das duas explicitas ANTES de qualquer decisao, pelo
        # mesmo criterio que a `policy` ja usa na l.465: ameacados ou mais lentos,
        # sai-se a defender; caso contrario, a atacar. As duas ramificacoes
        # atribuem, logo a partir daqui NENHUM caminho do executor volta a ver um
        # `SWITCH` generico, e por isso ele saiu da lista do bloco de switches.
        if base_action == "SWITCH":
            if active and opponent:
                ameacado = self._is_threatening(active, opponent, battle)
                mais_lentos = not self.physics.mais_rapido(active, opponent, battle)
                base_action = ("SWITCH_DEFENSIVE" if (ameacado or mais_lentos)
                               else "SWITCH_OFFENSIVE")
            else:
                # Sem leitura possivel do confronto, a conservadora.
                base_action = "SWITCH_DEFENSIVE"

        # Se estamos ameaçados e feridos, cancela ações de setup lento e ataca.
        if active and opponent:
            is_threat = self._is_threatening(active, opponent, battle)
            if is_threat and active.current_hp_fraction < 0.45:
                if base_action in ["BUFF", "HAZARD", "STATUS", "DEBUFF", "FIELD_CONTROL"]:
                    base_action = "ATTACK_STRONG"

        try:
            cat = MoveCategory[base_action]
            non_offensive = [
                MoveCategory.STATUS, MoveCategory.BUFF, MoveCategory.DEBUFF,
                MoveCategory.HAZARD, MoveCategory.HEAL, MoveCategory.FIELD_CONTROL,
                MoveCategory.CLEAN_HAZARD, MoveCategory.PROTECT, MoveCategory.ATTACK_PIVOT,
                MoveCategory.ATTACK_TECH, MoveCategory.STAT_CLEAN, MoveCategory.HEAL_STATUS, MoveCategory.PHAZE,
                # BARRIER e DISRUPTION ACRESCENTADAS EM 30/08/2026.
                #
                # Eram as DUAS UNICAS categorias nao ofensivas que faltavam aqui, e a
                # poda por `is_move_useless` vive DENTRO deste bloco. Com elas de
                # fora, o golpe nunca passava pela mascara e caia para ATTACK_STRONG.
                #
                # Consequencia medida em batalha manual: Aurora Veil usado com Aurora
                # Veil JA ATIVO, e outra vez DEPOIS de a neve ter parado. Os dois
                # casos tem guarda escrita no filtro 6 do masking, e a guarda nunca
                # corria. O filtro 12 (Taunt/Torment/Encore/Disable ja ativos) estava
                # inerte pela mesma razao, e nunca chegou a ser observado porque
                # ninguem foi procurar.
                #
                # E o mesmo padrao do `history` inerte (6.32) e do Stealth Rock
                # (6.35): codigo que existe e nao corre. A lista tem de ser revista
                # sempre que uma MoveCategory nova nascer.
                MoveCategory.BARRIER, MoveCategory.DISRUPTION,
            ]
            if cat in non_offensive:
                # CONTEXTO PASSADO AO classify_move (28/08/2026).
                # Um ATTACK_TECH vale pelo EFEITO. Depois de o efeito estar aplicado
                # sobra-lhe so o dano, que costuma ser fraco. O caso concreto: Knock
                # Off com o item ja removido continuava a ser escolhido como TECH, e
                # um Pokemon com DOIS golpes tech ficava a alterna-los sem nunca usar
                # o ataque forte que tinha no moveset.
                #
                # Com `defender` e `battle`, o physics reclassifica esses golpes como
                # ATTACK_STRONG: saem daqui e passam a competir pelo DANO, que e o
                # unico valor que ainda tem. Se ainda assim forem o golpe mais forte,
                # continuam a ser escolhidos — pela razao certa.
                #
                # Vive no EXECUTOR e nao no masking de proposito: e conhecimento do
                # MOTOR DO JOGO, nao estrategia, logo aplica-se por igual aos quatro
                # agentes e nao alarga a diferenca entre o Blue e o Green.
                candidates = [
                    m for m in battle.available_moves
                    if self.physics.classify_move(m, opponent, battle) == cat
                    and not self.masker.is_move_useless(m, opponent, battle, history)
                ]
                if cat == MoveCategory.HAZARD:
                    candidates = [m for m in candidates if not self.masker.is_hazard_already_set(m, battle)]
                if candidates:
                    best = self._select_best_move_in_category(candidates, cat, active, opponent, battle)
                    if best:
                        # SO ATTACK_TECH. O ATTACK_PIVOT fica de fora de
                        # proposito: um U-turn e escolhido pela TROCA que
                        # provoca, nao pelo dano, e compara-lo por dano
                        # eliminaria o pivo do repertorio.
                        if comparar_ataques and cat == MoveCategory.ATTACK_TECH:
                            # `history` PASSADO EM 04/09/2026. O parametro
                            # existia com `None` por omissao e NENHUM dos sete
                            # chamadores de `is_move_useless` o passava, logo o
                            # filtro do Wish nunca podava. Wish e Future Sight sao
                            # SLOT conditions e o poke-env nao as expoe, logo a
                            # memoria e nossa e tem de CHEGAR ao masking.
                            best = self._preferir_ataque_muito_melhor(
                                best, active, opponent, battle, history)
                        return best
                # Sem golpes viáveis na categoria pedida -> ataca.
                base_action = "ATTACK_STRONG"
        except KeyError:
            pass

        # SWITCHES (com atalho de pivot, so para o InstinctBot)
        # `"SWITCH"` NAO consta desta lista de proposito: foi normalizado no topo
        # da funcao para uma das duas variantes explicitas, e nunca chega aqui.
        if base_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"]:
            # ==================================================================
            # ATALHO DE PIVO: QUATRO DEFEITOS CORRIGIDOS EM 03/09/2026
            # ==================================================================
            # 1. SEM MASCARA E SEM ORDENACAO. Devolvia `pivot_moves[0]`, o
            #    PRIMEIRO golpe de pivo do moveset. Ordem de moveset nao e
            #    criterio nenhum. Volt Switch contra um Terra e o PIOR resultado
            #    possivel para uma intencao de TROCA: dano zero E a troca nao
            #    acontece, porque o golpe falha. Perde-se o turno inteiro e
            #    fica-se em campo, que era exatamente o que a intencao queria
            #    evitar. Observado no log da guarda de imunidade contra nidoking,
            #    gliscor e steelix, nas DUAS intencoes de troca.
            #
            # 2. O EXECUTOR E PARTILHADO, E `ATTACK_PIVOT` JA E UMA ACAO DO
            #    ESPACO. O atalho fazia `SWITCH_*` executar silenciosamente um
            #    `ATTACK_PIVOT`, e a Q-table registava a recompensa em `SWITCH_*`.
            #    Isso colapsa duas acoes que o cerebro existe para distinguir, e
            #    com o pivo imune a recompensa de um turno em que NENHUMA troca
            #    ocorreu ia para as duas accoes de troca. Esteve activo em todos
            #    os ciclos medidos ate ao v9/v6 inclusive.
            #    Dai o `atalho_de_pivo`: para o Blue e o Green SWITCH significa
            #    SWITCH, e o pivo continua acessivel pelo rotulo certo, porque
            #    `physics.classify_move` devolve ATTACK_PIVOT para os cinco
            #    golpes (u-turn, volt switch, flip turn, parting shot, teleport).
            #
            # 3. `is_faster` COMO GATE UNICO. Sendo mais LENTOS o pivo continua a
            #    valer: levamos o golpe, saimos, e o substituto entra limpo. O
            #    que muda e que quem paga a conta somos nos, logo a pergunta
            #    passa a ser se aguentamos. Ver `_sobrevive_ao_proximo_golpe`.
            #
            # 4. O TELEPORT NA MESMA LISTA DOS OUTROS. Prioridade -6: sai SEMPRE
            #    por ultimo, logo `is_faster` nunca se lhe aplica. Ele nao e
            #    momentum, e uma TROCA DE QUEM PAGA A CONTA: o Pokemon
            #    resistente absorve o ataque para o seguinte entrar sem dano.
            if atalho_de_pivo and active and opponent and battle.available_switches:
                # MIGRADO PARA `physics.mais_rapido` EM 30/08/2026: a comparacao crua
                # ignorava clima, terreno, Tailwind e a INVERSAO do Trick Room.
                is_faster = self.physics.mais_rapido(active, opponent, battle)
                aguenta = self._sobrevive_ao_proximo_golpe(active, opponent, battle)

                # (a) PIVOS DE PRIORIDADE 0. Servem se agirmos antes, ou se
                # aguentarmos o golpe e agirmos depois.
                if is_faster or aguenta:
                    pivot_moves = [
                        m for m in battle.available_moves
                        if m.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot']
                        and not self.masker.is_move_useless(m, opponent, battle, history)
                    ]
                    if pivot_moves:
                        # Ordenados por dano estimado, mesmo criterio do
                        # `pivot_score`. O Parting Shot pontua 0 e perde para um
                        # pivo de dano, que e o resultado certo: bater e sair vale
                        # mais que so debuffar e sair.
                        return self._select_best_move_in_category(
                            pivot_moves, MoveCategory.ATTACK_PIVOT,
                            active, opponent, battle)

                # (b) TELEPORT, fora da lista acima de proposito: nao passa por
                # `is_faster` e nao compete por dano, compete por sobrevivencia.
                if aguenta:
                    teleport = next((m for m in battle.available_moves
                                     if m.id == 'teleport'), None)
                    if teleport:
                        return teleport
            if base_action == "SWITCH_DEFENSIVE":
                switch = self.get_defensive_switch(battle, history)
            else:
                switch = self.get_offensive_switch(battle, history)
            if switch:
                return switch

        # BLOCO DE ATAQUE
        if base_action in ["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH"]:
            valid_moves = [m for m in battle.available_moves if self.physics.classify_move(m, opponent, battle) in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT]]
            useful_moves = [m for m in valid_moves if not self.masker.is_move_useless(m, opponent, battle, history)]
            if useful_moves:
                valid_moves = useful_moves
            elif not fallback_obediencia:
                # ==========================================================
                # QUEM PEDE DECIDE (30/08/2026)
                # ==========================================================
                # O fallback de OBEDIENCIA logo abaixo descarta a mascara de
                # proposito e devolve um golpe qualquer com potencia. Isso e
                # CERTO para o Blue e o Green: o cerebro escolheu ATTACK, tem de
                # receber a consequencia de ter escolhido mal, senao aprende que
                # ATTACK nunca custa nada.
                #
                # MAS O InstinctBot NAO TEM Q-TABLE PARA PUNIR. Para ele o
                # fallback e so usar um golpe inutil, e nada no sistema regista
                # que foi mau — por isso repete.
                #
                # Observado em batalha manual: Zeraora usou Plasma Fists contra
                # Steelix (Aco/Terra) em DOIS turnos seguidos. O filtro 0 do
                # masking (imunidade de tipo, multiplicador 0) funcionou e podou
                # o golpe as duas vezes; o fallback ignorou o resultado.
                #
                # Devolver None faz o ranking do instinto AVANCAR para a
                # intencao seguinte (trocar, status, hazard), que e sempre
                # melhor do que gastar o turno num golpe de dano zero.
                #
                # O executor e partilhado pelos tres agentes e cada um constroi
                # o seu (`build_instinct()` em `InstinctBot.__init__` e em
                # `TabularAgent.__init__`), mas o comportamento e escolhido por
                # CHAMADA e nao por instancia: o parametro tem por omissao o
                # valor antigo, logo nenhum chamador existente muda.
                return None
            else:
                # Obediência: entrega um golpe (mesmo inútil) para a Q-Table ser punida.
                valid_moves = [m for m in battle.available_moves if m.base_power > 0]
                if not valid_moves:
                    valid_moves = battle.available_moves

            if valid_moves:
                strong_move = None
                max_strong_score = -9999
                opp_hp_frac = opponent.current_hp_fraction
                opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
                # MIGRADO PARA `physics.equipa_adversaria` EM 04/09/2026. O banco
                # vinha quase vazio no inicio da batalha, logo o preditivo
                # pontuava contra um conjunto pequeno e escolhia mal. Caso
                # observado: Excadrill contra Toxapex (WATER/POISON) escolheu
                # `ironhead` (0,5x) tendo `earthquake` (2x) no moveset — quatro
                # vezes pior. A 6.46 registou, ERRADAMENTE, que o preditivo
                # funcionava desde o turno 1; corrigir esse ponto no documento.
                _, benched_opponents = self.physics.equipa_adversaria(battle)

                # PONDERACAO POR PRECISAO, com prioridade a LETALIDADE.
                # Regra (decidida no projeto): um golpe letal e sempre preferido a um
                # nao-letal, mesmo que seja menos preciso — matar agora vale mais que
                # dano esperado. A precisao so decide o desempate:
                #   - se AMBOS sao letais  -> vence o mais PRECISO (mata com mais certeza)
                #   - se NENHUM e letal    -> vence o de maior DANO ESPERADO (dano x prec)
                # Sem isto o executor escolhia por dano bruto e preferia, por exemplo,
                # Hydro Pump (110 x 80% = 88) a Surf (90 x 100% = 90), e o cerebro
                # levava a culpa pelo miss de uma escolha que nao fez.
                def _precisao(mv):
                    # BLIZZARD NUNCA ERRA NO GRANIZO (30/08/2026). Regra de jogo
                    # desde a Gen IV: com granizo em campo o golpe salta a
                    # verificacao de precisao. Sem isto o Blizzard era pontuado a 70%
                    # justamente na equipa que instala o granizo para o usar, e perdia
                    # para golpes mais fracos e mais certeiros.
                    #
                    # SNOW e SNOWSCAPE NAO contam: na Gen 9 so o GRANIZO da o efeito.
                    try:
                        clima = (next(iter(battle.weather)).name.upper()
                                 if battle and battle.weather else "CLEAR")
                        if getattr(mv, "id", "") == 'blizzard' and clima == 'HAIL':
                            return 1.0
                    except Exception:
                        pass
                    a = getattr(mv, "accuracy", None)
                    if a is None or a is True:
                        return 1.0          # nunca falha
                    try:
                        a = float(a)
                    except (TypeError, ValueError):
                        return 1.0
                    return a / 100.0 if a > 1.0 else a

                # Contexto usado pelos bonus de prioridade e de dreno.
                # MIGRADO 30/08/2026. `battle` vem do fecho de
                # `get_best_execution_object`. Importa aqui em particular: o bonus de
                # cura antecipada do dreno so faz sentido se soubermos mesmo quem age
                # primeiro, e sob Trick Room a resposta e a oposta.
                somos_mais_rapidos = self.physics.mais_rapido(active, opponent, battle)
                # Melhor golpe CONHECIDO do adversario contra nos. Se ainda nao
                # revelou nenhum, fica 0.0 e as regras que dependem disto nao
                # disparam — preferivel a inventar um valor.
                dano_do_adversario = 0.0
                for mv in getattr(opponent, "moves", {}).values():
                    if getattr(mv, "base_power", 0) > 0:
                        dano_do_adversario = max(
                            dano_do_adversario,
                            self.physics.estimate_damage_percent(mv, opponent, active, battle))
                meu_hp = active.current_hp_fraction

                letais = []
                for m in valid_moves:
                    dmg_cru = self.physics.estimate_damage_percent(m, active, opponent, battle)
                    if dmg_cru >= opp_hp_frac:
                        letais.append(m)
                ha_letal = len(letais) > 0

                for m in valid_moves:
                    dano = self.physics.estimate_damage_percent(m, active, opponent, battle)
                    prec = _precisao(m)
                    e_letal = dano >= opp_hp_frac

                    if ha_letal:
                        # Ha pelo menos um golpe letal: os nao-letais ficam para tras.
                        # Entre os letais, decide a PRECISAO (probabilidade de matar).
                        score = (1000.0 + prec * 100.0) if e_letal else dano * prec
                    else:
                        # Nenhum mata: maximiza o DANO ESPERADO.
                        score = dano * prec

                    # 1. PRIORIDADE (revisto em 29/08/2026).
                    #
                    # A regra antiga era `m_priority > 0 and score >= opp_hp_frac`, e
                    # tinha dois defeitos: comparava o SCORE (que para um letal ja vale
                    # 1000+) com uma fracao de HP, logo era sempre verdadeira nos
                    # letais; e dava os mesmos 5 pontos fossemos rapidos ou lentos.
                    #
                    # Um golpe de prioridade vale acima de tudo quando somos MAIS
                    # LENTOS e ele MATA: e a unica forma de agir primeiro, e troca um
                    # turno em que iamos levar dano por um KO. Sendo ja mais rapidos, a
                    # prioridade nao acrescenta nada — matamos na mesma primeiro — e o
                    # golpe deve ser tratado como um ataque normal, competindo so pelo
                    # dano esperado.
                    try:
                        m_priority = m.priority
                    except (KeyError, AttributeError):
                        m_priority = 0
                    if m_priority > 0 and e_letal and not somos_mais_rapidos:
                        score += 500.0

                    # 1b. DRENO (29/08/2026): a cura so vale quando ha o que curar E
                    # quando ela muda o desfecho do turno. Duas situacoes, e so:
                    #
                    #   (a) SOMOS MAIS RAPIDOS e estamos magoados: curamos ANTES de
                    #       levar o proximo golpe, e podemos passar a sobreviver-lhe.
                    #   (b) A CURA COMPENSA o dano que ele nos faz e sobrevivemos ao
                    #       golpe dele: e sustento real, o confronto passa a favor.
                    #       Cobre o caso de sermos mais LENTOS — levamos o golpe e
                    #       recuperamos a seguir.
                    #
                    # Com vida cheia o bonus e zero por construcao (`falta` = 0): nao
                    # ha nada para curar e o golpe compete so pelo dano, que e o
                    # correto. Nao e preciso regra binaria para isso.
                    fracao = self.physics.fracao_de_dreno(m)
                    if fracao > 0.0:
                        cura = dano * fracao
                        falta_agora = max(0.0, 1.0 - meu_hp)
                        sobrevive = dano_do_adversario > 0.0 and meu_hp > dano_do_adversario
                        compensa = cura >= dano_do_adversario > 0.0
                        if (somos_mais_rapidos and meu_hp <= 0.5) or (sobrevive and compensa):
                            # CURA ANTECIPADA (corrigido 29/08/2026). Sendo mais
                            # LENTOS, levamos o golpe ANTES de drenar: mesmo com vida
                            # cheia ha o que curar quando a cura acontecer. Usar so
                            # `1 - hp_atual` dava bonus ZERO a vida cheia e apagava
                            # exatamente o caso em que o dreno brilha para um Pokemon
                            # lento.
                            #
                            # O peso e METADE quando a cura e toda antecipada, porque
                            # o dano previsto e incerto: o adversario pode trocar,
                            # falhar, ou usar status. Com HP ja em falta, a cura e
                            # certa e pesa a dobrar.
                            falta_prevista = falta_agora
                            peso = 1.0
                            if not somos_mais_rapidos and dano_do_adversario > 0.0:
                                falta_prevista = min(1.0, falta_agora + dano_do_adversario)
                                if falta_agora <= 0.05:
                                    peso = 0.5
                            # Escala: cura efetiva (limitada pelo que falta) em pontos
                            # de HP. Fica acima do desempate por precisao e MUITO
                            # abaixo dos 1000 de um golpe letal, que continua a mandar.
                            score += min(cura, falta_prevista) * 100.0 * peso

                    # 2. Inteligência de recoil: preserva-te se fores útil ao resto do time.
                    has_recoil = m.id in ['bravebird', 'flareblitz', 'doubleedge', 'woodhammer', 'wildcharge']
                    if has_recoil and score >= opp_hp_frac and opp_alive > 1:
                        future_utility = any(
                            self.parser.get_matchup_state(active, b_opp) in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]
                            for b_opp in benched_opponents
                        )
                        if future_utility:
                            for other_m in valid_moves:
                                if other_m.id != m.id and self.physics.estimate_damage_percent(other_m, active, opponent, battle) >= opp_hp_frac:
                                    score -= 2.0
                                    break

                    # 3. Consciência de self-drops: evita sujar stats sem necessidade.
                    self_drop = m.id in ['closecombat', 'superpower', 'dracometeor', 'leafstorm', 'overheat', 'makeitrain', 'fleurcannon']
                    if self_drop:
                        if score < opp_hp_frac:
                            score -= 0.3
                        if m.category.name == "SPECIAL" and active.boosts.get('spa', 0) < 0:
                            score -= 1.5
                        elif m.category.name == "PHYSICAL" and active.boosts.get('atk', 0) < 0:
                            score -= 1.5

                    # 4. Bónus multi-hit e efeito secundário.
                    if m.id in ['iciclespear', 'rockblast', 'bulletseed', 'tailslap', 'pinmissile', 'watershuriken']:
                        score += 0.2
                    if getattr(m, 'secondary', None):
                        score += 0.05

                    # 5. PIVO COMO ATAQUE COM HAZARD NO NOSSO CAMPO (03/09/2026)
                    #
                    # `valid_moves` inclui ATTACK_PIVOT, logo um U-turn compete
                    # como ataque normal quando a intencao e ATTACK_STRONG. Isso
                    # e certo: ele faz dano. O que faltava era o CUSTO ESCONDIDO
                    # — a troca que ele provoca faz entrar um Pokemon nosso, e
                    # com Stealth Rock ou Spikes do nosso lado essa entrada paga
                    # dano que um ataque normal nao pagaria.
                    #
                    # Observado em batalha manual: pivo escolhido como ataque com
                    # hazards em campo, e o substituto entrou a perder vida de
                    # graca.
                    #
                    # O desconto e o DANO REAL DE ENTRADA, medio sobre quem
                    # poderia entrar, na mesma escala de fraccao de HP do resto
                    # do score. Nao e uma constante inventada: se o nosso lado
                    # estiver limpo vale zero por construcao, e com Spikes a tres
                    # camadas pesa o que tem de pesar.
                    #
                    # DESINCENTIVA, NAO PROIBE: um pivo LETAL continua a valer
                    # 1000+ e ganha na mesma, que e o resultado certo. E vale
                    # para os TRES agentes, porque e escolha DENTRO da categoria
                    # pedida e nao muda a atribuicao de recompensa (mesmo
                    # criterio do `tech_score` por dano estimado, 6.44 D).
                    if (self.physics.classify_move(m, opponent, battle)
                            == MoveCategory.ATTACK_PIVOT
                            and self.parser.get_hazard_state(battle.side_conditions) == "SET"):
                        entradas = battle.available_switches or []
                        if entradas:
                            custo = sum(self.physics_get_hazard_damage(c, battle)
                                        for c in entradas) / len(entradas)
                            score -= custo

                    if score > max_strong_score:
                        max_strong_score = score
                        strong_move = m

                if not strong_move:
                    strong_move = valid_moves[0]

                if base_action == "ATTACK_STRONG":
                    return strong_move

                if base_action == "ATTACK_PREDICTIVE":
                    # BANCO VAZIO: o adversario esta no ultimo Pokemon, nao ha
                    # troca para prever, e o preditivo degenera no `strong_move`
                    # mascarado. Nesse estado ATTACK_PREDICTIVE e ATTACK_STRONG
                    # executam a mesma coisa, e isso e CORRECTO: as duas
                    # intencoes SAO a mesma jogada quando nao ha nada a antecipar.
                    if benched_opponents:
                        all_offensive_moves = [
                            m for m in battle.available_moves
                            if self.physics.classify_move(m, opponent, battle) in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT]
                        ]

                        # ==========================================================
                        # EXCLUSAO POR ID, NAO POR TIPO (03/09/2026)
                        # ==========================================================
                        # Era `m.type != strong_move.type`, que eliminava TODOS os
                        # golpes do tipo do `strong_move`. Um Pokemon com dois
                        # golpes de Fogo, um fraco contra o activo mas o unico que
                        # resolve tres Pokemon do banco, perdia o segundo junto
                        # com o primeiro.
                        #
                        # E `strong_move` sai de `valid_moves`, que foi mascarado
                        # contra o ACTIVO. Ou seja: a mascara aplicada a quem esta
                        # em campo decidia que TIPOS podiam concorrer para prever
                        # quem esta no BANCO. Acoplamento sem justificacao.
                        #
                        # A exclusao continua a existir porque protege UMA coisa
                        # real: o preditivo nao pode devolver o mesmo objecto que
                        # o ATTACK_STRONG devolveria, senao as duas accoes
                        # colapsam na Q-table (mesmo defeito de representacao do
                        # atalho de pivo). Excluir o GOLPE resolve isso; excluir a
                        # familia dele nao protegia nada a mais.
                        #
                        # E POR ISSO UM GOLPE IMUNE CONTRA O ACTIVO E CANDIDATO
                        # LEGITIMO: o alvo do preditivo e quem ENTRA. Gunk Shot
                        # contra um Ferrothorn em campo vale zero, e pode ser a
                        # melhor resposta ao que vem a seguir.
                        predictive_candidates = [m for m in all_offensive_moves
                                                 if m.id != strong_move.id]
                        if predictive_candidates:
                            # MATRIZ golpe x banco, calculada UMA vez. Serve a
                            # media E o argmax por alvo, que de outro modo
                            # exigiriam duas passagens sobre a mesma estimativa.
                            def _dano_no_banco(mv, alvo):
                                try:
                                    return self.physics.estimate_damage_percent(
                                        mv, active, alvo, battle)
                                except Exception:
                                    return 0.0

                            matriz = {m.id: [_dano_no_banco(m, b) for b in benched_opponents]
                                      for m in all_offensive_moves}

                            # Para cada alvo do banco, qual golpe MAXIMIZA o dano.
                            # Calculado sobre o repertorio INTEIRO, `strong_move`
                            # incluido: se o golpe que o ATTACK_STRONG ja
                            # escolheria e a melhor resposta a um Pokemon do
                            # banco, prever nao acrescenta nada nesse alvo e
                            # nenhum candidato leva credito por ele.
                            melhor_por_alvo = [
                                max(all_offensive_moves,
                                    key=lambda mv: matriz[mv.id][i]).id
                                for i in range(len(benched_opponents))
                            ]

                            best_pred_move = None
                            max_pred_score = -9999
                            for m in predictive_candidates:
                                danos = matriz[m.id]
                                avg_bench_dmg = sum(danos) / len(danos)
                                # COBERTURA: ver `PESO_COBERTURA_PREDITIVA`. Em
                                # quantos Pokemon do banco este golpe e a melhor
                                # resposta, como fracao do banco.
                                cobertura = (melhor_por_alvo.count(m.id)
                                             / len(benched_opponents))
                                score = (avg_bench_dmg
                                         + self.PESO_COBERTURA_PREDITIVA * cobertura)
                                # Efeitos que valem contra QUALQUER entrada, logo
                                # sobrevivem a incerteza sobre quem vem.
                                if m.id in ['knockoff', 'scald', 'nuzzle', 'saltcure', 'uturn', 'voltswitch', 'flipturn']:
                                    score += 0.20
                                if getattr(m, 'secondary', None):
                                    score += 0.05
                                if score > max_pred_score:
                                    max_pred_score = score
                                    best_pred_move = m
                            if best_pred_move:
                                return best_pred_move
                    return strong_move

        # Fallback final
        if battle.available_switches:
            return battle.available_switches[0]
        if battle.available_moves:
            return battle.available_moves[0]
        return None

    # ======================================================================
    # Helpers que espelham a policy (mantidos aqui para o executor ser autónomo)
    # ======================================================================

    def physics_get_hazard_damage(self, candidate, battle):
        """Dano de hazards na entrada. Delega ao mesmo cálculo da policy via física.
        Mantido como método próprio para o executor não depender da policy."""
        dmg = 0.0
        # MIGRADO PARA `physics.nomes_de` EM 03/09/2026. ESTE E O BUG QUE A 6.16
        # DEU POR CORRIGIDO E NAO ESTAVA: `physics_get_hazard_damage` devolvia
        # SEMPRE 0, com hazards em 91% dos times do pool. A escolha de troca
        # ignorou o dano de entrada em todos os ciclos ja treinados, e o desconto
        # do pivo sob hazards escrito em 03/09 nascia inerte por causa disto.
        cond_keys = self.physics.nomes_de(battle.side_conditions)
        cand_types_str = [t.name for t in candidate.types if t]
        if 'STEALTH_ROCK' in cond_keys:
            for t in candidate.types:
                if t:
                    rock_enum = getattr(type(t), 'ROCK', None)
                    if rock_enum:
                        dmg += 0.125 * candidate.damage_multiplier(rock_enum)
                        break
        if 'SPIKES' in cond_keys and 'FLYING' not in cand_types_str and str(candidate.ability).lower() != 'levitate':
            layers = int(battle.side_conditions.get('spikes', 1))
            dmg += 0.041 * layers
        return dmg

    def _is_threatening(self, my_mon, opp_mon, battle=None):
        if not opp_mon or not my_mon:
            return False
        if opp_mon.boosts.get('atk', 0) >= 2 or opp_mon.boosts.get('spa', 0) >= 2:
            return True
        # MIGRADO 30/08/2026. `mais_rapido(opp, my)` e o equivalente exacto de
        # `opp_speed > my_speed`: estrito, logo o empate continua a nao contar.
        if my_mon.current_hp_fraction < 0.45 and self.physics.mais_rapido(opp_mon, my_mon, battle):
            opp_atk = max(self.physics.estimate_stat(opp_mon, 'atk'), self.physics.estimate_stat(opp_mon, 'spa'))
            if opp_atk > 250:
                return True
        return False
