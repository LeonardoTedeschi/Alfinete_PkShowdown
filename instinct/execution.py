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

    def get_best_lead(self, battle):
        try:
            my_team = list(battle.team.values())
            opp_team = list(battle.opponent_team.values())
            if not opp_team:
                return "/team 123456"
            best_lead = None

            # Árvore 1: guerra de climas
            weather_setters = ['drought', 'drizzle', 'sandstream', 'snowwarning']
            my_weather_setter = next((m for m in my_team if str(m.ability) in weather_setters), None)
            opp_has_weather = any(str(m.ability) in weather_setters for m in opp_team)
            if my_weather_setter and opp_has_weather:
                best_lead = my_weather_setter

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
            score = 0.0
            if hp_frac >= 0.7:
                score += 200
            elif hp_frac >= 0.35:
                score += 100
            else:
                score += 50
            if self.physics.get_role(candidate) == Role.TANK:
                score += 100
            if opponent:
                cand_spe = self.physics.estimate_stat(candidate, 'spe')
                opp_spe = self.physics.estimate_stat(opponent, 'spe')
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
                if opp_spe > cand_spe and has_weakness:
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

        return max(candidates, key=get_score)

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
            score = 0.0
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
                cand_spe = self.physics.estimate_stat(candidate, 'spe')
                opp_spe = self.physics.estimate_stat(opponent, 'spe')
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
                if opp_spe > cand_spe and has_weakness:
                    score -= 300
                if cand_spe > opp_spe:
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

        return max(candidates, key=get_score)

    def get_post_faint_switch(self, battle, history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not opponent or not candidates:
            return None
        opp_spe = self.physics.estimate_stat(opponent, 'spe')
        opp_types_obj = [t for t in opponent.types if t]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]

        def get_general_score(cand):
            score = 0.0
            cand_spe = self.physics.estimate_stat(cand, 'spe')
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
            if cand_spe > opp_spe:
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
                s = float(m.accuracy) if isinstance(m.accuracy, (int, float)) else 100.0
                if m.id in ['spore', 'sleeppowder', 'yawn']:
                    s += 50
                elif m.id in ['willowisp', 'thunderwave', 'glare']:
                    s += 30
                elif m.id in ['toxic']:
                    s += 20
                return s
            return max(candidates, key=status_score)

        if cat == MoveCategory.ATTACK_TECH:
            def tech_score(m):
                s = float(m.base_power)
                if m.id in ['rapidspin', 'mortalspin'] and self.parser.get_hazard_state(battle.side_conditions) == "SET":
                    s += 500
                elif m.id in ['nuzzle', 'scald', 'discharge', 'lavaplume'] and self.parser.get_status_state(opponent) == "CLEAN":
                    s += 300
                elif m.id == 'knockoff':
                    s += 200
                return s
            return max(candidates, key=tech_score)

        if cat == MoveCategory.ATTACK_PIVOT:
            return max(candidates, key=lambda m: m.base_power * (1.5 if m.type in active.types else 1.0))

        return candidates[0]

    # ======================================================================
    # PONTO DE ENTRADA: intenção -> objeto concreto
    # ======================================================================

    def get_best_execution_object(self, base_action, battle, history=None):
        if isinstance(base_action, list):
            base_action = base_action[0]

        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon

        # Se estamos ameaçados e feridos, cancela ações de setup lento e ataca.
        if active and opponent:
            is_threat = self._is_threatening(active, opponent)
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
            ]
            if cat in non_offensive:
                candidates = [
                    m for m in battle.available_moves
                    if self.physics.classify_move(m) == cat and not self.masker.is_move_useless(m, opponent, battle)
                ]
                if cat == MoveCategory.HAZARD:
                    candidates = [m for m in candidates if not self.masker.is_hazard_already_set(m, battle)]
                if candidates:
                    best = self._select_best_move_in_category(candidates, cat, active, opponent, battle)
                    if best:
                        return best
                # Sem golpes viáveis na categoria pedida -> ataca.
                base_action = "ATTACK_STRONG"
        except KeyError:
            pass

        # SWITCHES (com atalho de pivot se formos mais rápidos)
        if base_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "SWITCH"]:
            if active and opponent and battle.available_switches:
                my_spe = self.physics.estimate_stat(active, 'spe')
                opp_spe = self.physics.estimate_stat(opponent, 'spe')
                is_faster = my_spe > opp_spe
                pivot_moves = [m for m in battle.available_moves if m.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']]
                if pivot_moves and is_faster:
                    return pivot_moves[0]
            if base_action == "SWITCH_DEFENSIVE":
                switch = self.get_defensive_switch(battle, history)
            else:
                switch = self.get_offensive_switch(battle, history)
            if switch:
                return switch

        # BLOCO DE ATAQUE
        if base_action in ["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH"]:
            valid_moves = [m for m in battle.available_moves if self.physics.classify_move(m) in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT]]
            useful_moves = [m for m in valid_moves if not self.masker.is_move_useless(m, opponent, battle)]
            if useful_moves:
                valid_moves = useful_moves
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
                benched_opponents = [m for m in battle.opponent_team.values() if not m.fainted and not m.active]

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
                    a = getattr(mv, "accuracy", None)
                    if a is None or a is True:
                        return 1.0          # nunca falha
                    try:
                        a = float(a)
                    except (TypeError, ValueError):
                        return 1.0
                    return a / 100.0 if a > 1.0 else a

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

                    # 1. Malícia de prioridade: golpe rápido que mata vale mais.
                    try:
                        m_priority = m.priority
                    except (KeyError, AttributeError):
                        m_priority = 0
                    if m_priority > 0 and score >= opp_hp_frac:
                        score += 5.0

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

                    if score > max_strong_score:
                        max_strong_score = score
                        strong_move = m

                if not strong_move:
                    strong_move = valid_moves[0]

                if base_action == "ATTACK_STRONG":
                    return strong_move

                if base_action == "ATTACK_PREDICTIVE":
                    if benched_opponents:
                        all_offensive_moves = [
                            m for m in battle.available_moves
                            if self.physics.classify_move(m) in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT]
                        ]
                        predictive_candidates = [m for m in all_offensive_moves if m.type != strong_move.type]
                        if predictive_candidates:
                            best_pred_move = None
                            max_pred_score = -9999
                            for m in predictive_candidates:
                                avg_bench_dmg = sum(self.physics.estimate_damage_percent(m, active, bench_mon, battle) for bench_mon in benched_opponents) / len(benched_opponents)
                                score = avg_bench_dmg
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
        cond_keys = [str(k).upper() for k in battle.side_conditions.keys()]
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

    def _is_threatening(self, my_mon, opp_mon):
        if not opp_mon or not my_mon:
            return False
        if opp_mon.boosts.get('atk', 0) >= 2 or opp_mon.boosts.get('spa', 0) >= 2:
            return True
        my_speed = self.physics.estimate_stat(my_mon, 'spe')
        opp_speed = self.physics.estimate_stat(opp_mon, 'spe')
        if my_mon.current_hp_fraction < 0.45 and opp_speed > my_speed:
            opp_atk = max(self.physics.estimate_stat(opp_mon, 'atk'), self.physics.estimate_stat(opp_mon, 'spa'))
            if opp_atk > 250:
                return True
        return False
