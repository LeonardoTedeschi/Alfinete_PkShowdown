"""
Camada 4 — Política de Instinto (InstinctPolicy).

O instinto tático propriamente dito: dado o estado da battle, produz um RANKING de
intenções de ação (categorias como ATTACK_STRONG, HEAL, HAZARD...), da mais à menos
recomendada. É o "conhecimento de domínio" que o agente híbrido (Blue) usa como prior
de exploração e que o agente Instinto-puro usa diretamente.

Arquitetura em 3 sub-camadas internas:
  1. _get_tactical_mode: matchup + papéis -> um de 6 MODOS táticos (PRESS, CONTEST,
     GRIND, ESCAPE, LEAD, WALLBREAK).
  2. mode_templates: cada modo tem uma LISTA base de prioridades de categoria.
  3. _mod_*: 12 funções que REORDENAM a lista base conforme o contexto fino
     (velocidade, HP, ameaça, papéis role-vs-role).
Depois, get_instinct_profile filtra o ranking pelo action mask e aplica ajustes
finais (letalidade, barreiras, fadiga de troca).

DEPENDÊNCIAS (todas por injeção): GamePhysics, StateParser, ActionMasker.
A policy é o topo da hierarquia — depende de todas as camadas abaixo.

CONTRATO (get_instinct_profile) — CORRIGIDO:
  retorna SEMPRE 5 valores: (primary, confidence, ranking_list, candidate_mask, has_lethal)
  A versão monolítica tinha um early-return com apenas 4 valores, o que provocava
  ValueError no desempacotamento do agente e queda em choose_random_move. Aqui todos
  os caminhos devolvem 5 valores.
"""

from shared.definitions import Role, MatchupState, TacticalMode


class InstinctPolicy:
    """Decisão de intenção tática. Recebe física, parser e masker por injeção."""

    def __init__(self, physics, parser, masker):
        self.physics = physics
        self.parser = parser
        self.masker = masker
        self.mode_templates = self._build_mode_templates()
        self.role_modifiers = {
            (Role.SWEEPER, Role.SWEEPER): self._mod_sweeper_vs_sweeper,
            (Role.SWEEPER, Role.TANK): self._mod_sweeper_vs_tank,
            (Role.SWEEPER, Role.UTILITY): self._mod_sweeper_vs_utility,
            (Role.TANK, Role.SWEEPER): self._mod_tank_vs_sweeper,
            (Role.TANK, Role.TANK): self._mod_tank_vs_tank,
            (Role.TANK, Role.UTILITY): self._mod_tank_vs_utility,
            (Role.UTILITY, Role.SWEEPER): self._mod_utility_vs_sweeper,
            (Role.UTILITY, Role.TANK): self._mod_utility_vs_tank,
            (Role.UTILITY, Role.UTILITY): self._mod_utility_vs_utility,
        }

    # ======================================================================
    # SUB-CAMADA 2: templates base de prioridade por modo tático
    # ======================================================================

    def _build_mode_templates(self):
        return {
            TacticalMode.PRESS: [
                "ATTACK_PREDICTIVE", "ATTACK_STRONG", "BUFF", "ATTACK_TECH", "DISRUPTION",
                "HAZARD", "FIELD_CONTROL", "ATTACK_PIVOT", "CLEAN_HAZARD",
                "STATUS", "DEBUFF", "HEAL", "HEAL_STATUS", "STAT_CLEAN",
                "PHAZE", "PROTECT", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE",
            ],
            TacticalMode.CONTEST: [
                "ATTACK_STRONG", "ATTACK_TECH", "PROTECT", "ATTACK_PIVOT",
                "STATUS", "BUFF", "HEAL", "HAZARD", "CLEAN_HAZARD",
                "DEBUFF", "FIELD_CONTROL", "ATTACK_PREDICTIVE", "DISRUPTION", "STAT_CLEAN",
                "PHAZE", "HEAL_STATUS", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE",
            ],
            TacticalMode.GRIND: [
                "HAZARD", "STATUS", "HEAL", "PROTECT", "DISRUPTION", "DEBUFF",
                "CLEAN_HAZARD", "PHAZE", "STAT_CLEAN", "HEAL_STATUS",
                "BUFF", "FIELD_CONTROL", "ATTACK_TECH", "ATTACK_PIVOT",
                "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE", "ATTACK_STRONG", "ATTACK_PREDICTIVE",
            ],
            TacticalMode.ESCAPE: [
                "SWITCH_DEFENSIVE", "ATTACK_PIVOT", "PROTECT", "SWITCH_OFFENSIVE",
                "DISRUPTION", "ATTACK_TECH", "STATUS", "DEBUFF", "STAT_CLEAN", "PHAZE",
                "HEAL", "CLEAN_HAZARD", "HEAL_STATUS", "FIELD_CONTROL",
                "HAZARD", "BUFF", "ATTACK_STRONG", "ATTACK_PREDICTIVE",
            ],
            TacticalMode.LEAD: [
                "HAZARD", "FIELD_CONTROL", "ATTACK_PIVOT", "DISRUPTION",
                "ATTACK_STRONG", "STATUS", "DEBUFF", "BUFF", "PROTECT",
                "CLEAN_HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
                "ATTACK_PREDICTIVE", "ATTACK_TECH", "STAT_CLEAN", "HEAL_STATUS", "PHAZE",
            ],
            TacticalMode.WALLBREAK: [
                "ATTACK_TECH", "DISRUPTION", "STATUS", "BUFF", "ATTACK_PIVOT",
                "DEBUFF", "HAZARD", "ATTACK_STRONG", "ATTACK_PREDICTIVE",
                "HEAL", "CLEAN_HAZARD", "PROTECT", "SWITCH_OFFENSIVE",
                "STAT_CLEAN", "HEAL_STATUS", "PHAZE", "FIELD_CONTROL", "SWITCH_DEFENSIVE",
            ],
        }

    # ======================================================================
    # SUB-CAMADA 1: matchup + papéis -> modo tático
    # ======================================================================

    def _get_tactical_mode(self, matchup, my_role, opp_role, is_faster,
                           my_hp_frac, opp_hp_frac, is_threat, active, opponent):
        # Tank com a defesa "errada" contra o sweeper inimigo -> foge
        if my_role == Role.TANK and opp_role == Role.SWEEPER:
            opp_is_physical = self.physics._is_physical(opponent)
            my_def = active.base_stats.get('def', 0)
            my_spd = active.base_stats.get('spd', 0)
            is_right_def = (opp_is_physical and my_def >= my_spd) or (not opp_is_physical and my_spd > my_def)
            if not is_right_def:
                return TacticalMode.ESCAPE

        if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
            return TacticalMode.PRESS
        if matchup in [MatchupState.VOLATILE, MatchupState.NEUTRAL]:
            return TacticalMode.CONTEST
        if matchup in [MatchupState.STALEMATE, MatchupState.DEFENSIVE_ADV]:
            return TacticalMode.GRIND
        return TacticalMode.ESCAPE

    # ======================================================================
    # HELPERS de apoio à decisão
    # ======================================================================

    def is_threatening(self, my_mon, opp_mon):
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

    def _has_recovery(self, pokemon):
        if not pokemon:
            return False
        recovery_moves = ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun',
                          'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap']
        return any(m.id in recovery_moves for m in pokemon.moves.values())

    def _has_move(self, pokemon, move_ids):
        if not pokemon:
            return False
        return any(m.id in move_ids for m in pokemon.moves.values())

    def _opponent_can_setup(self, opponent):
        if not opponent:
            return False
        setup_moves = ['swordsdance', 'dragondance', 'nastyplot', 'quiverdance',
                       'shellsmash', 'shiftgear', 'calmmind', 'bulkup', 'workup', 'coil']
        return any(m.id in setup_moves for m in opponent.moves.values())

    def _get_hazard_damage(self, candidate, battle):
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

    def _get_survival_score(self, candidate, opponent, battle, is_active=False):
        if not candidate:
            return -9999
        hp_frac = candidate.current_hp_fraction
        if not is_active:
            hazard_dmg = self._get_hazard_damage(candidate, battle)
            if hp_frac <= hazard_dmg + 0.05:
                return -9999
        score = 0.0
        if hp_frac >= 0.7:
            score += 150
        elif hp_frac >= 0.4:
            score += 50
        else:
            score -= 100
        if not opponent:
            return score

        opp_types_obj = [t for t in opponent.types if t]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]
        has_weakness = False
        for opp_type in opp_types_obj:
            mult = candidate.damage_multiplier(opp_type)
            if mult > 1.0:
                score -= 100 * mult
                has_weakness = True
            elif mult < 1.0:
                score += 50 / max(mult, 0.1)
        for move in known_opp_moves:
            mult = candidate.damage_multiplier(move)
            if mult > 1.0:
                score -= 150 * mult
                has_weakness = True
            elif mult < 1.0:
                score += 75 / max(mult, 0.1)

        cand_spe = self.physics.estimate_stat(candidate, 'spe')
        opp_spe = self.physics.estimate_stat(opponent, 'spe')
        if cand_spe > opp_spe:
            score += 100
            has_se_move = any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in candidate.moves.values())
            if has_se_move:
                score += 150
        else:
            if has_weakness:
                score -= 200

        matchup = self.parser.get_matchup_state(candidate, opponent)
        if matchup == MatchupState.DOMINANT:
            score += 200
        elif matchup == MatchupState.DEFENSIVE_ADV:
            score += 100
        elif matchup == MatchupState.CRITICAL_DIS:
            score -= 300
        return score

    def _is_active_best_remaining(self, active, opponent, battle):
        # NOTA: no monólito original este método estava DUPLICADO (definido duas
        # vezes, linhas 325 e 1061). As duas versões eram funcionalmente idênticas
        # (mesma margem de +50), então o comportamento nunca variou — em Python a
        # segunda definição apenas sobrepunha a primeira. Aqui fica uma só cópia.
        if not battle.available_switches:
            return True
        active_score = self._get_survival_score(active, opponent, battle, is_active=True)
        best_bench_score = -9999
        for bench_mon in battle.available_switches:
            bench_score = self._get_survival_score(bench_mon, opponent, battle, is_active=False)
            if bench_score > best_bench_score:
                best_bench_score = bench_score
        # Só troca se o banco for consideravelmente mais seguro (+50), evitando
        # trocas infinitas entre dois Pokémon ruins.
        if best_bench_score > active_score + 50:
            return False
        return True

    # ======================================================================
    # SUB-CAMADA 3: os 12 modificadores de role (reordenam a lista base)
    # ======================================================================

    def _mod_sweeper_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if is_faster:
            if "ATTACK_STRONG" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
            if "ATTACK_PREDICTIVE" in modified:
                modified.insert(1, modified.pop(modified.index("ATTACK_PREDICTIVE")))
            if "BUFF" in modified:
                modified.remove("BUFF")
                modified.append("BUFF")
        else:
            if is_threat:
                if "SWITCH_DEFENSIVE" in modified:
                    modified.insert(0, "SWITCH_DEFENSIVE")
                if "ATTACK_TECH" in modified:
                    modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
            else:
                if "BUFF" in modified:
                    modified.insert(0, modified.pop(modified.index("BUFF")))
        return modified

    def _mod_sweeper_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        my_atk = active.base_stats.get('atk', 0)
        my_spa = active.base_stats.get('spa', 0)
        i_am_physical = my_atk >= my_spa
        opp_def = opponent.base_stats.get('def', 0)
        opp_spd = opponent.base_stats.get('spd', 0)
        tank_defense_high = (i_am_physical and opp_def >= 100) or (not i_am_physical and opp_spd >= 100)
        if tank_defense_high and "BUFF" in modified:
            modified.insert(0, modified.pop(modified.index("BUFF")))
        elif "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        if "DEBUFF" in modified:
            modified.insert(0, modified.pop(modified.index("DEBUFF")))
        if "BUFF" in modified and "DEBUFF" in modified:
            debuff_idx = modified.index("DEBUFF")
            buff_idx = modified.index("BUFF")
            if buff_idx < debuff_idx:
                modified[buff_idx], modified[debuff_idx] = modified[debuff_idx], modified[buff_idx]
        if my_hp_frac <= 0.5 and "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        return modified

    def _mod_sweeper_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        if opp_hp_frac >= 0.7 and "BUFF" in modified:
            modified.insert(0, modified.pop(modified.index("BUFF")))
        if not is_faster and is_threat and "SWITCH_DEFENSIVE" in modified:
            modified.insert(0, "SWITCH_DEFENSIVE")
        return modified

    def _mod_tank_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        opp_is_physical = self.physics._is_physical(opponent)
        my_def = active.base_stats.get('def', 0)
        my_spd = active.base_stats.get('spd', 0)
        is_right_def = (opp_is_physical and my_def >= my_spd) or (not opp_is_physical and my_spd > my_def)
        if not is_right_def:
            if "SWITCH_DEFENSIVE" in modified:
                modified.insert(0, "SWITCH_DEFENSIVE")
            if "PROTECT" in modified:
                modified.insert(0, "PROTECT")
            return modified
        if "STATUS" in modified:
            modified.insert(0, modified.pop(modified.index("STATUS")))
        if "ATTACK_TECH" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        if "HEAL" in modified:
            modified.insert(0, modified.pop(modified.index("HEAL")))
        if self._has_recovery(opponent) and "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        return modified

    def _mod_tank_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        has_delay = self._has_move(active, ['futuresight', 'doomdesire'])
        has_pivot = self._has_move(active, ['uturn', 'voltswitch', 'flipturn', 'teleport'])
        has_protect = self._has_move(active, ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker'])
        if has_delay and has_pivot:
            if "ATTACK_TECH" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
            if "ATTACK_PIVOT" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
            return modified
        if "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "STATUS" in modified:
            modified.insert(0, modified.pop(modified.index("STATUS")))
        if "HEAL" in modified:
            modified.insert(0, modified.pop(modified.index("HEAL")))
        if has_protect and "PROTECT" in modified:
            modified.insert(0, modified.pop(modified.index("PROTECT")))
        if "SWITCH_OFFENSIVE" in modified:
            modified.insert(0, modified.pop(modified.index("SWITCH_OFFENSIVE")))
        if "ATTACK_TECH" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        return modified

    def _mod_tank_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "STATUS" in modified:
            modified.insert(0, modified.pop(modified.index("STATUS")))
        if self._has_move(opponent, ['uturn', 'voltswitch', 'flipturn', 'teleport']) and "ATTACK_TECH" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        return modified

    def _mod_utility_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if is_threat and not is_faster:
            if "SWITCH_DEFENSIVE" in modified:
                modified.insert(0, "SWITCH_DEFENSIVE")
            if "ATTACK_TECH" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        if self._opponent_can_setup(opponent) and "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        elif "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        if is_faster and opp_hp_frac <= 0.4 and "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        return modified

    def _mod_utility_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "STATUS" in modified:
            modified.insert(0, modified.pop(modified.index("STATUS")))
        if "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        if "DEBUFF" in modified:
            modified.insert(0, modified.pop(modified.index("DEBUFF")))
        if "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        # NOTA: bloco original removia ATTACK_STRONG quando havia SWITCH_DEFENSIVE.
        if "ATTACK_STRONG" in modified and "SWITCH_DEFENSIVE" in modified:
            modified.remove("ATTACK_STRONG")
        return modified

    def _mod_utility_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        for cat in ["HAZARD", "STATUS", "FIELD_CONTROL", "ATTACK_PIVOT", "DEBUFF"]:
            if cat in modified:
                modified.insert(0, modified.pop(modified.index(cat)))
        return modified

    def _mod_escape(self, base, active, opponent, is_faster, my_role, battle):
        modified = base.copy()
        if self._is_active_best_remaining(active, opponent, battle):
            for cat in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]:
                if cat in modified:
                    modified.remove(cat)
            if is_faster:
                if "ATTACK_STRONG" in modified:
                    modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
                if "ATTACK_TECH" in modified:
                    modified.insert(1, modified.pop(modified.index("ATTACK_TECH")))
            elif my_role == Role.TANK:
                opp_is_physical = self.physics._is_physical(opponent)
                my_def = active.base_stats.get('def', 0)
                my_spd = active.base_stats.get('spd', 0)
                is_right_def = (opp_is_physical and my_def >= my_spd) or (not opp_is_physical and my_spd > my_def)
                if is_right_def and "STATUS" in modified:
                    modified.insert(0, modified.pop(modified.index("STATUS")))
                elif "ATTACK_STRONG" in modified:
                    modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
            else:
                if "ATTACK_STRONG" in modified:
                    modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
            modified.extend(["ATTACK_PIVOT", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"])
        return modified

    def _mod_lead(self, base, active, opponent, battle, is_faster):
        modified = base.copy()
        my_team = list(battle.team.values())
        weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce',
                           'solarpower', 'hydration', 'drought', 'drizzle', 'sandstream', 'snowwarning']
        team_needs_weather = any(str(m.ability).lower() in weather_abusers for m in my_team)
        avg_speed = sum(m.base_stats.get('spe', 50) for m in my_team) / len(my_team)
        team_needs_tr = avg_speed < 70
        needs_field_control = team_needs_weather or team_needs_tr
        weather_active = battle.weather is not None and len(battle.weather) > 0
        matchup = self.parser.get_matchup_state(active, opponent)
        matchup_lost = matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]
        matchup_won = matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]

        if matchup_lost:
            if is_faster and "ATTACK_PIVOT" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
            elif not is_faster and "SWITCH_DEFENSIVE" in modified:
                modified.insert(0, modified.pop(modified.index("SWITCH_DEFENSIVE")))
            return modified
        if needs_field_control and not weather_active:
            if "FIELD_CONTROL" in modified:
                modified.insert(0, modified.pop(modified.index("FIELD_CONTROL")))
        elif weather_active and matchup_won:
            if "HAZARD" in modified:
                modified.insert(0, modified.pop(modified.index("HAZARD")))
        if weather_active and is_faster:
            if "ATTACK_STRONG" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        return modified

    def _mod_wallbreak(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac):
        modified = base.copy()
        opponent_has_recovery = self._has_recovery(opponent)
        if opponent_has_recovery:
            if "STATUS" in modified:
                modified.insert(0, modified.pop(modified.index("STATUS")))
            if "ATTACK_TECH" in modified:
                modified.insert(1, modified.pop(modified.index("ATTACK_TECH")))
        if "BUFF" in modified and my_hp_frac >= 0.60:
            insert_idx = 2 if opponent_has_recovery else 0
            modified.insert(insert_idx, modified.pop(modified.index("BUFF")))
        if my_hp_frac < 0.40 and "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        return modified

    # ======================================================================
    # ORQUESTRADOR: produz o ranking final de intenções
    # ======================================================================

    def get_instinct_profile(self, battle, history=None):
        candidate_mask = self.masker.get_available_actions(battle)

        # CORRIGIDO: early-return agora devolve 5 valores (era 4 no monólito).
        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            primary = "SWITCH_DEFENSIVE" if battle.available_switches else "ATTACK_STRONG"
            return (primary, 1.0, [primary], candidate_mask, False)

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        my_role = self.physics.get_role(active)
        opp_role = self.physics.get_role(opp)
        matchup = self.parser.get_matchup_state(active, opp)
        my_spe = self.physics.estimate_stat(active, 'spe')
        opp_spe = self.physics.estimate_stat(opp, 'spe')
        is_faster = my_spe > opp_spe
        my_hp_frac = active.current_hp_fraction
        opp_hp_frac = opp.current_hp_fraction
        is_threat = self.is_threatening(active, opp)
        macro_context = self.parser.get_macro_context(battle)

        # Sub-camada 1: escolhe o modo
        if macro_context == "OPENING":
            mode = TacticalMode.LEAD
        else:
            mode = self._get_tactical_mode(matchup, my_role, opp_role, is_faster,
                                           my_hp_frac, opp_hp_frac, is_threat, active, opp)

        # Sub-camada 2: template base
        base_priorities = self.mode_templates[mode].copy()

        # Sub-camada 3: modificador de role
        if mode == TacticalMode.LEAD:
            priorities = self._mod_lead(base_priorities, active, opp, battle, is_faster)
        elif mode == TacticalMode.WALLBREAK:
            priorities = self._mod_wallbreak(base_priorities, active, opp, is_faster, my_hp_frac, opp_hp_frac)
        elif mode == TacticalMode.ESCAPE:
            priorities = self._mod_escape(base_priorities, active, opp, is_faster, my_role, battle)
        else:
            modifier_fn = self.role_modifiers.get((my_role, opp_role))
            if modifier_fn:
                priorities = modifier_fn(base_priorities, active, opp, is_faster, my_hp_frac, opp_hp_frac, is_threat)
            else:
                priorities = base_priorities

        # Filtragem pelo action mask + regras de bom-senso
        my_hp_crit = my_hp_frac <= 0.35
        my_hp_full = my_hp_frac >= 0.85
        ranking_list = []
        for intent in priorities:
            if intent in candidate_mask:
                if intent == "HEAL" and my_hp_full:
                    continue
                if intent == "BUFF" and my_hp_crit:
                    continue
                if intent == "STATUS" and opp.status is not None:
                    continue
                if intent not in ranking_list:
                    ranking_list.append(intent)

        # Hazards desvalorizam quando restam poucos oponentes
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        if opp_alive <= 2 and "HAZARD" in ranking_list:
            ranking_list.remove("HAZARD")
            ranking_list.append("HAZARD")

        # Barreiras (screens) reorganizam a macro-estratégia
        opp_side_conds = [str(k).upper() for k in battle.opponent_side_conditions.keys()]
        physical_blocked = 'REFLECT' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds
        special_blocked = 'LIGHT_SCREEN' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds
        if physical_blocked or special_blocked:
            benched_mons = [m for m in battle.team.values() if not m.fainted and not m.active]
            can_bypass = False
            if physical_blocked and not special_blocked:
                if any(self.physics.get_role(m) == Role.SWEEPER and not self.physics._is_physical(m) for m in benched_mons):
                    can_bypass = True
            elif special_blocked and not physical_blocked:
                if any(self.physics.get_role(m) == Role.SWEEPER and self.physics._is_physical(m) for m in benched_mons):
                    can_bypass = True
            base = ["CLEAN_HAZARD"]
            if can_bypass:
                boost_intents = base + ["SWITCH_OFFENSIVE", "ATTACK_PIVOT", "BUFF"]
            else:
                boost_intents = base + ["SWITCH_DEFENSIVE", "STATUS", "HEAL", "PROTECT", "DEBUFF"]
            for b_intent in reversed(boost_intents):
                if b_intent in ranking_list:
                    ranking_list.remove(b_intent)
                    ranking_list.insert(0, b_intent)

        # ------------------------------------------------------------------
        # REGRA GLOBAL 1: NAO BUFAR EM SITUACAO INSTAVEL
        # ------------------------------------------------------------------
        # Sintoma observado: o agente bufava em matchups VOLATILE ou piores, onde a
        # probabilidade de perder o Pokemon (e o buff) no turno seguinte e alta.
        # Bufar so compensa quando ha margem para aproveitar o buff.
        matchup_atual = self.parser.get_matchup_state(active, opp)
        matchups_inseguros = {
            MatchupState.VOLATILE, MatchupState.OFFENSIVE_DIS,
            MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS,
        }
        hp_frac = getattr(active, "current_hp_fraction", 1.0)
        if (matchup_atual in matchups_inseguros or hp_frac < 0.5) and "BUFF" in ranking_list:
            ranking_list.remove("BUFF")
            ranking_list.append("BUFF")

        # ------------------------------------------------------------------
        # REGRA GLOBAL 2: NAO DEITAR FORA UM BUFF JA CONQUISTADO
        # ------------------------------------------------------------------
        # Sintoma observado: trocas em excesso, perdendo boosts conquistados em turnos
        # anteriores. Um boost ofensivo/velocidade so existe enquanto o Pokemon estiver
        # em campo: trocar apaga-o. Com buff ativo e situacao sustentavel, atacar deve
        # subir e trocar deve descer.
        boosts = getattr(active, "boosts", {}) or {}
        buff_ativo = any(boosts.get(k, 0) > 0 for k in ("atk", "spa", "spe"))
        if buff_ativo and matchup_atual not in {MatchupState.CRITICAL_DIS} and hp_frac >= 0.35:
            for intent in reversed(["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_TECH"]):
                if intent in ranking_list:
                    ranking_list.remove(intent)
                    ranking_list.insert(0, intent)
            # Trocar apaga o buff: desce para o fim.
            for intent in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]:
                if intent in ranking_list:
                    ranking_list.remove(intent)
                    ranking_list.append(intent)

        # ------------------------------------------------------------------
        # REGRA GLOBAL 3: SEM BANCO, NAO EXISTE TROCA
        # ------------------------------------------------------------------
        # Se nao ha para onde trocar, manter intencoes de troca no topo faz o agente
        # perder turnos em acoes impossiveis (o executor cai no fallback).
        if not getattr(battle, "available_switches", None):
            for intent in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]:
                if intent in ranking_list:
                    ranking_list.remove(intent)
                    ranking_list.append(intent)

        # ------------------------------------------------------------------
        # DETECAO DE LETALIDADE
        # ------------------------------------------------------------------
        # Criterio PRIMARIO: dano REALMENTE observado no turno anterior.
        # A estimativa (estimate_damage_percent) desconhece EVs, IVs, item e nature do
        # adversario, logo tem erro grande. Se no turno anterior os MESMOS dois Pokemon
        # estavam em campo e usamos um ataque de dano, a diferenca de HP do oponente e
        # o dano REAL daquele confronto — facto medido, nao estimativa.
        #
        # So se aplica quando o confronto e identico (mesmo aliado, mesmo oponente);
        # se qualquer um trocou, a medicao nao transita e voltamos a estimativa.
        dano_observado = None
        if history:
            mesmo_ativo = (history.get('last_active_id') is not None and
                           history.get('last_active_id') == getattr(active, 'species', None))
            mesmo_opp = (history.get('last_opponent_id') is not None and
                         history.get('last_opponent_id') == getattr(opp, 'species', None))
            if mesmo_ativo and mesmo_opp and history.get('last_action_was_damage'):
                hp_antes = history.get('last_opp_hp')
                if hp_antes is not None:
                    delta = hp_antes - opp_hp_frac
                    if delta > 0.0:
                        dano_observado = delta

        has_lethal = False
        if "ATTACK_STRONG" in candidate_mask or "ATTACK_PREDICTIVE" in candidate_mask:
            if dano_observado is not None and dano_observado >= opp_hp_frac:
                # O dano que ja causamos neste confronto chega para o rematar.
                has_lethal = True
            else:
                for m in battle.available_moves:
                    if m.base_power > 0 and not self.masker.is_move_useless(m, opp, battle):
                        dmg = self.physics.estimate_damage_percent(m, active, opp, battle)
                        # Calibra a estimativa pelo que foi REALMENTE observado neste
                        # confronto: se estimamos 30% e causamos 45%, a estimativa esta
                        # subavaliada por um fator ~1.5 e corrigimo-la.
                        if dano_observado is not None:
                            est_anterior = self.physics.estimate_damage_percent(m, active, opp, battle)
                            if est_anterior > 0.01:
                                fator = dano_observado / est_anterior
                                fator = max(0.5, min(2.0, fator))   # trava contra outliers
                                dmg = dmg * fator
                        if dmg >= opp_hp_frac:
                            has_lethal = True
                            break
        if has_lethal:
            for atk in reversed(["ATTACK_PREDICTIVE", "ATTACK_STRONG"]):
                if atk in ranking_list:
                    ranking_list.remove(atk)
                    ranking_list.insert(0, atk)

        # Anti-fadiga de troca: se acabou de trocar, empurra trocas para baixo
        if history:
            # CORREÇÃO: last_action pode estar PRESENTE com valor None (primeiro turno
            # da batalha). O default de .get() só cobre a chave ausente, não o valor
            # None, por isso normalizamos explicitamente antes de indexar.
            last = history.get('last_action') or (None, None)
            prev_action = last[0]
            if prev_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"]:
                for sw in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"]:
                    if sw in ranking_list:
                        current_idx = ranking_list.index(sw)
                        ranking_list.remove(sw)
                        new_idx = min(len(ranking_list), current_idx + 2)
                        ranking_list.insert(new_idx, sw)

        # Rede de segurança
        confidence = 1.0
        if not ranking_list:
            confidence = 0.5
            atk_options = [a for a in candidate_mask if "ATTACK" in a]
            if atk_options:
                ranking_list.append(atk_options[0])
            elif candidate_mask:
                ranking_list.append(candidate_mask[0])
            else:
                ranking_list.append("ATTACK_STRONG")

        primary = ranking_list[0]
        return (primary, confidence, ranking_list, candidate_mask, has_lethal)
