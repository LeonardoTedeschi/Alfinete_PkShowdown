import traceback
import random
from enum import Enum

# =============================================================================
# DEFINIÇÕES DE DADOS
# =============================================================================
class Role(Enum):
    SPEED_SWEEPER = 1
    UTILITY = 2
    TANK_BULK = 3

class MatchupState(Enum):
    DOMINANT = 1       
    VOLATILE = 2       
    OFFENSIVE_ADV = 3  
    DEFENSIVE_ADV = 4  
    DEFENSIVE_DIS = 5  
    OFFENSIVE_DIS = 6  
    STALEMATE = 7      
    NEUTRAL = 8        
    CRITICAL_DIS = 9   

class MoveCategory(Enum):
    ATTACK_PHYSICAL = 1
    ATTACK_SPECIAL = 2
    SETUP_BUFF = 3
    STATUS_CTRL = 4
    HAZARD = 5
    RECOVERY = 6
    WEATHER = 7
    TEAM_CURE = 8      
    PROTECT = 9        
    DEBUFF = 10        
    STAT_CLEAN = 11
    UNKNOWN = 12


class InstinctCore:
    """
    O Cérebro Especialista (Regras Fixas e Lógicas de Seleção).
    Não executa jogadas no servidor, apenas analisa o campo e retorna os objetos ideais (Pokemon ou Move).
    """
    def __init__(self):
        pass

    def get_hp_bucket(self, pokemon):
        if not pokemon or pokemon.fainted: return "DEAD"
        hp = pokemon.current_hp_fraction
        if hp >= 0.8: return "FULL"
        if hp >= 0.5: return "MED"
        if hp >= 0.2: return "LOW"
        return "CRIT"

    def get_weather_state(self, battle):
        if "trickroom" in battle.fields: return "TRICK_ROOM"
        if battle.weather:
            w_name = next(iter(battle.weather)).name
            if w_name in ["HAIL", "SNOW"]: return "HAIL"
            if w_name == "SANDSTORM": return "SANDSTORM"
            if w_name in ["RAINDANCE", "PRIMORDIALSEA"]: return "RAIN"
            if w_name in ["SUNNYDAY", "DESOLATELAND"]: return "SUN"
        return "NORMAL"

    def get_speed_tier(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not active or not opponent: return "UNKNOWN"
        my_speed = self.estimate_stat(active, 'spe')
        opp_speed = self.estimate_stat(opponent, 'spe')
        if my_speed > opp_speed * 1.1: return "FASTER"
        if opp_speed > my_speed * 1.1: return "SLOWER"
        return "TIED"

    def get_state(self, battle):
        """Gera a tupla de 6 atributos de forma segura, sem usar blocos try/except cegos."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Se não há Pokémon ativo, retorna um estado neutro padrão viável
        if not active or not opponent:
            return ("UTILITY", "UTILITY", "NEUTRAL", "FULL", "NORMAL", "TIED")
        
        # A extração agora é direta, as funções internas já tratam os limites numéricos
        my_role = self.get_role(active).name
        opp_role = self.get_role(opponent).name
        matchup = self.get_matchup_state(active, opponent).name

        return (
            my_role,
            opp_role,
            matchup,
            self.get_hp_bucket(active),
            self.get_weather_state(battle),
            self.get_speed_tier(battle)
        )

    # =========================================================================
    # PERCEPÇÃO E ANÁLISE DE CAMPO
    # =========================================================================

    def get_role(self, pokemon) -> Role:
        if not pokemon: return Role.UTILITY
        b_spd = pokemon.base_stats.get('spe', 0)
        b_atk = pokemon.base_stats.get('atk', 0)
        b_spa = pokemon.base_stats.get('spa', 0)
        b_hp  = pokemon.base_stats.get('hp', 0)
        b_def = pokemon.base_stats.get('def', 0)
        b_res = pokemon.base_stats.get('spd', 0)

        if b_spd >= 80 and (b_atk >= 90 or b_spa >= 90):
            return Role.SPEED_SWEEPER
        if (b_hp >= 100 and (b_def >= 100 or b_res >= 100)) or (b_def >= 100 and b_res >= 100):
            return Role.TANK_BULK
        return Role.UTILITY

    def _get_speed_mod(self, pokemon):
        mod = 1.0
        if pokemon.status and 'PAR' in str(pokemon.status).upper(): mod *= 0.5
        stage = pokemon.boosts.get('spe', 0)
        if stage > 0: mod *= (1 + 0.5 * stage)
        elif stage < 0: mod *= (2 / (2 + abs(stage)))
        return mod

    def estimate_stat(self, pokemon, stat_name):
        if pokemon.stats and pokemon.stats.get(stat_name) is not None:
            val = pokemon.stats[stat_name]
            if stat_name == 'spe': return val * self._get_speed_mod(pokemon)
            modifier = pokemon.boosts.get(stat_name, 0)
            if modifier > 0: val *= (1 + 0.5 * modifier)
            elif modifier < 0: val *= (2 / (2 + abs(modifier)))
            return int(val)

        base = pokemon.base_stats.get(stat_name, 50)
        role = self.get_role(pokemon)
        def calc_max(b, is_hp=False): return int(b * 2 + 204) if is_hp else int((b * 2 + 99) * 1.1)
        def calc_min(b, is_hp=False): return int(b * 2 + 141) if is_hp else int(b * 2 + 36)

        estimated = 0
        if role == Role.SPEED_SWEEPER:
            if stat_name == 'spe' or stat_name in ['atk', 'spa']: estimated = calc_max(base)
            else: estimated = calc_min(base, stat_name=='hp')
        elif role == Role.TANK_BULK:
            if stat_name == 'hp': estimated = calc_max(base, True)
            elif stat_name in ['def', 'spd']: estimated = calc_max(base)
            else: estimated = calc_min(base)
        else:
            estimated = calc_max(base, stat_name=='hp') if stat_name == 'hp' else calc_min(base)

        if stat_name == 'spe': return estimated * self._get_speed_mod(pokemon)
        modifier = pokemon.boosts.get(stat_name, 0)
        if modifier > 0: estimated *= (1 + 0.5 * modifier)
        elif modifier < 0: estimated *= (2 / (2 + abs(modifier)))
        return int(estimated)

    def classify_move(self, move) -> MoveCategory:
        mid = move.id
        if mid in ['protect', 'detect', 'banefulbunker', 'spikyshield', 'kingsshield', 'silktrap']: return MoveCategory.PROTECT
        if mid in ['haze', 'clearsmog']: return MoveCategory.STAT_CLEAN
        if mid in ['healbell', 'aromatherapy']: return MoveCategory.TEAM_CURE
        if mid in ['snarl', 'strugglebug', 'confusray', 'fakeout', 'tickle', 'nobleroar', 'charm', 'partingshot']: return MoveCategory.DEBUFF
        if move.category.name == 'PHYSICAL': return MoveCategory.ATTACK_PHYSICAL
        if move.category.name == 'SPECIAL': return MoveCategory.ATTACK_SPECIAL
        if move.heal > 0 or mid in ['roost', 'recover', 'synthesis', 'softboiled', 'wish', 'moonlight', 'morning sun', 'slackoff']: return MoveCategory.RECOVERY
        if move.weather or mid in ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape']: return MoveCategory.WEATHER
        if mid in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'defog', 'rapidspin', 'mortalspin', 'tidyup']: return MoveCategory.HAZARD
        if move.boosts and move.target == 'self': return MoveCategory.SETUP_BUFF
        if move.status or (move.boosts and move.target == 'normal'): return MoveCategory.STATUS_CTRL
        return MoveCategory.UNKNOWN

    def is_move_useless(self, move, opp_pokemon, battle):
        if not opp_pokemon: return False
        opp_types = [str(t).upper() for t in opp_pokemon.types if t]
        move_id = move.id
        move_type = str(move.type).upper()
        
        opp_abilities = []
        if opp_pokemon.ability: opp_abilities = [str(opp_pokemon.ability).lower()]
        elif opp_pokemon.possible_abilities: opp_abilities = [str(a).lower() for a in opp_pokemon.possible_abilities]

        if 'wonderguard' in opp_abilities:
            if move.category.name in ['PHYSICAL', 'SPECIAL'] and move.base_power > 0:
                if opp_pokemon.damage_multiplier(move) < 2.0: return True
            if move.category.name == 'STATUS': return True 

        type_absorb_map = {
            'water': ['waterabsorb', 'dryskin', 'stormdrain'],
            'ground': ['levitate'],
            'grass': ['sapsipper'],
            'fire': ['flashfire'],
            'electric': ['voltabsorb', 'lightningrod', 'motordrive']
        }
        if move.category.name in ['PHYSICAL', 'SPECIAL'] and move.base_power > 0:
            if opp_pokemon.damage_multiplier(move) == 0: return True
            for immune_type, abilities in type_absorb_map.items():
                if move_type == immune_type.upper() and any(ab in opp_abilities for ab in abilities): return True

        if move.category.name == 'STATUS':
            if move_id in ['toxic', 'poisonpowder', 'poisongas'] and ('STEEL' in opp_types or 'POISON' in opp_types): return True
            if move_id == 'thunderwave' and ('GROUND' in opp_types or 'ELECTRIC' in opp_types): return True
            if move_id == 'willowisp' and 'FIRE' in opp_types: return True
            if move_id in ['leechseed', 'spore', 'sleeppowder', 'stunspore', 'ragepowder'] and 'GRASS' in opp_types: return True
            if any(ab in opp_abilities for ab in ['immunity']) and move_id in ['toxic', 'poisongas']: return True
            if any(ab in opp_abilities for ab in ['limber']) and move_id == 'thunderwave': return True
            if any(ab in opp_abilities for ab in ['insomnia', 'vitalspirit', 'sweetveil']) and move_id in ['spore', 'hypnosis', 'sleeppowder']: return True
            if any(ab in opp_abilities for ab in ['owntempo', 'oblivious']) and move_id in ['confuseray', 'swagger']: return True
            if any(ab in opp_abilities for ab in ['waterveil', 'waterbubble']) and move_id == 'willowisp': return True
            if any(ab in opp_abilities for ab in ['goodasgold', 'magicguard']): return True

        if move.priority > 0 and any(ab in opp_abilities for ab in ['dazzling', 'queenlymajesty', 'armortail']): return True
        if move.priority > 0 and 'psychicsurge' in opp_abilities and 'psychicterrain' in str(battle.fields): return True
        return False

    def get_matchup_state(self, my_mon, opp_mon) -> MatchupState:
        if not my_mon or not opp_mon: return MatchupState.NEUTRAL
        my_moves = [m for m in my_mon.moves.values() if m.base_power > 0]
        my_best_mult = max([opp_mon.damage_multiplier(move) for move in my_moves]) if my_moves else 1.0
        opp_best_mult = max([my_mon.damage_multiplier(type_) for type_ in opp_mon.types if type_]) if opp_mon.types else 1.0

        my_se, my_nve = my_best_mult > 1.5, my_best_mult < 0.9
        opp_se, opp_nve = opp_best_mult > 1.5, opp_best_mult < 0.9

        if my_se: return MatchupState.VOLATILE if opp_se else (MatchupState.DOMINANT if opp_nve else MatchupState.OFFENSIVE_ADV)
        if my_nve: return MatchupState.CRITICAL_DIS if opp_se else (MatchupState.STALEMATE if opp_nve else MatchupState.OFFENSIVE_DIS)
        return MatchupState.DEFENSIVE_DIS if opp_se else (MatchupState.DEFENSIVE_ADV if opp_nve else MatchupState.NEUTRAL)

    def is_threatening(self, my_mon, opp_mon):
        if not opp_mon or not my_mon: return False
        if opp_mon.boosts.get('atk', 0) >= 2 or opp_mon.boosts.get('spa', 0) >= 2: return True
        my_speed = self.estimate_stat(my_mon, 'spe')
        opp_speed = self.estimate_stat(opp_mon, 'spe')
        if my_mon.current_hp_fraction < 0.45 and opp_speed > my_speed:
            opp_atk = max(self.estimate_stat(opp_mon, 'atk'), self.estimate_stat(opp_mon, 'spa'))
            if opp_atk > 250: return True
        return False

    def is_hazard_already_set(self, move, battle):
        conditions = [str(k).upper() for k in battle.opponent_side_conditions.keys()]
        if move.id == 'stealthrock': return any('STEALTH_ROCK' in c for c in conditions)
        if move.id == 'stickyweb': return any('STICKY_WEB' in c for c in conditions)
        if move.id == 'toxicspikes':
            for k, v in battle.opponent_side_conditions.items():
                if 'TOXIC_SPIKES' in str(k).upper(): return v >= 2
            return False
        if move.id == 'spikes':
            for k, v in battle.opponent_side_conditions.items():
                if 'SPIKES' in str(k).upper() and 'TOXIC' not in str(k).upper(): return v >= 3
            return False
        return False

    # =========================================================================
    # ROTINAS DE PREVIEW E TROCA
    # =========================================================================

    def get_best_lead(self, battle):
        """Lógica intocada de abertura."""
        try:
            opp_team = list(battle.opponent_team.values())
            my_team = list(battle.team.values())
            
            if opp_team:
                avg_base_speed = sum(m.base_stats.get('spe', 50) for m in opp_team) / len(opp_team)
            else:
                avg_base_speed = 100
            
            is_slow_archetype = avg_base_speed < 80 
            weather_setters = ['drought', 'drizzle', 'sandstream', 'snowwarning']
            opp_has_weather = any(m.ability in weather_setters for m in opp_team)
            my_weather_setter = next((m for m in my_team if m.ability in weather_setters), None)

            predicted_lead = None
            if opp_has_weather:
                predicted_lead = next((m for m in opp_team if m.ability in weather_setters), opp_team[0])
            elif is_slow_archetype:
                 predicted_lead = min(opp_team, key=lambda m: m.base_stats.get('spe', 50))
            else:
                 if opp_team: predicted_lead = max(opp_team, key=lambda m: m.base_stats.get('spe', 50))

            if not predicted_lead and opp_team: predicted_lead = opp_team[0]
            
            best_lead = None
            if my_weather_setter:
                best_lead = my_weather_setter
            elif predicted_lead:
                def lead_score(m):
                    advantage = 10 if self.get_matchup_state(m, predicted_lead) in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV] else 0
                    if is_slow_archetype: return advantage
                    return advantage + m.stats.get('spe', 50)
                best_lead = max(battle.team.values(), key=lead_score)
            else:
                best_lead = list(battle.team.values())[0]

            try: lead_index = my_team.index(best_lead) + 1
            except ValueError: lead_index = 1
            rest_indices = [str(i + 1) for i in range(len(my_team)) if i + 1 != lead_index]
            team_order = str(lead_index) + "".join(rest_indices)
            return f"/team {team_order}"
        except Exception: return "/team 123456"

    def get_best_switch(self, battle):
        """A matriz de pontuação exata para seleção de trocas, sem o envio via create_order."""
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        
        weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration', 'raindish', 'icebody']
        active_weather = battle.weather
        
        def get_score(candidate):
            score = 0
            role = self.get_role(candidate)
            matchup = self.get_matchup_state(candidate, opponent)
            
            score += candidate.current_hp_fraction * 100
            
            # Prioridade de Troca por Clima
            cand_abi = str(candidate.ability).lower() if candidate.ability else ""
            
            if active_weather: 
                if cand_abi in weather_abusers:
                    if 'sun' in str(active_weather).lower() and cand_abi in ['chlorophyll', 'solarpower']: score += 1000
                    if 'rain' in str(active_weather).lower() and cand_abi in ['swiftswim', 'hydration', 'raindish']: score += 1000
                    if 'sand' in str(active_weather).lower() and cand_abi in ['sandrush', 'sandforce']: score += 1000
                    if ('hail' in str(active_weather).lower() or 'snow' in str(active_weather).lower()) and cand_abi in ['slushrush', 'icebody']: score += 1000

            if matchup == MatchupState.DOMINANT: score += 500
            elif matchup == MatchupState.OFFENSIVE_ADV: score += 300
            elif matchup == MatchupState.DEFENSIVE_ADV: score += 100
            elif matchup == MatchupState.DEFENSIVE_DIS: score -= 200
            elif matchup == MatchupState.OFFENSIVE_DIS: score -= 300
            elif matchup == MatchupState.CRITICAL_DIS: score -= 500
            elif matchup == MatchupState.STALEMATE: score -= 50
            
            if role == Role.TANK_BULK: score += 50
            return score

        if candidates:
            return max(candidates, key=get_score)
        return None

    # =========================================================================
    # LÓGICA DE INTENÇÃO E MATRIZES
    # =========================================================================

    def get_intent_list(self, battle) -> list:
        """Avalia a batalha e retorna a lista de intenções (Ex: ['BUFF', 'ATTACK'])."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        if not active or not opponent:
            return ["SWITCH"]

        my_role = self.get_role(active)
        opp_role = self.get_role(opponent)
        matchup = self.get_matchup_state(active, opponent)
        is_faster = self.estimate_stat(active, 'spe') >= self.estimate_stat(opponent, 'spe')
        is_threatening = self.is_threatening(active, opponent)

        if my_role == Role.SPEED_SWEEPER:
            if opp_role == Role.SPEED_SWEEPER: return self._matrix_sweeper_vs_sweeper(is_faster, matchup, is_threatening)
            elif opp_role == Role.UTILITY: return self._matrix_sweeper_vs_utility(matchup, is_faster, is_threatening)
            else: return self._matrix_sweeper_vs_tank(active, opponent, matchup)
        elif my_role == Role.UTILITY:
            return self._matrix_utility_logic(active, opponent, matchup, opp_role, is_faster)
        else:
            return self._matrix_tank_logic(active, opponent, matchup, opp_role)

    def _matrix_sweeper_vs_sweeper(self, is_faster, matchup, is_threatening):
        if is_faster:
            if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]: return ["BUFF", "ATTACK"] if not is_threatening else ["ATTACK"]
            if matchup in [MatchupState.STALEMATE, MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]: return ["SWITCH"]
            if matchup in [MatchupState.NEUTRAL, MatchupState.DEFENSIVE_ADV]: return ["SWITCH"] if is_threatening else ["BUFF", "ATTACK"]
        else: 
            if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]: return ["SWITCH"] if is_threatening else ["ATTACK"]
            return ["SWITCH"]
        return ["ATTACK"]

    def _matrix_sweeper_vs_utility(self, matchup, is_faster, is_threatening):
        if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]: return ["ATTACK", "BUFF"]
        if matchup == MatchupState.STALEMATE: return ["SWITCH"]
        if matchup == MatchupState.DEFENSIVE_ADV: return ["BUFF", "ATTACK"]
        if matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS]: return ["SWITCH"]
        if matchup == MatchupState.VOLATILE: return ["ATTACK"] if is_faster else ["SWITCH"]
        return ["ATTACK"]

    def _matrix_sweeper_vs_tank(self, my_mon, opp_mon, matchup):
        my_atk, my_spa = self.estimate_stat(my_mon, 'atk'), self.estimate_stat(my_mon, 'spa')
        opp_def, opp_spd = self.estimate_stat(opp_mon, 'def'), self.estimate_stat(opp_mon, 'spd')
        bate_no_forte = ("PHYSICAL" if my_atk > my_spa else "SPECIAL") == ("PHYSICAL" if opp_def > opp_spd else "SPECIAL")

        if matchup == MatchupState.DOMINANT: return ["BUFF", "ATTACK"] if bate_no_forte else ["ATTACK"]
        if matchup == MatchupState.STALEMATE: return ["SWITCH"]
        if matchup == MatchupState.DEFENSIVE_ADV: return ["BUFF", "ATTACK"]
        if matchup in [MatchupState.VOLATILE, MatchupState.OFFENSIVE_ADV]: return ["ATTACK"]
        if matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.NEUTRAL, MatchupState.OFFENSIVE_DIS]: return ["SWITCH"]
        return ["ATTACK"]

    def _matrix_utility_logic(self, active, opponent, matchup, opp_role, is_faster):
        if opp_role in [Role.SPEED_SWEEPER, Role.UTILITY]:
            if matchup == MatchupState.DOMINANT: return ["HAZARD", "STATUS", "CLEAN", "ATTACK", "DEBUFF", "HEAL", "TEAM_CURE"]
            if matchup == MatchupState.STALEMATE: return ["STATUS", "HAZARD", "HEAL", "ATTACK"]
            if matchup == MatchupState.VOLATILE:
                if is_faster: return ["ATTACK"]
                opp_is_physical = self.estimate_stat(opponent, 'atk') > self.estimate_stat(opponent, 'spa')
                is_right_def = (opp_is_physical and active.base_stats.get('def',0) >= active.base_stats.get('spd',0)) or (not opp_is_physical and active.base_stats.get('spd',0) > active.base_stats.get('def',0))
                return ["ATTACK"] if is_right_def else ["SWITCH"]
            if matchup == MatchupState.OFFENSIVE_ADV: return ["HAZARD", "STATUS", "ATTACK", "DEBUFF", "CLEAN", "HEAL", "TEAM_CURE"]
            if matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS]: return ["SWITCH"]
            if matchup == MatchupState.NEUTRAL: return ["HAZARD", "STATUS", "ATTACK", "DEBUFF", "CLEAN", "HEAL", "TEAM_CURE"]
        return ["HAZARD", "STATUS", "CLEAN", "ATTACK", "DEBUFF", "HEAL", "TEAM_CURE"]

    def _matrix_tank_logic(self, active, opponent, matchup, opp_role):
        opp_is_physical = self.estimate_stat(opponent, 'atk') > self.estimate_stat(opponent, 'spa')
        is_right_def = (opp_is_physical and active.base_stats.get('def',0) >= active.base_stats.get('spd',0)) or (not opp_is_physical and active.base_stats.get('spd',0) > active.base_stats.get('def',0))
        
        if opp_role == Role.TANK_BULK: return ["STATUS", "HEAL", "PROTECT", "ATTACK", "TEAM_CURE"]
        if opp_role == Role.UTILITY: return ["STATUS", "ATTACK", "HEAL_50", "PROTECT", "TEAM_CURE"]
        if opp_role == Role.SPEED_SWEEPER:
            if matchup == MatchupState.STALEMATE: return ["STATUS", "HAZARD", "ATTACK"]
            if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV, MatchupState.NEUTRAL]: return ["STATUS", "TEAM_CURE", "ATTACK", "HEAL", "PROTECT"]
            if matchup in [MatchupState.VOLATILE, MatchupState.DEFENSIVE_DIS]: return ["STATUS", "HEAL_50", "PROTECT", "ATTACK"] if is_right_def else ["STATUS", "HEAL_50", "PROTECT", "STAT_CLEAN", "SWITCH"]
            if matchup == MatchupState.CRITICAL_DIS: return ["STATUS", "HEAL_50", "PROTECT", "STAT_CLEAN", "SWITCH"]
        return ["ATTACK"]

    # =========================================================================
    # TRADUÇÃO DE AÇÃO (Seleciona o Objeto Correto)
    # =========================================================================

    def get_best_execution_object(self, priority_list, battle):
        """
        Dada uma lista de intenções (ex: ['ATTACK', 'SWITCH']), retorna o objeto
        exato (Move ou Pokemon) que cumpre a melhor opção viável.
        A seleção de ataque letal/dano bruto otimizada está aqui.
        """
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if hasattr(active, 'effects') and any("TAUNT" in str(e).upper() for e in active.effects):
            priority_list = ["ATTACK"]

        for action in priority_list:
            if action in ["HEAL", "HEAL_50"]:
                threshold = 0.55 if action == "HEAL_50" else (0.85 if self.is_threatening(active, opponent) else 0.55)
                if active.current_hp_fraction <= threshold:
                    move = next((m for m in active.moves.values() if self.classify_move(m) == MoveCategory.RECOVERY), None)
                    if move: return move

            if action == "STATUS":
                for m in active.moves.values():
                    if self.classify_move(m) == MoveCategory.STATUS_CTRL and not opponent.status and not self.is_move_useless(m, opponent, battle): 
                        return m

            if action == "HAZARD":
                move = next((m for m in active.moves.values() if self.classify_move(m) == MoveCategory.HAZARD and m.id not in ['defog', 'rapidspin']), None)
                if move and not self.is_hazard_already_set(move, battle): return move

            if action in ["CLEAN", "STAT_CLEAN"]:
                 move = next((m for m in active.moves.values() if self.classify_move(m) in [MoveCategory.STAT_CLEAN, MoveCategory.HAZARD] and m.id in ['defog', 'rapidspin', 'haze', 'clearsmog']), None)
                 if move:
                     if move.id in ['defog', 'rapidspin'] and battle.side_conditions: return move
                     if move.id in ['haze', 'clearsmog'] and any(v > 0 for v in opponent.boosts.values()): return move

            if action == "ATTACK":
                valid_moves = [m for m in battle.available_moves if m.base_power > 0 and not self.is_move_useless(m, opponent, battle)]
                if valid_moves:
                    lethal_moves = []
                    for m in valid_moves:
                        stab = 1.5 if m.type in active.types else 1.0
                        power = m.base_power * stab * opponent.damage_multiplier(m)
                        if opponent.current_hp_fraction < 0.35 and power > 60:
                            lethal_moves.append(m)
                    
                    if lethal_moves:
                        return max(lethal_moves, key=lambda m: m.accuracy if m.accuracy != True else 100)
                    else:
                        return max(valid_moves, key=lambda m: m.base_power * (1.5 if m.type in active.types else 1.0) * opponent.damage_multiplier(m))

            if action == "PROTECT":
                move = next((m for m in active.moves.values() if self.classify_move(m) == MoveCategory.PROTECT), None)
                if move: return move

            if action == "TEAM_CURE":
                move = next((m for m in active.moves.values() if self.classify_move(m) == MoveCategory.TEAM_CURE), None)
                if move and any(mon.status for mon in battle.team.values()): return move

            if action == "DEBUFF":
                move = next((m for m in active.moves.values() if self.classify_move(m) == MoveCategory.DEBUFF), None)
                if move: return move

            if action == "BUFF":
                move = next((m for m in active.moves.values() if self.classify_move(m) == MoveCategory.SETUP_BUFF), None)
                if move: return move

            if action == "SWITCH":
                if battle.available_switches:
                    return self.get_best_switch(battle)

        # Fallback se nada servir
        if battle.available_switches:
            return self.get_best_switch(battle)
        return None