import random
import traceback # Para logar erros sem travar o bot
from enum import Enum
from poke_env.player import Player

# --- 1. DEFINIÇÕES DE DADOS ---

class Role(Enum):
    SPEED_SWEEPER = 1
    UTILITY = 2
    TANK_BULK = 3

class MatchupState(Enum):
    DOMINANT = 1       # Eu SE / Ele NVE
    VOLATILE = 2       # Eu SE / Ele SE
    OFFENSIVE_ADV = 3  # Eu SE / Ele Neutro
    DEFENSIVE_ADV = 4  # Eu Neutro / Ele NVE (NOVO)
    DEFENSIVE_DIS = 5  # Eu Neutro / Ele SE
    OFFENSIVE_DIS = 6  # Eu NVE / Ele Neutro (NOVO)
    STALEMATE = 7      # Eu NVE / Ele NVE (NOVO)
    NEUTRAL = 8        # Neutro / Neutro
    CRITICAL_DIS = 9   # Eu NVE / Ele SE

class MoveCategory(Enum):
    ATTACK_PHYSICAL = 1
    ATTACK_SPECIAL = 2
    SETUP_BUFF = 3
    STATUS_CTRL = 4
    HAZARD = 5
    RECOVERY = 6
    WEATHER = 7
    TEAM_CURE = 8      # Heal Bell, Aromatherapy
    PROTECT = 9        # Protect, Detect
    DEBUFF = 10        # Snarl, Charm, etc
    STAT_CLEAN = 11    # Haze, Clear Smog
    UNKNOWN = 12

# --- 2. O CÉREBRO DO BOT ---

class InstinctBot(Player):
    
    # =========================================================================
    # PERCEPÇÃO & MEMÓRIA
    # =========================================================================

    def _get_role(self, pokemon) -> Role:
        """Classifica a Role baseada nos Base Stats fixos da espécie."""
        b_spd = pokemon.base_stats['spe']
        b_atk = pokemon.base_stats['atk']
        b_spa = pokemon.base_stats['spa']
        b_hp  = pokemon.base_stats['hp']
        b_def = pokemon.base_stats['def']
        b_res = pokemon.base_stats['spd']

        # 1. SWEEPER (Rápido e Ofensivo)
        if b_spd >= 80 and (b_atk >= 90 or b_spa >= 90):
            return Role.SPEED_SWEEPER

        # 2. TANK (Resistente)
        is_high_hp_tank = (b_hp >= 100 and (b_def >= 100 or b_res >= 100))
        is_wall_tank = (b_def >= 100 and b_res >= 100)

        if is_high_hp_tank or is_wall_tank:
            return Role.TANK_BULK

        # 3. RESTO (Utility/Pivot)
        return Role.UTILITY

    def _get_speed_mod(self, pokemon):
        mod = 1.0
        if pokemon.status and 'PAR' in str(pokemon.status).upper(): 
            mod *= 0.5
        stage = pokemon.boosts.get('spe', 0)
        if stage > 0: mod *= (1 + 0.5 * stage)
        elif stage < 0: mod *= (2 / (2 + abs(stage)))
        return mod

    def _estimate_stat(self, pokemon, stat_name):
        """Calcula ou Estima os Stats numéricos."""
        # 1. Se for MEU Pokémon, retorna o valor exato
        if pokemon.stats and pokemon.stats.get(stat_name) is not None:
            val = pokemon.stats[stat_name]
            if stat_name == 'spe': return val * self._get_speed_mod(pokemon)
            
            modifier = pokemon.boosts.get(stat_name, 0)
            if modifier > 0: val *= (1 + 0.5 * modifier)
            elif modifier < 0: val *= (2 / (2 + abs(modifier)))
            return int(val)

        # 2. Se for INIMIGO, Inferência baseada na Role
        base = pokemon.base_stats[stat_name]
        role = self._get_role(pokemon)
        
        def calc_max(b, is_hp=False):
            if is_hp: return int(b * 2 + 204)
            return int((b * 2 + 99) * 1.1)

        def calc_min(b, is_hp=False):
            if is_hp: return int(b * 2 + 141)
            return int(b * 2 + 36)

        estimated = 0
        if role == Role.SPEED_SWEEPER:
            if stat_name == 'spe': estimated = calc_max(base)
            elif stat_name == 'atk' and pokemon.base_stats['atk'] >= pokemon.base_stats['spa']: estimated = calc_max(base)
            elif stat_name == 'spa' and pokemon.base_stats['spa'] > pokemon.base_stats['atk']: estimated = calc_max(base)
            else: estimated = calc_min(base, stat_name=='hp')
        
        elif role == Role.TANK_BULK:
            if stat_name == 'hp': estimated = calc_max(base, is_hp=True)
            elif stat_name == 'def' and pokemon.base_stats['def'] >= pokemon.base_stats['spd']: estimated = calc_max(base)
            elif stat_name == 'spd' and pokemon.base_stats['spd'] > pokemon.base_stats['def']: estimated = calc_max(base)
            else: estimated = calc_min(base)

        else: # Utility
            if stat_name == 'hp': estimated = calc_max(base, is_hp=True)
            else: estimated = calc_min(base) 

        if stat_name == 'spe': return estimated * self._get_speed_mod(pokemon)
        
        modifier = pokemon.boosts.get(stat_name, 0)
        if modifier > 0: estimated *= (1 + 0.5 * modifier)
        elif modifier < 0: estimated *= (2 / (2 + abs(modifier)))
        
        return int(estimated)

    def _classify_move(self, move) -> MoveCategory:
        mid = move.id
        if mid in ['protect', 'detect', 'banefulbunker', 'spikyshield', 'kingsshield', 'silktrap']: return MoveCategory.PROTECT
        if mid in ['haze', 'clearsmog']: return MoveCategory.STAT_CLEAN
        if mid in ['healbell', 'aromatherapy']: return MoveCategory.TEAM_CURE
        if mid in ['snarl', 'strugglebug', 'confusray', 'fakeout', 'tickle', 'nobleroar', 'charm', 'partingshot']: return MoveCategory.DEBUFF

        if move.category.name == 'PHYSICAL': return MoveCategory.ATTACK_PHYSICAL
        if move.category.name == 'SPECIAL': return MoveCategory.ATTACK_SPECIAL
        
        if move.heal > 0 or mid in ['roost', 'recover', 'synthesis', 'softboiled', 'wish', 'moonlight', 'morning sun', 'slackoff']:
            return MoveCategory.RECOVERY
        if move.weather or mid in ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape']:
            return MoveCategory.WEATHER
        if mid in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'defog', 'rapidspin', 'mortalspin', 'tidyup']:
            return MoveCategory.HAZARD
        if move.boosts and move.target == 'self':
            return MoveCategory.SETUP_BUFF
        if move.status or (move.boosts and move.target == 'normal'):
            return MoveCategory.STATUS_CTRL
            
        return MoveCategory.UNKNOWN

    def _is_move_useless(self, move, opp_pokemon):
        """Verifica imunidades de tipo, status e habilidades especiais."""
        if not opp_pokemon: return False
        
        opp_types = [str(t).upper() for t in opp_pokemon.types if t]
        move_id = move.id
        move_type = str(move.type).upper()
        
        # --- CORREÇÃO: Tratamento seguro de habilidades ---
        opp_abilities = []
        if opp_pokemon.ability:
            opp_abilities = [str(opp_pokemon.ability).lower()]
        elif opp_pokemon.possible_abilities:
            opp_abilities = [str(a).lower() for a in opp_pokemon.possible_abilities]

        # 1. Wonder Guard (Imune a tudo que não for Super Efetivo)
        if 'wonderguard' in opp_abilities:
            if move.category.name in ['PHYSICAL', 'SPECIAL'] and move.base_power > 0:
                if opp_pokemon.damage_multiplier(move) < 2.0:
                    return True
            if move.category.name == 'STATUS':
                return True # Wonder Guard bloqueia status move na maioria das regras

        # 2. Imunidades de Absorção de Tipo (Habilidades)
        type_absorb_map = {
            'water': ['waterabsorb', 'dryskin', 'stormdrain'],
            'ground': ['levitate'],
            'grass': ['sapsipper'],
            'fire': ['flashfire'],
            'electric': ['voltabsorb', 'lightningrod', 'motordrive']
        }
        
        if move.category.name in ['PHYSICAL', 'SPECIAL'] and move.base_power > 0:
            # Imunidade Natural (Tipo x Tipo)
            if opp_pokemon.damage_multiplier(move) == 0: return True
            
            # Imunidade por Habilidade
            for immune_type, abilities in type_absorb_map.items():
                if move_type == immune_type.upper():
                    if any(ab in opp_abilities for ab in abilities): return True

        # 3. Imunidades a Status Específicos
        if move.category.name == 'STATUS':
            # Imunidades Gerais de Tipo
            if move_id in ['toxic', 'poisonpowder', 'poisongas'] and ('STEEL' in opp_types or 'POISON' in opp_types): return True
            if move_id == 'thunderwave' and ('GROUND' in opp_types or 'ELECTRIC' in opp_types): return True
            if move_id == 'willowisp' and 'FIRE' in opp_types: return True
            if move_id in ['leechseed', 'spore', 'sleeppowder', 'stunspore', 'ragepowder'] and 'GRASS' in opp_types: return True
            
            # Imunidades por Habilidade
            if any(ab in opp_abilities for ab in ['immunity']) and move_id in ['toxic', 'poisongas']: return True
            if any(ab in opp_abilities for ab in ['limber']) and move_id == 'thunderwave': return True
            if any(ab in opp_abilities for ab in ['insomnia', 'vitalspirit', 'sweetveil']) and move_id in ['spore', 'hypnosis', 'sleeppowder']: return True
            if any(ab in opp_abilities for ab in ['owntempo', 'oblivious']) and move_id in ['confuseray', 'swagger']: return True
            if any(ab in opp_abilities for ab in ['waterveil', 'waterbubble']) and move_id == 'willowisp': return True
            if any(ab in opp_abilities for ab in ['magmaarmor']) and 'freeze' in move_id: return True
            if any(ab in opp_abilities for ab in ['goodasgold']): return True # Gholdengo
            if any(ab in opp_abilities for ab in ['overcoat']) and move_id in ['spore', 'sleeppowder', 'stunspore', 'poisonpowder']: return True
            if any(ab in opp_abilities for ab in ['magicguard']) and move_id in ['toxic', 'willowisp']: return True

        # 4. Imunidade a Prioridade (Dazzling/Queenly Majesty/Armor Tail)
        if move.priority > 0:
            if any(ab in opp_abilities for ab in ['dazzling', 'queenlymajesty', 'armortail']): return True
            # CORREÇÃO: Usa a lista segura opp_abilities aqui também
            if 'psychicsurge' in opp_abilities and 'psychicterrain' in str(battle.fields): return True

        # 5. Imunidade a Stat Drops / Troca Forçada
        if any(ab in opp_abilities for ab in ['clearbody', 'whitesmoke', 'fullmetalbody']):
            if move.boosts and move.target == 'normal' and any(v < 0 for v in move.boosts.values()): return True
        
        if any(ab in opp_abilities for ab in ['suctioncups']):
            if move_id in ['roar', 'whirlwind']: return True

        return False

    def _get_matchup_state(self, my_mon, opp_mon) -> MatchupState:
        # Analisa MEUS ataques
        my_moves = [m for m in my_mon.moves.values() if m.base_power > 0]
        if not my_moves:
            my_best_mult = 1.0
        else:
            my_best_mult = max([opp_mon.damage_multiplier(move) for move in my_moves])

        # Analisa TIPOS do oponente (STAB presumido)
        opp_best_mult = 0.0
        for type_ in opp_mon.types:
             if type_:
                 multiplier = my_mon.damage_multiplier(type_)
                 if multiplier > opp_best_mult: opp_best_mult = multiplier
        if opp_best_mult == 0.0: opp_best_mult = 1.0 

        # Classificação Expandida
        my_se = my_best_mult > 1.5
        my_neutral = 0.9 <= my_best_mult <= 1.5
        my_nve = my_best_mult < 0.9
        
        opp_se = opp_best_mult > 1.5
        opp_neutral = 0.9 <= opp_best_mult <= 1.5
        opp_nve = opp_best_mult < 0.9

        # Matriz 3x3 de estados atualizada
        if my_se:
            if opp_se: return MatchupState.VOLATILE       # Perigo Mútuo
            if opp_neutral: return MatchupState.OFFENSIVE_ADV
            if opp_nve: return MatchupState.DOMINANT      # Vantagem Total
        
        if my_neutral:
            if opp_se: return MatchupState.DEFENSIVE_DIS
            if opp_neutral: return MatchupState.NEUTRAL
            if opp_nve: return MatchupState.DEFENSIVE_ADV # NOVO: Eu Neutro / Ele NVE
            
        if my_nve:
            if opp_se: return MatchupState.CRITICAL_DIS   # Desvantagem Total
            if opp_neutral: return MatchupState.OFFENSIVE_DIS
            if opp_nve: return MatchupState.STALEMATE     # NOVO: Empate NVE vs NVE

        return MatchupState.NEUTRAL

    def _is_threatening(self, my_mon, opp_mon):
        if opp_mon.boosts.get('atk', 0) >= 2 or opp_mon.boosts.get('spa', 0) >= 2:
            return True
        
        my_speed = self._estimate_stat(my_mon, 'spe')
        opp_speed = self._estimate_stat(opp_mon, 'spe')
        
        if my_mon.current_hp_fraction < 0.45 and opp_speed > my_speed:
            opp_atk = max(self._estimate_stat(opp_mon, 'atk'), self._estimate_stat(opp_mon, 'spa'))
            if opp_atk > 250:
                return True
        return False

    # --- SAFETY CHECKS ---
    def _is_hazard_already_set(self, move, battle):
        conditions = [str(k).upper() for k in battle.opponent_side_conditions.keys()]
        
        if move.id == 'stealthrock':
            return any('STEALTH_ROCK' in c for c in conditions)
        if move.id == 'stickyweb':
            return any('STICKY_WEB' in c for c in conditions)
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
    # MÓDULO DE ABERTURA (LEAD)
    # =========================================================================

    def teampreview(self, battle):
        # PROTEÇÃO CONTRA CRASH NO PREVIEW
        try:
            opp_team = list(battle.opponent_team.values())
            my_team = list(battle.team.values())
            
            if opp_team:
                avg_base_speed = sum(m.base_stats['spe'] for m in opp_team) / len(opp_team)
            else:
                avg_base_speed = 100
            
            is_slow_archetype = avg_base_speed < 80 

            # Inteligência de Clima
            weather_setters = ['drought', 'drizzle', 'sandstream', 'snowwarning']
            opp_has_weather = any(m.ability in weather_setters for m in opp_team)
            my_weather_setter = next((m for m in my_team if m.ability in weather_setters), None)

            predicted_lead = None
            if opp_has_weather:
                predicted_lead = next((m for m in opp_team if m.ability in weather_setters), opp_team[0])
            elif is_slow_archetype:
                 predicted_lead = min(opp_team, key=lambda m: m.base_stats['spe'])
            else:
                 if opp_team:
                    predicted_lead = max(opp_team, key=lambda m: m.base_stats['spe'])

            if not predicted_lead and opp_team: predicted_lead = opp_team[0]
            
            best_lead = None
            if my_weather_setter:
                best_lead = my_weather_setter # Prioriza meu setter
            elif predicted_lead:
                def lead_score(m):
                    advantage = 10 if self._get_matchup_state(m, predicted_lead) in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV] else 0
                    if is_slow_archetype: return advantage
                    return advantage + m.stats['spe']

                best_lead = max(battle.team.values(), key=lead_score)
            else:
                best_lead = list(battle.team.values())[0]

            try:
                lead_index = my_team.index(best_lead) + 1
            except ValueError:
                lead_index = 1
                
            rest_indices = [str(i + 1) for i in range(len(my_team)) if i + 1 != lead_index]
            team_order = str(lead_index) + "".join(rest_indices)
            
            return f"/team {team_order}"
        except Exception:
            return "/team 123456"

    # =========================================================================
    # MÓDULO DE COMBATE
    # =========================================================================

    def choose_move(self, battle):
        # BLINDAGEM CONTRA CRASH: Try/Except global para o turno
        try:
            # --- ADIÇÃO: TRATAMENTO DE U-TURN / VOLT SWITCH / FORCE SWITCH ---
            # Se o servidor exigir troca (bool ou lista), trocamos imediatamente.
            switch_forced = False
            if isinstance(battle.force_switch, list):
                switch_forced = any(battle.force_switch)
            else:
                switch_forced = bool(battle.force_switch)

            if switch_forced:
                return self._choose_switch(battle)

            active = battle.active_pokemon
            opponent = battle.opponent_active_pokemon
            
            if active.fainted:
                return self._choose_switch(battle)

            # Coleta de Dados
            my_role = self._get_role(active)
            opp_role = self._get_role(opponent)
            matchup = self._get_matchup_state(active, opponent)
            
            my_speed = self._estimate_stat(active, 'spe')
            opp_speed = self._estimate_stat(opponent, 'spe')
            is_faster = my_speed >= opp_speed
            
            is_threatening = self._is_threatening(active, opponent)
            
            # --- ROTEAMENTO PARA AS MATRIZES ---
            action_list = ["ATTACK"]

            if my_role == Role.SPEED_SWEEPER:
                if opp_role == Role.SPEED_SWEEPER:
                    action_list = self._matrix_sweeper_vs_sweeper(is_faster, matchup, is_threatening)
                elif opp_role == Role.UTILITY:
                    # IMPLEMENTAÇÃO: NOVA MATRIZ Sweeper vs Utility
                    action_list = self._matrix_sweeper_vs_utility(matchup, is_faster, is_threatening)
                else:
                    action_list = self._matrix_sweeper_vs_tank(active, opponent, matchup)
            
            elif my_role == Role.UTILITY:
                action_list = self._matrix_utility_logic(active, opponent, matchup, opp_role, is_faster)
            
            elif my_role == Role.TANK_BULK:
                action_list = self._matrix_tank_logic(active, opponent, matchup, opp_role)
            
            else:
                action_list = ["ATTACK"]

            # Envia a lista para o executor
            return self._execute_action(action_list, battle)
            
        except Exception as e:
            # EM CASO DE ERRO DESCONHECIDO (CRASH DO CÉREBRO)
            print(f"InstinctBot Panic (Erro Interno): {e}")
            traceback.print_exc()
            return self.choose_random_move(battle)

    # --- MATRIZES (Retornam Listas de Prioridade) ---

    def _matrix_sweeper_vs_sweeper(self, is_faster, matchup, is_threatening):
        if is_faster:
            if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
                return ["BUFF", "ATTACK"] if not is_threatening else ["ATTACK"]
            if matchup in [MatchupState.STALEMATE, MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]:
                return ["SWITCH"]
            if matchup in [MatchupState.NEUTRAL, MatchupState.DEFENSIVE_ADV]:
                return ["SWITCH"] if is_threatening else ["BUFF", "ATTACK"]
        else: 
            if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
                return ["SWITCH"] if is_threatening else ["ATTACK"]
            return ["SWITCH"]
        return ["ATTACK"]

    def _matrix_sweeper_vs_utility(self, matchup, is_faster, is_threatening):
        # NOVA MATRIZ: Sweepers odeiam levar status de Utility Pokémon.
        if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
            return ["ATTACK", "BUFF"] # Mata antes que ele use status
        
        # Em Stalemate (NVE vs NVE), Utility ganha com o tempo. Sai fora.
        if matchup == MatchupState.STALEMATE:
            return ["SWITCH"]
            
        # Defensive Adv: Se ele não ameaça dano, Sweeper pode tentar buffar.
        if matchup == MatchupState.DEFENSIVE_ADV:
            return ["BUFF", "ATTACK"]
            
        if matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS]:
            return ["SWITCH"]
            
        if matchup == MatchupState.VOLATILE:
            return ["ATTACK"] if is_faster else ["SWITCH"]
            
        return ["ATTACK"]

    def _matrix_sweeper_vs_tank(self, my_mon, opp_mon, matchup):
        my_atk = self._estimate_stat(my_mon, 'atk')
        my_spa = self._estimate_stat(my_mon, 'spa')
        my_atk_type = "PHYSICAL" if my_atk > my_spa else "SPECIAL"
        
        opp_def = self._estimate_stat(opp_mon, 'def')
        opp_spd = self._estimate_stat(opp_mon, 'spd')
        opp_strong_def = "PHYSICAL" if opp_def > opp_spd else "SPECIAL"
        
        bate_no_forte = my_atk_type == opp_strong_def

        if matchup == MatchupState.DOMINANT:
            return ["BUFF", "ATTACK"] if bate_no_forte else ["ATTACK"]
        
        if matchup == MatchupState.STALEMATE:
            return ["SWITCH"] # Sweeper não quebra Tank em Stalemate
            
        if matchup == MatchupState.DEFENSIVE_ADV:
            return ["BUFF", "ATTACK"] # Setup livre
            
        if matchup in [MatchupState.VOLATILE, MatchupState.OFFENSIVE_ADV]:
            return ["ATTACK"]
        if matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.NEUTRAL, MatchupState.OFFENSIVE_DIS]:
            return ["SWITCH"]
        return ["ATTACK"]

    def _matrix_utility_logic(self, active, opponent, matchup, opp_role, is_faster):
        if opp_role in [Role.SPEED_SWEEPER, Role.UTILITY]:
            if matchup == MatchupState.DOMINANT:
                return ["HAZARD", "STATUS", "CLEAN", "ATTACK", "DEBUFF", "HEAL", "TEAM_CURE"]
            
            if matchup == MatchupState.STALEMATE:
                return ["STATUS", "HAZARD", "HEAL", "ATTACK"]
            
            if matchup == MatchupState.VOLATILE:
                if is_faster: return ["ATTACK"]
                opp_is_physical = self._estimate_stat(opponent, 'atk') > self._estimate_stat(opponent, 'spa')
                is_right_def = (opp_is_physical and active.base_stats['def'] >= active.base_stats['spd']) or (not opp_is_physical and active.base_stats['spd'] > active.base_stats['def'])
                return ["ATTACK"] if is_right_def else ["SWITCH"]
            
            if matchup == MatchupState.OFFENSIVE_ADV:
                return ["HAZARD", "STATUS", "ATTACK", "DEBUFF", "CLEAN", "HEAL", "TEAM_CURE"]
            if matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS]:
                return ["SWITCH"]
            if matchup == MatchupState.NEUTRAL:
                return ["HAZARD", "STATUS", "ATTACK", "DEBUFF", "CLEAN", "HEAL", "TEAM_CURE"]
        
        return ["HAZARD", "STATUS", "CLEAN", "ATTACK", "DEBUFF", "HEAL", "TEAM_CURE"]

    def _matrix_tank_logic(self, active, opponent, matchup, opp_role):
        opp_is_physical = self._estimate_stat(opponent, 'atk') > self._estimate_stat(opponent, 'spa')
        is_right_def = (opp_is_physical and active.base_stats['def'] >= active.base_stats['spd']) or (not opp_is_physical and active.base_stats['spd'] > active.base_stats['def'])

        if opp_role == Role.TANK_BULK:
            return ["STATUS", "HEAL", "PROTECT", "ATTACK", "TEAM_CURE"]
        
        if opp_role == Role.UTILITY:
            return ["STATUS", "ATTACK", "HEAL_50", "PROTECT", "TEAM_CURE"]

        if opp_role == Role.SPEED_SWEEPER:
            if matchup == MatchupState.STALEMATE:
                return ["STATUS", "HAZARD", "ATTACK"] # Tank ganha por desgaste
                
            if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV, MatchupState.NEUTRAL]:
                return ["STATUS", "TEAM_CURE", "ATTACK", "HEAL", "PROTECT"]
            
            if matchup in [MatchupState.VOLATILE, MatchupState.DEFENSIVE_DIS]:
                if is_right_def:
                    return ["STATUS", "HEAL_50", "PROTECT", "ATTACK"]
                else:
                    return ["STATUS", "HEAL_50", "PROTECT", "STAT_CLEAN", "SWITCH"]
            
            if matchup == MatchupState.CRITICAL_DIS:
                return ["STATUS", "HEAL_50", "PROTECT", "STAT_CLEAN", "SWITCH"]
        
        return ["ATTACK"]

    # =========================================================================
    # EXECUTOR (LOOP INTELIGENTE)
    # =========================================================================

    def _execute_action(self, priority_list, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        if isinstance(priority_list, str): priority_list = [priority_list]

        if hasattr(active, 'effects') and any("TAUNT" in str(e).upper() for e in active.effects):
            priority_list = ["ATTACK"]

        for action in priority_list:
            
            # 1. CURA
            if action == "HEAL" or action == "HEAL_50":
                threshold = 0.55 if action == "HEAL_50" else (0.85 if self._is_threatening(active, opponent) else 0.55)
                if active.current_hp_fraction <= threshold:
                    move = next((m for m in active.moves.values() if self._classify_move(m) == MoveCategory.RECOVERY), None)
                    if move: return self.create_order(move)

            # 2. STATUS
            if action == "STATUS":
                moves = [m for m in active.moves.values() if self._classify_move(m) == MoveCategory.STATUS_CTRL]
                for m in moves:
                    if not opponent.status and not self._is_move_useless(m, opponent): return self.create_order(m)

            # 3. HAZARD
            if action == "HAZARD":
                move = next((m for m in active.moves.values() if self._classify_move(m) == MoveCategory.HAZARD and m.id not in ['defog', 'rapidspin']), None)
                if move and not self._is_hazard_already_set(move, battle): return self.create_order(move)

            # 4. LIMPEZA / HAZE
            if action == "CLEAN" or action == "STAT_CLEAN":
                 move = next((m for m in active.moves.values() if self._classify_move(m) in [MoveCategory.STAT_CLEAN, MoveCategory.HAZARD] and m.id in ['defog', 'rapidspin', 'haze', 'clearsmog']), None)
                 if move:
                     if move.id in ['defog', 'rapidspin'] and battle.side_conditions: return self.create_order(move)
                     if move.id in ['haze', 'clearsmog'] and any(v > 0 for v in opponent.boosts.values()): return self.create_order(move)

            # 5. ATTACK (NOVO: LETHAL CHECK)
            if action == "ATTACK":
                valid_moves = [m for m in battle.available_moves if m.base_power > 0 and not self._is_move_useless(m, opponent)]
                if valid_moves:
                    # Tenta achar golpe letal (KO)
                    lethal_moves = []
                    for m in valid_moves:
                        stab = 1.5 if m.type in active.types else 1.0
                        power = m.base_power * stab * opponent.damage_multiplier(m)
                        # Se o oponente tem pouco HP e o golpe é forte
                        if opponent.current_hp_fraction < 0.35 and power > 60:
                            lethal_moves.append(m)
                    
                    if lethal_moves:
                        # Prioriza o mais preciso para finalizar
                        best = max(lethal_moves, key=lambda m: m.accuracy if m.accuracy != True else 100)
                    else:
                        # Dano bruto máximo
                        best = max(valid_moves, key=lambda m: m.base_power * opponent.damage_multiplier(m))
                    return self.create_order(best)

            # 6. PROTECT
            if action == "PROTECT":
                move = next((m for m in active.moves.values() if self._classify_move(m) == MoveCategory.PROTECT), None)
                if move: return self.create_order(move)

            # 7. TEAM_CURE
            if action == "TEAM_CURE":
                move = next((m for m in active.moves.values() if self._classify_move(m) == MoveCategory.TEAM_CURE), None)
                if move and any(mon.status for mon in battle.team.values()): return self.create_order(move)

            # 8. DEBUFF
            if action == "DEBUFF":
                move = next((m for m in active.moves.values() if self._classify_move(m) == MoveCategory.DEBUFF), None)
                if move: return self.create_order(move)

            # 9. BUFF
            if action == "BUFF":
                move = next((m for m in active.moves.values() if self._classify_move(m) == MoveCategory.SETUP_BUFF), None)
                if move: return self.create_order(move)

            # 10. SWITCH
            if action == "SWITCH":
                if battle.available_switches: return self._choose_switch(battle)

        # Fallback
        if battle.available_switches:
            return self._choose_switch(battle)
        
        return self.choose_random_move(battle)

    def _choose_switch(self, battle):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        
        weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration', 'raindish', 'icebody']
        active_weather = battle.weather
        
        def get_score(candidate):
            score = 0
            role = self._get_role(candidate)
            matchup = self._get_matchup_state(candidate, opponent)
            
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
            best = max(candidates, key=get_score)
            return self.create_order(best)
        
        return self.choose_random_move(battle)