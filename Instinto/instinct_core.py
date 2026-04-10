import traceback
import random
from enum import Enum


# =============================================================================
# 1. DEFINIÇÕES DE DADOS 
# =============================================================================

class Role(Enum):
    SWEEPER = 1
    UTILITY = 2
    TANK = 3

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
    CLEAN_HAZARD = 8      
    PROTECT = 9        
    DEBUFF = 10        
    STAT_CLEAN = 11
    HEAL_STATUS = 12
    PHAZE = 13
    FIELD_CONTROL = 14
    UNKNOWN = 15

# =============================================================================
# 2. O CÉREBRO ESPECIALISTA (CORE)
# =============================================================================

class InstinctCore:
    def __init__(self):
        # =====================================================================
        # MATRIZ TÁTICA : Mapeamento de MyRole -> OppRole -> MatchupState
        # O ápice da heurística: Responde não apenas à vantagem, mas ao TIPO de inimigo.
        # =====================================================================
        self.tactical_matrix = {
            Role.SWEEPER: {
                Role.SWEEPER: {
                    MatchupState.DOMINANT:      ["ATTACK_PREDICTIVE", "ATTACK_STRONG"],
                    MatchupState.OFFENSIVE_ADV: ["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_TECH"],
                    MatchupState.DEFENSIVE_ADV: ["ATTACK_STRONG", "ATTACK_PIVOT"],
                    MatchupState.NEUTRAL:       ["ATTACK_STRONG", "ATTACK_TECH"],
                    MatchupState.STALEMATE:     ["ATTACK_STRONG", "ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.VOLATILE:      ["ATTACK_STRONG", "ATTACK_PIVOT"],
                    MatchupState.OFFENSIVE_DIS: ["ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"]
                },
                Role.TANK: {
                    MatchupState.DOMINANT:      ["BUFF", "ATTACK_TECH", "ATTACK_STRONG"],
                    MatchupState.OFFENSIVE_ADV: ["BUFF", "ATTACK_TECH", "ATTACK_STRONG"],
                    MatchupState.DEFENSIVE_ADV: ["BUFF", "ATTACK_STRONG", "ATTACK_PIVOT"],
                    MatchupState.NEUTRAL:       ["BUFF", "ATTACK_TECH", "ATTACK_PIVOT"],
                    MatchupState.STALEMATE:     ["ATTACK_TECH", "ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.VOLATILE:      ["ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.OFFENSIVE_DIS: ["ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"]
                },
                Role.UTILITY: {
                    MatchupState.DOMINANT:      ["ATTACK_PREDICTIVE", "ATTACK_STRONG"],
                    MatchupState.OFFENSIVE_ADV: ["ATTACK_STRONG", "ATTACK_TECH"],
                    MatchupState.DEFENSIVE_ADV: ["ATTACK_STRONG", "ATTACK_PIVOT"],
                    MatchupState.NEUTRAL:       ["ATTACK_STRONG", "ATTACK_TECH"],
                    MatchupState.STALEMATE:     ["ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.VOLATILE:      ["ATTACK_STRONG", "ATTACK_PIVOT"],
                    MatchupState.OFFENSIVE_DIS: ["ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"]
                }
            },
            Role.TANK: {
                Role.SWEEPER: {
                    MatchupState.DOMINANT:      ["STATUS", "DEBUFF", "ATTACK_STRONG", "HEAL"],
                    MatchupState.OFFENSIVE_ADV: ["STATUS", "ATTACK_STRONG", "HEAL"],
                    MatchupState.DEFENSIVE_ADV: ["STATUS", "HEAL", "ATTACK_STRONG"],
                    MatchupState.NEUTRAL:       ["STATUS", "HEAL", "ATTACK_TECH", "SWITCH_DEFENSIVE"],
                    MatchupState.STALEMATE:     ["STATUS", "HEAL", "ATTACK_TECH", "DEBUFF"],
                    MatchupState.VOLATILE:      ["PROTECT", "STATUS", "SWITCH_DEFENSIVE"],
                    MatchupState.OFFENSIVE_DIS: ["SWITCH_DEFENSIVE", "PROTECT", "HEAL"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "PROTECT", "HEAL"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "PROTECT"]
                },
                Role.TANK: {
                    MatchupState.DOMINANT:      ["HAZARD", "ATTACK_TECH", "STATUS", "HEAL"],
                    MatchupState.OFFENSIVE_ADV: ["ATTACK_TECH", "HAZARD", "STATUS", "HEAL"],
                    MatchupState.DEFENSIVE_ADV: ["HAZARD", "ATTACK_TECH", "HEAL"],
                    MatchupState.NEUTRAL:       ["ATTACK_TECH", "STATUS", "HAZARD", "HEAL"],
                    MatchupState.STALEMATE:     ["ATTACK_TECH", "STATUS", "HEAL"],
                    MatchupState.VOLATILE:      ["HEAL", "STATUS", "PROTECT"],
                    MatchupState.OFFENSIVE_DIS: ["SWITCH_DEFENSIVE", "HEAL"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "HEAL", "PROTECT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "PROTECT"]
                },
                Role.UTILITY: {
                    MatchupState.DOMINANT:      ["ATTACK_TECH", "STATUS", "HAZARD", "HEAL"],
                    MatchupState.OFFENSIVE_ADV: ["ATTACK_TECH", "STATUS", "HEAL"],
                    MatchupState.DEFENSIVE_ADV: ["ATTACK_TECH", "STATUS", "HEAL"],
                    MatchupState.NEUTRAL:       ["ATTACK_TECH", "HEAL", "SWITCH_OFFENSIVE"],
                    MatchupState.STALEMATE:     ["ATTACK_TECH", "HEAL", "SWITCH_OFFENSIVE"],
                    MatchupState.VOLATILE:      ["HEAL", "PROTECT", "SWITCH_OFFENSIVE"],
                    MatchupState.OFFENSIVE_DIS: ["SWITCH_OFFENSIVE", "HEAL"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_OFFENSIVE", "PROTECT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_OFFENSIVE", "PROTECT"]
                }
            },
            Role.UTILITY: {
                Role.SWEEPER: {
                    MatchupState.DOMINANT:      ["STATUS", "HAZARD", "DEBUFF", "ATTACK_PIVOT"],
                    MatchupState.OFFENSIVE_ADV: ["STATUS", "HAZARD", "ATTACK_PIVOT"],
                    MatchupState.DEFENSIVE_ADV: ["HAZARD", "STATUS", "ATTACK_PIVOT"],
                    MatchupState.NEUTRAL:       ["STATUS", "HAZARD", "ATTACK_PIVOT"],
                    MatchupState.STALEMATE:     ["ATTACK_PIVOT", "STATUS", "SWITCH_DEFENSIVE"],
                    MatchupState.VOLATILE:      ["ATTACK_PIVOT", "DEBUFF", "SWITCH_DEFENSIVE"],
                    MatchupState.OFFENSIVE_DIS: ["ATTACK_PIVOT", "SWITCH_DEFENSIVE"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "ATTACK_PIVOT", "PROTECT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"]
                },
                Role.TANK: {
                    MatchupState.DOMINANT:      ["DEBUFF", "ATTACK_TECH", "HAZARD", "ATTACK_PIVOT"],
                    MatchupState.OFFENSIVE_ADV: ["DEBUFF", "ATTACK_TECH", "HAZARD", "ATTACK_PIVOT"],
                    MatchupState.DEFENSIVE_ADV: ["DEBUFF", "ATTACK_TECH", "HAZARD", "ATTACK_PIVOT"],
                    MatchupState.NEUTRAL:       ["DEBUFF", "ATTACK_TECH", "ATTACK_PIVOT"],
                    MatchupState.STALEMATE:     ["ATTACK_TECH", "ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.VOLATILE:      ["ATTACK_PIVOT", "DEBUFF", "SWITCH_OFFENSIVE"],
                    MatchupState.OFFENSIVE_DIS: ["ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"]
                },
                Role.UTILITY: {
                    MatchupState.DOMINANT:      ["HAZARD", "DEBUFF", "ATTACK_TECH", "ATTACK_PIVOT"],
                    MatchupState.OFFENSIVE_ADV: ["HAZARD", "ATTACK_TECH", "ATTACK_PIVOT"],
                    MatchupState.DEFENSIVE_ADV: ["HAZARD", "DEBUFF", "ATTACK_PIVOT"],
                    MatchupState.NEUTRAL:       ["ATTACK_TECH", "HAZARD", "ATTACK_PIVOT"],
                    MatchupState.STALEMATE:     ["ATTACK_TECH", "ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.VOLATILE:      ["ATTACK_PIVOT", "DEBUFF", "ATTACK_TECH"],
                    MatchupState.OFFENSIVE_DIS: ["ATTACK_PIVOT", "SWITCH_OFFENSIVE"],
                    MatchupState.DEFENSIVE_DIS: ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"],
                    MatchupState.CRITICAL_DIS:  ["SWITCH_DEFENSIVE", "ATTACK_PIVOT"]
                }
            }
        }
    # =========================================================================
    # EXTRAÇÃO DE ESTADO PARA O Q-LEARNING
    # =========================================================================

    def get_hp_bucket(self, pokemon):
        if not pokemon or pokemon.fainted: return "DEAD"
        hp = pokemon.current_hp_fraction
        if hp >= 0.8: return "FULL"
        if hp >= 0.5: return "MED"
        if hp >= 0.2: return "LOW"
        return "CRIT"

    def get_opp_hp_bucket(self, pokemon):
        if not pokemon or pokemon.fainted: return "DEAD"
        hp = pokemon.current_hp_fraction
        if hp >= 0.5: return "HIGH"
        if hp >= 0.2: return "MID"
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

    def get_status_state(self, pokemon):
        if not pokemon or pokemon.fainted: return "CLEAN"
        if pokemon.status: return "AFFLICTED"
        return "CLEAN"

    def get_boost_state(self, pokemon):
        if not pokemon or not pokemon.boosts: return "NEUTRAL"
        relevant_boosts = [v for k, v in pokemon.boosts.items() if k in ['atk', 'def', 'spa', 'spd', 'spe']]
        if not relevant_boosts: return "NEUTRAL"
        if any(v > 0 for v in relevant_boosts): return "BUFFED"
        if any(v < 0 for v in relevant_boosts): return "DEBUFF"
        return "NEUTRAL"

    def get_hazard_state(self, side_conditions):
        if not side_conditions: return "CLEAR"
        cond_strings = [str(k).upper() for k in side_conditions.keys()]
        hazards = ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']
        if any(h in cond for cond in cond_strings for h in hazards): return "SET"
        return "CLEAR"

    def get_mechanic_state(self, battle):
        """14ª Dimensão Binária: Mecânica disponível ou já gasta/inexistente."""
        if battle.can_tera or battle.can_mega_evolve or battle.can_z_move or battle.can_dynamax:
            return "MEC_AVAIL"
        return "MEC_USED"

    def get_available_actions(self, battle):
        available = []
        
        # Novas opções de troca
        if battle.available_switches:
            available.extend(["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"])

        if battle.available_moves:
            for move in battle.available_moves:
                cat = self.classify_move(move)
                action = None
                
                # Novas opções de ataque ramificadas (Filtradas estritamente)
                if cat in [MoveCategory.ATTACK_PHYSICAL, MoveCategory.ATTACK_SPECIAL]: 
                    if "ATTACK_STRONG" not in available:
                        available.extend(["ATTACK_STRONG", "ATTACK_PREDICTIVE"])
                        
                    # Só oferece PIVOT se realmente tiver o golpe
                    if "ATTACK_PIVOT" not in available:
                        if move.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']:
                            available.append("ATTACK_PIVOT")
                            
                    if "ATTACK_TECH" not in available:
                        tech_moves = [
                            # Desarme e Punição
                            'knockoff', 'foulplay', 'thief',
                            
                            # Status Garantido ou Altíssima Chance
                            'nuzzle', 'scald', 'discharge', 'lavaplume',
                            
                            # Dano Fixo / Residual (Quebradores de Tank)
                            'saltcure', 'superfang', 'naturesmadness', 'ruination', 'seismictoss', 'nightshade',
                            
                            # Controle de Status do Oponente (Drops garantidos)
                            'icywind', 'electroweb', 'rocktomb', 'bulldoze', # Derrubam Velocidade
                            'snarl', 'mysticalfire', 'strugglebug',          # Derrubam Sp. Attack
                            
                            # Utilidade de Turno 1 e Quebra de Telas
                            'fakeout', 'brickbreak', 'psychicfangs',
                            
                            # Uso de Defesa como Ataque (Exclusivo de Tanks)
                            'bodypress'
                        ]
                        if move.id in tech_moves:
                            available.append("ATTACK_TECH")
                    continue
                    
                elif cat == MoveCategory.SETUP_BUFF: action = "BUFF"
                elif cat == MoveCategory.STATUS_CTRL: action = "STATUS"
                elif cat == MoveCategory.RECOVERY: action = "HEAL"
                elif cat == MoveCategory.CLEAN_HAZARD: action = "CLEAN_HAZARD"
                elif cat == MoveCategory.PROTECT: action = "PROTECT"
                elif cat == MoveCategory.DEBUFF: action = "DEBUFF"
                elif cat == MoveCategory.STAT_CLEAN: action = "STAT_CLEAN"
                elif cat == MoveCategory.HEAL_STATUS: action = "HEAL_STATUS"
                elif cat == MoveCategory.PHAZE: action = "PHAZE"
                elif cat == MoveCategory.FIELD_CONTROL: action = "FIELD_CONTROL"
                elif cat == MoveCategory.HAZARD:
                    if self.is_hazard_already_set(move, battle): continue
                    action = "HAZARD"
                
                if action and action not in available:
                    available.append(action)

        # Fallback de segurança
        return available if available else ["ATTACK_STRONG"]

        # Adiciona a macro binária de mecânica se for possível atacar
        if "ATTACK" in available:
            if battle.can_tera or battle.can_mega_evolve or battle.can_z_move or battle.can_dynamax:
                available.append("ATTACK_MEC")
                    
        # Fallback de Segurança
        if not available:
            if battle.available_switches:
                available.append("SWITCH")
            else:
                available.append("ATTACK")
                
        return available

    def get_state(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active or not opponent:
            print("[ERRO - INSTINTO] get_state chamado em arena inválida (Falta ativo ou oponente).")
            return ("UNKNOWN",) * 14
        
        my_role = self.get_role(active).name
        opp_role = self.get_role(opponent).name
        matchup = self.get_matchup_state(active, opponent).name

        return (
            my_role,
            opp_role,
            matchup,
            self.get_hp_bucket(active),                 
            self.get_opp_hp_bucket(opponent),          
            self.get_weather_state(battle),
            self.get_speed_tier(battle),
            self.get_status_state(active),              
            self.get_status_state(opponent),            
            self.get_boost_state(active),               
            self.get_boost_state(opponent),             
            self.get_hazard_state(battle.side_conditions),         
            self.get_hazard_state(battle.opponent_side_conditions),
            self.get_mechanic_state(battle) 
        )

    # =========================================================================
    # PERCEPÇÃO & MEMÓRIA
    # =========================================================================

    def get_role(self, pokemon) -> Role:
        if not pokemon: return Role.UTILITY
        b_atk = pokemon.base_stats.get('atk', 0)
        b_spa = pokemon.base_stats.get('spa', 0)
        b_hp  = pokemon.base_stats.get('hp', 0)
        b_def = pokemon.base_stats.get('def', 0)
        b_spd = pokemon.base_stats.get('spd', 0)

        # Agrupa Sweepers Rápidos e Wallbreakers
        if b_atk >= 100 or b_spa >= 100: return Role.SWEEPER
        if b_hp >= 80 and (b_def >= 100 or b_spd >= 100): return Role.TANK
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
        if role == Role.SWEEPER:
            if stat_name == 'spe': estimated = calc_max(base)
            elif stat_name == 'atk' and pokemon.base_stats.get('atk', 0) >= pokemon.base_stats.get('spa', 0): estimated = calc_max(base)
            elif stat_name == 'spa' and pokemon.base_stats.get('spa', 0) > pokemon.base_stats.get('atk', 0): estimated = calc_max(base)
            else: estimated = calc_min(base, stat_name=='hp')
        elif role == Role.TANK:
            if stat_name == 'hp': estimated = calc_max(base, is_hp=True)
            elif stat_name == 'def' and pokemon.base_stats.get('def', 0) >= pokemon.base_stats.get('spd', 0): estimated = calc_max(base)
            elif stat_name == 'spd' and pokemon.base_stats.get('spd', 0) > pokemon.base_stats.get('def', 0): estimated = calc_max(base)
            else: estimated = calc_min(base)
        else:
            if stat_name == 'hp': estimated = calc_max(base, is_hp=True)
            else: estimated = calc_min(base) 

        if stat_name == 'spe': return estimated * self._get_speed_mod(pokemon)
        
        modifier = pokemon.boosts.get(stat_name, 0)
        if modifier > 0: estimated *= (1 + 0.5 * modifier)
        elif modifier < 0: estimated *= (2 / (2 + abs(modifier)))
        
        return int(estimated)

    def classify_move(self, move):
        move_id = move.id
        if move_id in ['haze', 'clearsmog']: return MoveCategory.STAT_CLEAN
        if move_id in ['aromatherapy', 'healbell', 'junglehealing']: return MoveCategory.HEAL_STATUS
        if move_id in ['roar', 'whirlwind', 'dragontail', 'circlethrow']: return MoveCategory.PHAZE
        if move_id in ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape', 'trickroom', 'tailwind', 'electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain']: return MoveCategory.FIELD_CONTROL
        if move_id in ['defog', 'rapidspin', 'mortalspin', 'courtchange']: return MoveCategory.CLEAN_HAZARD
        if move_id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']: return MoveCategory.HAZARD
        if move_id in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark', 'silktrap', 'safeguard']: return MoveCategory.PROTECT
        if move_id in ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun', 'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap']: return MoveCategory.RECOVERY

        if move.category.name == "STATUS":
            if move.heal: return MoveCategory.RECOVERY
            if move.status: return MoveCategory.STATUS_CTRL
            if move.boosts:
                if any(v > 0 for v in move.boosts.values()): return MoveCategory.SETUP_BUFF
                if any(v < 0 for v in move.boosts.values()): return MoveCategory.DEBUFF
            return MoveCategory.STATUS_CTRL 

        if move.category.name == "PHYSICAL": return MoveCategory.ATTACK_PHYSICAL
        if move.category.name == "SPECIAL": return MoveCategory.ATTACK_SPECIAL

        return MoveCategory.UNKNOWN

    def is_move_useless(self, move, opp_pokemon, battle):
        if not opp_pokemon: return False
        
        # CORREÇÃO: Corta o prefixo "Type." e pega apenas o nome do elemento
        opp_types = [str(t).split('.')[-1].upper() for t in opp_pokemon.types if t]
        move_id = move.id
        move_type = str(move.type).split('.')[-1].upper() if move.type else "UNKNOWN"
        
        opp_abilities = []
        if opp_pokemon.ability: opp_abilities = [str(opp_pokemon.ability).lower()]
        elif opp_pokemon.possible_abilities: opp_abilities = [str(a).lower() for a in opp_pokemon.possible_abilities]

        if 'wonderguard' in opp_abilities:
            if move.category.name in ['PHYSICAL', 'SPECIAL'] and move.base_power > 0:
                if opp_pokemon.damage_multiplier(move) < 2.0: return True
            if move.category.name == 'STATUS': return True 

        # Mapa de Habilidades de Absorção/Imunidade
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
                if move_type == immune_type.upper():
                    if any(ab in opp_abilities for ab in abilities): return True

        if move.category.name == 'STATUS':
            if move.status and opp_pokemon.status: 
                return True

            if move_id in ['toxic', 'poisonpowder', 'poisongas'] and ('STEEL' in opp_types or 'POISON' in opp_types): return True
            if move_id == 'thunderwave' and ('GROUND' in opp_types or 'ELECTRIC' in opp_types): return True
            if move_id == 'willowisp' and 'FIRE' in opp_types: return True
            if move_id in ['leechseed', 'spore', 'sleeppowder', 'stunspore', 'ragepowder'] and 'GRASS' in opp_types: return True
            if any(ab in opp_abilities for ab in ['magicbounce']) and move.target in ['normal', 'allAdjacentFoes', 'foeSide']: return True
            if any(ab in opp_abilities for ab in ['immunity']) and move_id in ['toxic', 'poisongas']: return True
            if any(ab in opp_abilities for ab in ['limber']) and move_id == 'thunderwave': return True
            if any(ab in opp_abilities for ab in ['goodasgold']): return True 
            if any(ab in opp_abilities for ab in ['magicguard']) and move_id in ['toxic', 'willowisp']: return True

        return False

    def get_matchup_state(self, my_mon, opp_mon) -> MatchupState:
        if not my_mon or not opp_mon: return MatchupState.NEUTRAL
        
        my_moves = [m for m in my_mon.moves.values() if m.base_power > 0]
        my_best_mult = max([opp_mon.damage_multiplier(move) for move in my_moves]) if my_moves else 1.0

        opp_best_mult = 0.0
        for type_ in opp_mon.types:
             if type_:
                 multiplier = my_mon.damage_multiplier(type_)
                 if multiplier > opp_best_mult: opp_best_mult = multiplier
        if opp_best_mult == 0.0: opp_best_mult = 1.0 

        my_se = my_best_mult > 1.5
        my_neutral = 0.9 <= my_best_mult <= 1.5
        my_nve = my_best_mult < 0.9
        
        opp_se = opp_best_mult > 1.5
        opp_neutral = 0.9 <= opp_best_mult <= 1.5
        opp_nve = opp_best_mult < 0.9

        if my_se:
            if opp_se: return MatchupState.VOLATILE       
            if opp_neutral: return MatchupState.OFFENSIVE_ADV
            if opp_nve: return MatchupState.DOMINANT      
        if my_neutral:
            if opp_se: return MatchupState.DEFENSIVE_DIS
            if opp_neutral: return MatchupState.NEUTRAL
            if opp_nve: return MatchupState.DEFENSIVE_ADV 
        if my_nve:
            if opp_se: return MatchupState.CRITICAL_DIS   
            if opp_neutral: return MatchupState.OFFENSIVE_DIS
            if opp_nve: return MatchupState.STALEMATE     

        return MatchupState.NEUTRAL

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
        # 1. Identifica os hazards válidos e mapeia o ID do golpe para o nome da condição
        hazard_types = ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']
        move_to_hazard = {
            'stealthrock': 'STEALTH_ROCK',
            'stickyweb': 'STICKY_WEB',
            'spikes': 'SPIKES',
            'toxicspikes': 'TOXIC_SPIKES'
        }
        
        target_hazard = move_to_hazard.get(move.id)
        if not target_hazard:
            return False # Fallback: Se não for um hazard reconhecido, permite o uso
            
        current_hazards = {}
        for condition, layers in battle.opponent_side_conditions.items():
            cond_str = str(condition).upper()
            for haz in hazard_types:
                # O bloqueio 'TOXIC' not in cond_str é necessário apenas se a string crua for usada,
                # mas com a checagem exata abaixo, separamos perfeitamente Spikes de Toxic Spikes.
                if haz in cond_str:
                    if haz == 'SPIKES' and 'TOXIC' in cond_str:
                        continue # Pula para não contabilizar Toxic Spikes como Spikes normais
                    
                    # Garante que layers seja lido como inteiro (o padrão do poke-env)
                    current_hazards[haz] = int(layers) if isinstance(layers, int) else 1
                    break
                    
        if len(current_hazards) >= 2 and target_hazard not in current_hazards:
            return True 
            
        # 4. Regras Específicas de Acúmulo (Stacking)
        if target_hazard == 'STEALTH_ROCK':
            return 'STEALTH_ROCK' in current_hazards
            
        if target_hazard == 'STICKY_WEB':
            return 'STICKY_WEB' in current_hazards
            
        if target_hazard == 'SPIKES':
            # Bloqueia apenas se já existirem 3 ou mais camadas
            return current_hazards.get('SPIKES', 0) >= 3
            
        if target_hazard == 'TOXIC_SPIKES':
            # Bloqueia apenas se já existirem 2 ou mais camadas
            return current_hazards.get('TOXIC_SPIKES', 0) >= 2

        return False

    # =========================================================================
    # INTELIGÊNCIA DE DECISÃO & MATRIZES
    # =========================================================================

    def get_instinct_intent(self, battle):
        """
        O Instinto avalia a matriz 4D (Minha Role -> Role Inimiga -> Matchup) e cruza 
        com a realidade do campo para ditar a jogada perfeita no modo Autônomo.
        """
        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            if battle.available_switches: return "SWITCH_DEFENSIVE"
            return "ATTACK_STRONG"

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        
        # 1. Leitura Completa do Estado (Os Olhos do Instinto)
        my_role = self.get_role(active)
        opp_role = self.get_role(opp)
        matchup = self.get_matchup_state(active, opp)
        
        my_spe = self.estimate_stat(active, 'spe')
        opp_spe = self.estimate_stat(opp, 'spe')
        is_faster = my_spe > opp_spe
        
        my_hp_frac = active.current_hp_fraction
        opp_hp_frac = opp.current_hp_fraction
        
        my_hp_crit = my_hp_frac <= 0.35
        opp_hp_crit = opp_hp_frac <= 0.35
        my_hp_full = my_hp_frac >= 0.85

        # 2. Resgata as prioridades da Matriz 4D Teórica
        my_role_matrix = self.tactical_matrix.get(my_role, self.tactical_matrix[Role.SWEEPER])
        opp_role_matrix = my_role_matrix.get(opp_role, my_role_matrix[Role.SWEEPER])
        priorities = opp_role_matrix.get(matchup, ["ATTACK_STRONG"]).copy()

        # =====================================================================
        # 3. FILTROS DE REALIDADE E BOM SENSO (Sobrescrevem a Matriz Teórica)
        # =====================================================================
        
        # NOVO: Dinâmica Sweeper vs Sweeper (A Regra da Velocidade)
        if my_role == Role.SWEEPER and opp_role == Role.SWEEPER:
            if not is_faster and not opp_hp_crit:
                # Se somos mais lentos e o oponente não está morrendo (onde um ataque de prioridade salvaria),
                # bater de frente é suicídio. Sobrescreve a matriz e prioriza a fuga.
                priorities = ["SWITCH_DEFENSIVE", "ATTACK_PIVOT", "PROTECT"] + priorities

        # NOVO: Dinâmica Tank vs Sweeper (A Checagem do Escudo Correto)
        if my_role == Role.TANK and opp_role in [Role.SWEEPER, Role.UTILITY]:
            opp_is_physical = self.estimate_stat(opp, 'atk') > self.estimate_stat(opp, 'spa')
            my_def = active.base_stats.get('def', 0)
            my_spd = active.base_stats.get('spd', 0)
            
            # Verifica se o nosso Tank aguenta o tipo de dano do oponente
            is_right_def = (opp_is_physical and my_def >= my_spd) or (not opp_is_physical and my_spd > my_def)
            
            if not is_right_def:
                # Exemplo: Blissey (Defesa Especial) na frente de um Garchomp (Ataque Físico).
                # Não importa a vantagem de tipo, o escudo vai quebrar. Fuja imediatamente.
                priorities = ["SWITCH_DEFENSIVE", "ATTACK_PIVOT", "PROTECT"] + priorities

        # Filtro de Execução (Revenge Kill universal)
        if opp_hp_crit and is_faster:
            priorities.insert(0, "ATTACK_STRONG")
            
        # Filtro de Pânico Extremo (Morte iminente sem chance de bater)
        elif my_hp_crit and not is_faster:
            priorities.insert(0, "ATTACK_PIVOT")
            priorities.insert(0, "SWITCH_DEFENSIVE")

        # =====================================================================
        # 4. AVALIAÇÃO DO QUE É POSSÍVEL FAZER (Available Actions)
        # =====================================================================
        available = self.get_available_actions(battle)
        
        for intent in priorities:
            if intent in available:
                
                if intent == "HEAL" and my_hp_full: continue
                if intent == "BUFF" and my_hp_crit: continue
                
                if intent == "HAZARD":
                    hazards = [m for m in battle.available_moves if self.classify_move(m) == MoveCategory.HAZARD]
                    if not hazards or self.is_hazard_already_set(hazards[0], battle): continue
                        
                if intent == "STATUS" and opp.status is not None: continue

                return intent

        # 5. Fallback Final
        atk_options = [a for a in available if "ATTACK" in a]
        return atk_options[0] if atk_options else (available[0] if available else "ATTACK_STRONG")

    # =========================================================================
    # EXECUTORES DE OBJETOS (Retornam TUPLA: (Objeto, Flag_Mecanica))
    # =========================================================================

    def get_best_lead(self, battle):
        try:
            opp_team = list(battle.opponent_team.values())
            my_team = list(battle.team.values())
            
            if opp_team: avg_base_speed = sum(m.base_stats.get('spe', 50) for m in opp_team) / len(opp_team)
            else: avg_base_speed = 100
            
            is_slow_archetype = avg_base_speed < 80 
            weather_setters = ['drought', 'drizzle', 'sandstream', 'snowwarning']
            opp_has_weather = any(m.ability in weather_setters for m in opp_team)
            my_weather_setter = next((m for m in my_team if m.ability in weather_setters), None)

            predicted_lead = None
            if opp_has_weather: predicted_lead = next((m for m in opp_team if m.ability in weather_setters), opp_team[0])
            elif is_slow_archetype: predicted_lead = min(opp_team, key=lambda m: m.base_stats.get('spe', 50))
            else:
                 if opp_team: predicted_lead = max(opp_team, key=lambda m: m.base_stats.get('spe', 50))

            if not predicted_lead and opp_team: predicted_lead = opp_team[0]
            
            best_lead = None
            if my_weather_setter: best_lead = my_weather_setter
            elif predicted_lead:
                def lead_score(m):
                    advantage = 10 if self.get_matchup_state(m, predicted_lead) in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV] else 0
                    if is_slow_archetype: return advantage
                    return advantage + m.base_stats.get('spe', 50)
                best_lead = max(battle.team.values(), key=lead_score)
            else:
                best_lead = list(battle.team.values())[0]

            try: lead_index = my_team.index(best_lead) + 1
            except ValueError: lead_index = 1
                
            rest_indices = [str(i + 1) for i in range(len(my_team)) if i + 1 != lead_index]
            team_order = str(lead_index) + "".join(rest_indices)
            return f"/team {team_order}"
        except Exception:
            return "/team 123456"

    def get_best_switch(self, battle, intent="DEFENSIVE", history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not candidates:
            return None
        
        active_weather = battle.weather
        my_active = battle.active_pokemon
        matchup = self.get_matchup_state(my_active, opponent) if my_active and opponent else MatchupState.NEUTRAL
        is_bad_matchup = matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]

        # Preserva os objetos originais para os cálculos obrigatórios do poke_env
        opp_types_obj = [t for t in opponent.types if t] if opponent else []
        opp_is_physical = self.estimate_stat(opponent, 'atk') > self.estimate_stat(opponent, 'spa') if opponent else True
        
        def get_hazard_damage(candidate):
            dmg = 0.0
            cond_keys = [str(k).upper() for k in battle.side_conditions.keys()]
            cand_types_str = [str(t).split('.')[-1].upper() for t in candidate.types if t]
            
            if 'STEALTH_ROCK' in cond_keys: 
                valid_types = [t for t in candidate.types if t is not None]
                if valid_types:
                    PokemonTypeEnum = type(valid_types[0])
                    rock_enum = getattr(PokemonTypeEnum, 'ROCK', None)
                    if rock_enum:
                        dmg += 0.125 * candidate.damage_multiplier(rock_enum)
                
            if 'SPIKES' in cond_keys and 'FLYING' not in cand_types_str and str(candidate.ability).lower() != 'levitate':
                layers = int(battle.side_conditions.get('spikes', 1))
                dmg += 0.041 * layers 
            return dmg

        def get_score(candidate):
            score = 0.0
            role = self.get_role(candidate)
            cand_abi = str(candidate.ability).lower() if candidate.ability else ""
            
            hazard_dmg = get_hazard_damage(candidate)
            if candidate.current_hp_fraction <= hazard_dmg:
                return -9999 
                
            if candidate.current_hp_fraction < 0.3: score -= 150
            elif candidate.current_hp_fraction < 0.6: score -= 50

            if intent == "DEFENSIVE":
                if role == Role.TANK: score += 100
                score += (candidate.current_hp_fraction * 200) 

                if opponent:
                    if is_bad_matchup and my_active:
                        # Extrai a ameaça usando os objetos PokemonType matematicamente válidos
                        threat_types_obj = [t for t in opp_types_obj if my_active.damage_multiplier(t) > 1.0]
                        # Converte a lista final validada para strings utilizadas no dicionário
                        threat_types_str = [str(t).split('.')[-1].upper() for t in threat_types_obj]
                        
                        type_absorb_map = {
                            'WATER': ['waterabsorb', 'dryskin', 'stormdrain'],
                            'GROUND': ['levitate'],
                            'GRASS': ['sapsipper'],
                            'FIRE': ['flashfire'],
                            'ELECTRIC': ['voltabsorb', 'lightningrod', 'motordrive']
                        }
                        for t_str in threat_types_str:
                            if t_str in type_absorb_map and cand_abi in type_absorb_map[t_str]:
                                score += 500 
                                
                    cand_matchup = self.get_matchup_state(candidate, opponent)
                    if cand_matchup == MatchupState.DOMINANT: score += 300
                    elif cand_matchup == MatchupState.DEFENSIVE_ADV: score += 200
                    elif cand_matchup == MatchupState.STALEMATE: score += 100
                    elif cand_matchup == MatchupState.NEUTRAL: score += 50
                    elif cand_matchup == MatchupState.DEFENSIVE_DIS: score -= 150
                    elif cand_matchup == MatchupState.CRITICAL_DIS: score -= 300
                        
                    cand_def = candidate.base_stats.get('def', 0)
                    cand_spd = candidate.base_stats.get('spd', 0)
                    if opp_is_physical and cand_def > cand_spd: score += 100
                    elif not opp_is_physical and cand_spd > cand_def: score += 100

            elif intent == "OFFENSIVE":
                if role == Role.SWEEPER: score += 100
                
                weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration', 'raindish', 'icebody']
                weather_setters = {
                    'drizzle': ['swiftswim', 'hydration', 'raindish'],
                    'drought': ['chlorophyll', 'solarpower'],
                    'sandstream': ['sandrush', 'sandforce'],
                    'snowwarning': ['slushrush', 'icebody']
                }

                if active_weather:
                    weather_start = history.get('weather_start_turn', battle.turn) if history else battle.turn
                    turns_active = battle.turn - weather_start
                    
                    weather_is_ending = False
                    if turns_active == 4 or turns_active >= 7:
                        weather_is_ending = True

                    if cand_abi in weather_abusers:
                        if not weather_is_ending: score += 200 
                        else: score += 50 
                else:
                    if cand_abi in weather_setters:
                        team_has_abuser = any(not m.fainted and str(m.ability).lower() in weather_setters[cand_abi] for m in battle.team.values() if m != candidate)
                        if team_has_abuser: score += 150 
                    
                if opponent:
                    if self.estimate_stat(candidate, 'spe') > self.estimate_stat(opponent, 'spe'): score += 150
                    has_se_move = False
                    for m in candidate.moves.values():
                        if m.base_power > 0 and opponent.damage_multiplier(m) > 1.5:
                            has_se_move = True
                            break
                    if has_se_move: score += 200
                    
                    cand_atk = candidate.base_stats.get('atk', 0)
                    cand_spa = candidate.base_stats.get('spa', 0)
                    if max(cand_atk, cand_spa) >= 100: score += 100
                        
                    opp_def = self.estimate_stat(opponent, 'def')
                    opp_spd = self.estimate_stat(opponent, 'spd')
                    if cand_atk > cand_spa and opp_def < opp_spd: score += 80 
                    elif cand_spa > cand_atk and opp_spd < opp_def: score += 80 

            return score

        return max(candidates, key=get_score)

    def get_best_execution_object(self, base_action, battle, history=None):
        # Trava de segurança: se a ação chegar como lista, extrai a string principal
        if isinstance(base_action, list):
            base_action = base_action[0]

        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon
        mechanic_flag = "TERASTALLIZE" if battle.can_tera else "DYNAMAX" if battle.can_dynamax else None
        
        # 1. VETO ANTI-SUICÍDIO (Para setups e hazards)
        if active and opponent:
            is_threat = self.is_threatening(active, opponent)
            if is_threat and active.current_hp_fraction < 0.45:
                if base_action in ["BUFF", "HAZARD", "STATUS", "DEBUFF", "FIELD_CONTROL"]:
                    base_action = "ATTACK_STRONG" 

        # 2. LÓGICA DE UTILIDADE GERAL (Mapeamento Inteligente)
        category_map = {
            "STATUS": MoveCategory.STATUS_CTRL,
            "HAZARD": MoveCategory.HAZARD,
            "HEAL": MoveCategory.RECOVERY,
            "CLEAN_HAZARD": MoveCategory.CLEAN_HAZARD,
            "PROTECT": MoveCategory.PROTECT,
            "DEBUFF": MoveCategory.DEBUFF,
            "BUFF": MoveCategory.SETUP_BUFF,
            "STAT_CLEAN": MoveCategory.STAT_CLEAN,
            "HEAL_STATUS": MoveCategory.HEAL_STATUS,
            "PHAZE": MoveCategory.PHAZE,
            "FIELD_CONTROL": MoveCategory.FIELD_CONTROL
        }
        
        if base_action in category_map:
            cat = category_map[base_action]
            move = next((m for m in battle.available_moves if self.classify_move(m) == cat), None)
            if move: return (move, mechanic_flag)
            # Se pediu utilidade e não tem (ou o instinto vetou), Força o Cérebro a atacar!
            base_action = "ATTACK_STRONG"

        if active and opponent:
            # 3. LÓGICA DE ATAQUES RAMIFICADOS
            if base_action == "ATTACK_PIVOT":
                pivot_moves = [m for m in battle.available_moves if m.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']]
                if pivot_moves: return (pivot_moves[0], mechanic_flag)
                base_action = "ATTACK_STRONG"

            if base_action == "ATTACK_TECH":
                tech_moves_list = ['knockoff', 'foulplay', 'thief', 'nuzzle', 'scald', 'discharge', 'lavaplume', 'saltcure', 'superfang', 'naturesmadness', 'ruination', 'seismictoss', 'nightshade', 'icywind', 'electroweb', 'rocktomb', 'bulldoze', 'snarl', 'mysticalfire', 'strugglebug', 'fakeout', 'brickbreak', 'psychicfangs', 'bodypress']
                tech_moves = [m for m in battle.available_moves if m.id in tech_moves_list]
                if tech_moves: return (max(tech_moves, key=lambda m: m.base_power), mechanic_flag)
                base_action = "ATTACK_STRONG"

            if base_action in ["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK", "ATTACK_MEC"]:
                valid_moves = [m for m in battle.available_moves if m.base_power > 0 and not self.is_move_useless(m, opponent, battle)]
                if not valid_moves: valid_moves = [m for m in battle.available_moves if m.base_power > 0]
                
                if valid_moves:
                    my_atk = self.estimate_stat(active, 'atk')
                    my_spa = self.estimate_stat(active, 'spa')
                    my_spe = self.estimate_stat(active, 'spe')
                    opp_spe = self.estimate_stat(opponent, 'spe')
                    
                    is_faster = my_spe > opp_spe
                    matchup = self.get_matchup_state(active, opponent)
                    is_bad_matchup = matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]
                    
                    predicting_switch = False
                    if base_action == "ATTACK_PREDICTIVE":
                        if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV] or (opponent.current_hp_fraction < 0.3 and is_faster):
                            predicting_switch = True

                    best_move = None
                    max_score = -9999

                    if is_faster and is_bad_matchup:
                        pivot_moves = [m for m in valid_moves if m.id in ['uturn', 'voltswitch', 'flipturn']]
                        if pivot_moves: return (pivot_moves[0], mechanic_flag)

                    for m in valid_moves:
                        score = 0.0
                        acc = 100.0 if m.accuracy is True else float(m.accuracy) if isinstance(m.accuracy, (int, float)) else 100.0
                        stab = 1.5 if m.type in active.types else 1.0
                        
                        opp_mult = opponent.damage_multiplier(m)
                        if predicting_switch:
                            opp_mult = 1.0 if opp_mult < 1.0 else opp_mult * 0.8 

                        base_expected = (m.base_power * stab * opp_mult) * (acc / 100.0)
                        score += base_expected
                        
                        if m.category.name == "PHYSICAL" and my_atk > my_spa: score += 50
                        elif m.category.name == "SPECIAL" and my_spa > my_atk: score += 50
                        
                        m_priority = getattr(m, 'priority', 0)
                        estimated_damage_fraction = base_expected / 250.0

                        if m_priority > 0:
                            if estimated_damage_fraction >= opponent.current_hp_fraction:
                                score += 2000 + base_expected 
                            elif not is_faster and estimated_damage_fraction >= opponent.current_hp_fraction * 0.6:
                                score += 500 + base_expected 
                        elif not predicting_switch and estimated_damage_fraction >= opponent.current_hp_fraction:
                            score += 1000 
                            
                        if predicting_switch and m.id in ['knockoff', 'scald', 'nuzzle', 'rapidspin']: score += 80 

                        if score > max_score:
                            max_score = score
                            best_move = m

                    return (best_move, mechanic_flag)

        # 4. LÓGICA DE TROCAS ESPECÍFICAS
        if base_action == "SWITCH_DEFENSIVE":
            if battle.available_switches: return (self.get_best_switch(battle, "DEFENSIVE", history), None)
            
        if base_action == "SWITCH_OFFENSIVE":
            if battle.available_switches: return (self.get_best_switch(battle, "OFFENSIVE", history), None)

        if base_action == "SWITCH":
            if battle.available_switches: return (self.get_best_switch(battle, "DEFENSIVE", history), None)

        # 5. FALLBACK DE SEGURANÇA À PROVA DE BALAS
        # Se deu tudo errado, ATACA! Só troca se não tiver nenhum ataque disponível.
        if battle.available_moves:
            damaging = [m for m in battle.available_moves if m.base_power > 0]
            if damaging: return (max(damaging, key=lambda m: m.base_power), mechanic_flag)
            return (random.choice(battle.available_moves), mechanic_flag)
            
        if battle.available_switches:
            return (self.get_best_switch(battle, "DEFENSIVE", history), None)
        
        return None