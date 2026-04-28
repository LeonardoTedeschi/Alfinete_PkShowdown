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
    DOMINANT = 1       # SE vs NVE  
    VOLATILE = 2       # SE vs SE
    OFFENSIVE_ADV = 3  # SE vs N
    DEFENSIVE_ADV = 4  # N vs NVE
    DEFENSIVE_DIS = 5  # N vs SE
    OFFENSIVE_DIS = 6  # NVE vs N
    STALEMATE = 7      # NVE vs NVE
    NEUTRAL = 8        # N vs N
    CRITICAL_DIS = 9   # NVE vs SE

class TacticalMode(Enum):
    """
    4 Modos Táticos que refletem o instinto do comandante.

    PRESS   = Ataque direto. Matar ou quebrar. Não desperdiçar turno.
    CONTEST = Disputa aberta. Speed é lei. Avaliar risco a cada turno.
    GRIND   = Setup, attrition, stall. Hazard/Buff/Debuff/Heal/Status.
              Quando não tem opção de troca, desgasta pouco a pouco.
    ESCAPE  = Sair imediatamente. Defesa errada, 4x fraco, ou sem pressão.
    """
    PRESS = 1
    CONTEST = 2
    GRIND = 3
    ESCAPE = 4

class MoveCategory(Enum):
    ATTACK_STRONG = 1
    ATTACK_PREDICTIVE = 2
    ATTACK_PIVOT = 3
    ATTACK_TECH = 4
    BUFF = 5
    STATUS = 6
    HEAL = 7
    CLEAN_HAZARD = 8
    PROTECT = 9
    DEBUFF = 10
    STAT_CLEAN = 11
    HEAL_STATUS = 12
    PHAZE = 13
    FIELD_CONTROL = 14
    HAZARD = 15
    SWITCH_DEFENSIVE = 16
    SWITCH_OFFENSIVE = 17
    UNKNOWN = 18

# =============================================================================
# 2. O CÉREBRO ESPECIALISTA (CORE) — ARQUITETURA DE 3 CAMADAS
# =============================================================================

class InstinctCore:
    def __init__(self):
        # =====================================================================
        # CAMADA 2: TEMPLATES DE MODO TÁTICO (Base de Prioridades)
        # Contém todas as 17 ações mapeadas. As funções de Camada 3 farão o shift
        # com base em HP, Speed e ameaças.
        # =====================================================================
        self.mode_templates = {
            TacticalMode.PRESS: [
                # Vantagem Absoluta: Foco em predição, nuke, ou setup para varrer o time.
                "ATTACK_PREDICTIVE", "ATTACK_STRONG", "BUFF", "ATTACK_TECH",
                "HAZARD", "FIELD_CONTROL", "ATTACK_PIVOT", "CLEAN_HAZARD", 
                "STATUS", "DEBUFF", "HEAL", "HEAL_STATUS", "STAT_CLEAN", 
                "PHAZE", "PROTECT", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE"
            ],
            
            TacticalMode.CONTEST: [
                # Disputa Ativa (Neutro/Volátil): Foco em ganhar no 1v1 ou não perder momentum.
                "ATTACK_STRONG", "ATTACK_TECH", "PROTECT", "ATTACK_PIVOT",
                "STATUS", "BUFF", "HEAL", "HAZARD", "CLEAN_HAZARD",
                "DEBUFF", "FIELD_CONTROL", "ATTACK_PREDICTIVE", "STAT_CLEAN", 
                "PHAZE", "HEAL_STATUS", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE"
            ],
            
            TacticalMode.GRIND: [
                # Attrition/Stall: Foco em desgaste, recuperação e bloqueio do adversário.
                "HAZARD", "STATUS", "HEAL", "PROTECT", "DEBUFF", 
                "CLEAN_HAZARD", "PHAZE", "STAT_CLEAN", "HEAL_STATUS", 
                "BUFF", "FIELD_CONTROL", "ATTACK_TECH", "ATTACK_PIVOT", 
                "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE", "ATTACK_STRONG", "ATTACK_PREDICTIVE"
            ],
            
            TacticalMode.ESCAPE: [
                # Desvantagem Crítica: Foco absoluto em sobrevivência ou suicídio útil.
                "SWITCH_DEFENSIVE", "ATTACK_PIVOT", "PROTECT", "SWITCH_OFFENSIVE",
                "ATTACK_TECH", "STATUS", "DEBUFF", "STAT_CLEAN", "PHAZE", 
                "HEAL", "CLEAN_HAZARD", "HEAL_STATUS", "FIELD_CONTROL", 
                "HAZARD", "BUFF", "ATTACK_STRONG", "ATTACK_PREDICTIVE"
            ]
        }

        # =====================================================================
        # CAMADA 3: MODIFICADORES POR ROLE
        # Cada função recebe o template base e o contexto completo.
        # Retorna a lista REORDENADA para refletir o instinto do comandante.
        # =====================================================================
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

    # =========================================================================
    # EXTRAÇÃO DE ESTADO PARA O Q-LEARNING E AUXILIARES
    # =========================================================================

    def get_hp_bucket(self, pokemon):
        if not pokemon or pokemon.fainted: return "CRIT" 
        hp = pokemon.current_hp_fraction
        if hp >= 0.7: return "FULL"
        if hp >= 0.35: return "MED"
        return "CRIT"

    def get_opp_hp_bucket(self, pokemon):
        if not pokemon or pokemon.fainted: return "CRIT" 
        hp = pokemon.current_hp_fraction
        if hp >= 0.6: return "HIGH"
        if hp >= 0.3: return "MID"
        return "CRIT"

    def get_weather_state(self, battle):
        if battle.fields:
            field_keys = [str(f).upper() for f in battle.fields.keys()]
            if any("TRICK_ROOM" in f for f in field_keys): 
                return "TRICK_ROOM"
                
        if battle.weather:
            w_name = next(iter(battle.weather)).name
            if w_name in ["HAIL", "SNOW", "SANDSTORM"]: return "DAMAGE"
            if w_name in ["RAINDANCE", "PRIMORDIALSEA", "SUNNYDAY", "DESOLATELAND"]: return "BUFF"
        return "NORMAL"

    def get_speed_tier(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not active or not opponent: return "SLOWER"
        
        my_speed = self.estimate_stat(active, 'spe')
        opp_speed = self.estimate_stat(opponent, 'spe')
        
        if my_speed > opp_speed: return "FASTER"
        return "SLOWER"

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
        if battle.can_tera or battle.can_mega_evolve or battle.can_z_move or battle.can_dynamax:
            return "MEC_AVAIL"
        return "MEC_USED"

    def _get_hazard_damage(self, candidate, battle):
        dmg = 0.0
        cond_keys = [str(k).upper() for k in battle.side_conditions.keys()]
        cand_types_str = [str(t).split('.')[-1].upper() for t in candidate.types if t]
        
        if 'STEALTH_ROCK' in cond_keys: 
            valid_types = [t for t in candidate.types if t is not None]
            if valid_types:
                PokemonTypeEnum = type(valid_types[0])
                rock_enum = getattr(PokemonTypeEnum, 'ROCK', None)
                if rock_enum: dmg += 0.125 * candidate.damage_multiplier(rock_enum)
            
        if 'SPIKES' in cond_keys and 'FLYING' not in cand_types_str and str(candidate.ability).lower() != 'levitate':
            layers = int(battle.side_conditions.get('spikes', 1))
            dmg += 0.041 * layers 
        return dmg

    def get_available_actions(self, battle):
        available = set()
        
        if battle.available_switches:
            available.add(MoveCategory.SWITCH_DEFENSIVE.name)
            available.add(MoveCategory.SWITCH_OFFENSIVE.name)

        if battle.available_moves:
            damaging_types = set()  
            for move in battle.available_moves:
                if self.is_move_useless(move, battle.opponent_active_pokemon, battle):
                    continue
                
                cat = self.classify_move(move)
                if cat != MoveCategory.UNKNOWN:
                    if cat == MoveCategory.HAZARD and self.is_hazard_already_set(move, battle):
                        continue
                    available.add(cat.name)
                    
                    if cat in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT]:
                        if move.type:
                            damaging_types.add(move.type)
            
            if MoveCategory.ATTACK_STRONG.name in available and len(damaging_types) > 1:
                benched_opponents = [m for m in battle.opponent_team.values() if not m.fainted and not m.active]
                if benched_opponents:
                    available.add(MoveCategory.ATTACK_PREDICTIVE.name)

        available_list = list(available)
        
        if not available_list:
            if battle.available_switches:
                return [MoveCategory.SWITCH_DEFENSIVE.name, MoveCategory.SWITCH_OFFENSIVE.name]
            else:
                if battle.available_moves:
                    cats = [self.classify_move(m).name for m in battle.available_moves]
                    if MoveCategory.PROTECT.name in cats: return [MoveCategory.PROTECT.name]
                    return list(set(c for c in cats if c != "UNKNOWN")) or ["ATTACK_STRONG"]
                return ["ATTACK_STRONG"]

        return available_list

    def get_state(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active or not opponent:
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
        else:
            base = pokemon.base_stats.get(stat_name, 50)
            role = self.get_role(pokemon)
            
            if stat_name == 'hp':
                val = int(base * 2 + 204)
            else:
                calc_boosted = int((base * 2 + 99) * 1.1)
                calc_invested = int(base * 2 + 99)
                calc_uninvested = int(base * 2 + 36)
                
                if role == Role.SWEEPER:
                    if stat_name == 'spe': val = calc_boosted
                    elif stat_name == 'atk' and pokemon.base_stats.get('atk', 0) >= pokemon.base_stats.get('spa', 0): val = calc_invested
                    elif stat_name == 'spa' and pokemon.base_stats.get('spa', 0) > pokemon.base_stats.get('atk', 0): val = calc_invested
                    else: val = calc_uninvested
                elif role == Role.TANK:
                    base_def = pokemon.base_stats.get('def', 0)
                    base_spd = pokemon.base_stats.get('spd', 0)
                    best_def = 'def' if base_def >= base_spd else 'spd'
                    if stat_name == best_def: val = calc_boosted
                    elif stat_name in ['def', 'spd']: val = calc_invested
                    else: val = calc_uninvested
                else:
                    if stat_name == 'spe': val = calc_boosted
                    elif stat_name in ['def', 'spd']: val = calc_invested
                    else: val = calc_uninvested

        if stat_name == 'spe':
            val *= self._get_speed_mod(pokemon)
        else:
            modifier = pokemon.boosts.get(stat_name, 0)
            if modifier > 0: val *= (1 + 0.5 * modifier)
            elif modifier < 0: val *= (2 / (2 + abs(modifier)))

        item_str = str(pokemon.item).lower() if pokemon.item else ""
        item_mod = 1.0
        
        if stat_name == 'spe' and item_str == 'choicescarf': item_mod = 1.5
        elif stat_name == 'atk' and item_str == 'choiceband': item_mod = 1.5
        elif stat_name == 'spa' and item_str == 'choicespecs': item_mod = 1.5
        elif stat_name == 'spd' and item_str in ['assaultvest', 'eviolite']: item_mod = 1.5
            
        return int(val * item_mod)

    def estimate_damage_percent(self, move, attacker, defender):
        if move.category.name == "STATUS" or move.base_power == 0: return 0.0
            
        bp = float(move.base_power)
        
        if move.category.name == "PHYSICAL":
            atk = self.estimate_stat(attacker, 'atk')
            if move.id == 'bodypress': atk = self.estimate_stat(attacker, 'def')
            defense = self.estimate_stat(defender, 'def')
        else:
            atk = self.estimate_stat(attacker, 'spa')
            defense = self.estimate_stat(defender, 'spd')
            if move.id in ['psyshock', 'psystrike', 'secretsword']: defense = self.estimate_stat(defender, 'def')
                
        if defense <= 0: defense = 1
        
        base_dmg = (42.0 * bp * (atk / defense)) / 50.0 + 2.0
        stab = 1.5 if move.type in attacker.types else 1.0
        type_mod = defender.damage_multiplier(move)
        
        final_dmg = base_dmg * stab * type_mod
        max_hp = max(1, self.estimate_stat(defender, 'hp'))
        
        return final_dmg / max_hp

    def classify_move(self, move):
        move_id = move.id
        
        if move_id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']: return MoveCategory.ATTACK_PIVOT
        
        tech_moves = [
            'knockoff', 'foulplay', 'thief', 'nuzzle', 'scald', 'discharge', 'lavaplume', 'saltcure', 
            'superfang', 'naturesmadness', 'ruination', 'seismictoss', 'nightshade', 'icywind', 'electroweb', 
            'rocktomb', 'bulldoze', 'snarl', 'mysticalfire', 'strugglebug', 'fakeout', 'brickbreak', 
            'psychicfangs', 'bodypress'
        ]
        if move_id in tech_moves: return MoveCategory.ATTACK_TECH

        if move_id in ['haze', 'clearsmog']: return MoveCategory.STAT_CLEAN
        if move_id in ['aromatherapy', 'healbell', 'junglehealing']: return MoveCategory.HEAL_STATUS
        if move_id in ['roar', 'whirlwind', 'dragontail', 'circlethrow']: return MoveCategory.PHAZE
        if move_id in ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape', 'trickroom', 'tailwind', 'electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain']: return MoveCategory.FIELD_CONTROL
        if move_id in ['defog', 'rapidspin', 'mortalspin', 'courtchange']: return MoveCategory.CLEAN_HAZARD
        if move_id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']: return MoveCategory.HAZARD
        if move_id in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark', 'silktrap', 'safeguard']: return MoveCategory.PROTECT
        if move_id in ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun', 'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap']: return MoveCategory.HEAL

        if move.category.name == "STATUS":
            if getattr(move, 'heal', 0): return MoveCategory.HEAL
            if getattr(move, 'status', None): return MoveCategory.STATUS
            if getattr(move, 'boosts', None):
                if any(v > 0 for v in move.boosts.values()): return MoveCategory.BUFF
                if any(v < 0 for v in move.boosts.values()): return MoveCategory.DEBUFF
            return MoveCategory.STATUS 

        if move.category.name in ["PHYSICAL", "SPECIAL"] and move.base_power > 0: 
            return MoveCategory.ATTACK_STRONG

        return MoveCategory.UNKNOWN

    def is_move_useless(self, move, opp_pokemon, battle):
        my_pokemon = battle.active_pokemon
        if my_pokemon and move.id == 'substitute':
            my_effects = [str(e).lower() for e in my_pokemon.effects]
            if any('substitute' in e for e in my_effects): return True

        # === 1. BLOQUEIO DE CURA SE A VIDA ESTÁ CHEIA ===
        if move.id in ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun', 'synthesis', 'softboiled', 'milkdrink', 'shoreup']:
            if my_pokemon and my_pokemon.current_hp_fraction == 1.0:
                return True

        # === 2. BLOQUEIO DE LIMPEZA DE HAZARD SEM HAZARDS ===
        if move.id in ['defog', 'courtchange']:
            if not battle.side_conditions and not battle.opponent_side_conditions:
                return True

        if not opp_pokemon: return False
        
        opp_types = [str(t).split('.')[-1].upper() for t in opp_pokemon.types if t]
        move_id = move.id
        move_type = str(move.type).split('.')[-1].upper() if move.type else "UNKNOWN"
        
        opp_abilities = []
        if opp_pokemon.ability: opp_abilities = [str(opp_pokemon.ability).lower()]
        elif opp_pokemon.possible_abilities: opp_abilities = [str(a).lower() for a in opp_pokemon.possible_abilities]

        opp_effects = [str(effect).lower() for effect in opp_pokemon.effects]
        if any('substitute' in e for e in opp_effects):
            if move.category.name == 'STATUS' and move.target in ['normal', 'any', 'allAdjacentFoes', 'foeSide']: return True

        if 'wonderguard' in opp_abilities:
            if move.category.name in ['PHYSICAL', 'SPECIAL'] and move.base_power > 0:
                if opp_pokemon.damage_multiplier(move) < 2.0: return True

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
            if move.status and opp_pokemon.status: return True
            
            # === 3. PROTEÇÃO DE TERRENOS (Misty e Electric Terrain) ===
            fields = [str(f).lower() for f in battle.fields.keys()]
            opp_is_grounded = 'FLYING' not in opp_types and 'levitate' not in opp_abilities and 'magnetrise' not in opp_effects
            
            if opp_is_grounded:
                if 'mistyterrain' in fields and move.status in ['psn', 'tox', 'brn', 'par', 'slp']:
                    return True
                if 'electricterrain' in fields and move.status == 'slp':
                    return True

            if move_id in ['toxic', 'poisonpowder', 'poisongas'] and ('STEEL' in opp_types or 'POISON' in opp_types): return True
            if move_id == 'thunderwave' and ('GROUND' in opp_types or 'ELECTRIC' in opp_types): return True
            if move_id == 'willowisp' and 'FIRE' in opp_types: return True
            if move_id in ['leechseed', 'spore', 'sleeppowder', 'stunspore'] and 'GRASS' in opp_types: return True
            if any(ab in opp_abilities for ab in ['magicbounce']) and move.target in ['normal', 'allAdjacentFoes', 'foeSide']: return True
            if any(ab in opp_abilities for ab in ['immunity']) and move_id in ['toxic', 'poisongas']: return True
            if any(ab in opp_abilities for ab in ['limber']) and move_id == 'thunderwave': return True
            if any(ab in opp_abilities for ab in ['goodasgold']): return True 

        return False

    def get_matchup_state(self, my_mon, opp_mon) -> MatchupState:
        if not my_mon or not opp_mon: return MatchupState.NEUTRAL
        
        my_moves = [m for m in my_mon.moves.values() if m.base_power > 0]
        
        if my_moves:
            my_best_mult = max([opp_mon.damage_multiplier(move) for move in my_moves])
        else:
            my_best_mult = 0.0

        opp_best_mult = 0.0
        # 1. Checa contra os Tipos Base (STAB)
        for type_ in opp_mon.types:
             if type_:
                 multiplier = my_mon.damage_multiplier(type_)
                 if multiplier > opp_best_mult: 
                     opp_best_mult = multiplier
                     
        # 2. CORREÇÃO: Checa contra Golpes de Coverage Revelados
        known_opp_moves = [m for m in opp_mon.moves.values() if m.base_power > 0]
        for move in known_opp_moves:
             multiplier = my_mon.damage_multiplier(move)
             if multiplier > opp_best_mult:
                 opp_best_mult = multiplier

        my_se = my_best_mult > 1.0
        my_neutral = my_best_mult == 1.0
        my_nve = my_best_mult < 1.0
        
        opp_se = opp_best_mult > 1.0
        opp_neutral = opp_best_mult == 1.0
        opp_nve = opp_best_mult < 1.0

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
        hazard_types = ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']
        move_to_hazard = {
            'stealthrock': 'STEALTH_ROCK',
            'stickyweb': 'STICKY_WEB',
            'spikes': 'SPIKES',
            'toxicspikes': 'TOXIC_SPIKES'
        }
        
        target_hazard = move_to_hazard.get(move.id)
        if not target_hazard: return False 
            
        current_hazards = {}
        for condition, layers in battle.opponent_side_conditions.items():
            cond_str = str(condition).upper()
            for haz in hazard_types:
                if haz in cond_str:
                    if haz == 'SPIKES' and 'TOXIC' in cond_str: continue 
                    current_hazards[haz] = int(layers) if isinstance(layers, int) else 1
                    break
                    
        if len(current_hazards) >= 2 and target_hazard not in current_hazards: return True 
        if target_hazard == 'STEALTH_ROCK': return 'STEALTH_ROCK' in current_hazards
        if target_hazard == 'STICKY_WEB': return 'STICKY_WEB' in current_hazards
        if target_hazard == 'SPIKES': return current_hazards.get('SPIKES', 0) >= 3
        if target_hazard == 'TOXIC_SPIKES': return current_hazards.get('TOXIC_SPIKES', 0) >= 2

        return False

    # =========================================================================
    # HELPERS AUXILIARES DE ROLE
    # =========================================================================

    def _is_physical(self, pokemon):
        if not pokemon: return True
        return self.estimate_stat(pokemon, 'atk') > self.estimate_stat(pokemon, 'spa')

    def _has_recovery(self, pokemon):
        if not pokemon: return False
        recovery_moves = ['recover', 'roost', 'moonlight', 'slackoff',
                         'morningsun', 'synthesis', 'softboiled',
                         'milkdrink', 'shoreup', 'strengthsap']
        return any(m.id in recovery_moves for m in pokemon.moves.values())

    def _has_move(self, pokemon, move_ids):
        if not pokemon: return False
        return any(m.id in move_ids for m in pokemon.moves.values())

    def _opponent_can_setup(self, opponent):
        if not opponent: return False
        setup_moves = ['swordsdance', 'dragondance', 'nastyplot',
                      'quiverdance', 'shellsmash', 'shiftgear',
                      'calmmind', 'bulkup', 'workup', 'coil']
        return any(m.id in setup_moves for m in opponent.moves.values())

    # =========================================================================
    # CAMADA 1: MATCHUP → MODO TÁTICO
    # =========================================================================

    def _get_tactical_mode(self, matchup: MatchupState, 
                           my_role: Role, opp_role: Role,
                           is_faster: bool, my_hp_frac: float, 
                           opp_hp_frac: float, is_threat: bool,
                           active, opponent) -> TacticalMode:
        
        if my_role == Role.TANK and opp_role == Role.SWEEPER:
            opp_is_physical = self._is_physical(opponent)
            my_def = active.base_stats.get('def', 0)
            my_spd = active.base_stats.get('spd', 0)
            is_right_def = (opp_is_physical and my_def >= my_spd) or \
                           (not opp_is_physical and my_spd > my_def)
            if not is_right_def:
                return TacticalMode.ESCAPE

        if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
            return TacticalMode.PRESS
        if matchup in [MatchupState.VOLATILE, MatchupState.NEUTRAL]:
            return TacticalMode.CONTEST
        if matchup in [MatchupState.STALEMATE, MatchupState.DEFENSIVE_ADV]:
            return TacticalMode.GRIND
        return TacticalMode.ESCAPE

    # =========================================================================
    # CAMADA 3: MODIFICADORES DE ROLE (O INSTINTO DO COMANDANTE)
    # =========================================================================

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
        opp_is_physical = self._is_physical(opponent)
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

        opponent_has_recovery = self._has_recovery(opponent)
        if opponent_has_recovery and "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        return modified

    def _mod_tank_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        has_delay = self._has_move(active, ['futuresight', 'doomdesire'])
        has_pivot = self._has_move(active, ['uturn', 'voltswitch', 'flipturn', 'teleport'])
        has_protect = self._has_move(active, ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker'])

        if has_delay and has_pivot:
            if "ATTACK_TECH" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
            if "ATTACK_PIVOT" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
            return modified

        if "HAZARD" in modified: modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "STATUS" in modified: modified.insert(0, modified.pop(modified.index("STATUS")))
        if "HEAL" in modified: modified.insert(0, modified.pop(modified.index("HEAL")))
        if has_protect and "PROTECT" in modified: modified.insert(0, modified.pop(modified.index("PROTECT")))
        if "SWITCH_OFFENSIVE" in modified: modified.insert(0, modified.pop(modified.index("SWITCH_OFFENSIVE")))
        if "ATTACK_TECH" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        return modified

    def _mod_tank_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "HAZARD" in modified: modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "STATUS" in modified: modified.insert(0, modified.pop(modified.index("STATUS")))
        has_pivot = self._has_move(opponent, ['uturn', 'voltswitch', 'flipturn', 'teleport'])
        if has_pivot and "ATTACK_TECH" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        return modified

    def _mod_utility_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if is_threat and not is_faster:
            if "SWITCH_DEFENSIVE" in modified: modified.insert(0, "SWITCH_DEFENSIVE")
            if "ATTACK_TECH" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))

        opp_can_setup = self._opponent_can_setup(opponent)
        if opp_can_setup and "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        elif "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))

        if is_faster and opp_hp_frac <= 0.4 and "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        return modified

    def _mod_utility_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "STATUS" in modified: modified.insert(0, modified.pop(modified.index("STATUS")))
        if "ATTACK_PIVOT" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        if "DEBUFF" in modified: modified.insert(0, modified.pop(modified.index("DEBUFF")))
        if "HAZARD" in modified: modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "ATTACK_STRONG" in modified and "SWITCH_DEFENSIVE" in modified:
            if "ATTACK_STRONG" in modified: modified.remove("ATTACK_STRONG")
        return modified

    def _mod_utility_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "HAZARD" in modified: modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "STATUS" in modified: modified.insert(0, modified.pop(modified.index("STATUS")))
        if "FIELD_CONTROL" in modified: modified.insert(0, modified.pop(modified.index("FIELD_CONTROL")))
        if "ATTACK_PIVOT" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        if "DEBUFF" in modified: modified.insert(0, modified.pop(modified.index("DEBUFF")))
        return modified

    # =========================================================================
    # INTEGRAÇÃO DA ÁRVORE TÁTICA E ACTION MASKING
    # =========================================================================

    def get_instinct_profile(self, battle):
        candidate_mask = self.get_available_actions(battle)

        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            primary = "SWITCH_DEFENSIVE" if battle.available_switches else "ATTACK_STRONG"
            return (primary, 1.0, [primary], candidate_mask)

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_role = self.get_role(active)
        opp_role = self.get_role(opp)
        matchup = self.get_matchup_state(active, opp)

        my_spe = self.estimate_stat(active, 'spe')
        opp_spe = self.estimate_stat(opp, 'spe')
        is_faster = my_spe > opp_spe

        my_hp_frac = active.current_hp_fraction
        opp_hp_frac = opp.current_hp_fraction
        is_threat = self.is_threatening(active, opp)

        # CAMADA 1: Matchup → TacticalMode
        mode = self._get_tactical_mode(matchup, my_role, opp_role, 
                                        is_faster, my_hp_frac, opp_hp_frac, is_threat, 
                                        active, opp)

        # CAMADA 2: Template base do modo
        base_priorities = self.mode_templates[mode].copy()

        # CAMADA 3: Modificador de Role ajusta a lista
        modifier_fn = self.role_modifiers.get((my_role, opp_role))
        if modifier_fn:
            priorities = modifier_fn(
                base_priorities, active, opp, is_faster,
                my_hp_frac, opp_hp_frac, is_threat
            )
        else:
            priorities = base_priorities

        # --- FILTRAGEM POR CANDIDATE_MASK ---
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

        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        if opp_alive <= 2 and "HAZARD" in ranking_list:
            ranking_list.remove("HAZARD")
            ranking_list.append("HAZARD")

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
        return (primary, confidence, ranking_list, candidate_mask)


    # =========================================================================
    # EXECUTORES DE OBJETOS (Switches e Golpes)
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

    def get_defensive_switch(self, battle, history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not candidates: return None
        
        my_active = battle.active_pokemon
        matchup = self.get_matchup_state(my_active, opponent) if my_active and opponent else MatchupState.NEUTRAL
        is_bad_matchup = matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]

        opp_types_obj = [t for t in opponent.types if t] if opponent else []
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0] if opponent else []
        opp_is_physical = self._is_physical(opponent) if opponent else True
        
        def get_score(candidate):
            hazard_dmg = self._get_hazard_damage(candidate, battle)
            hp_frac = candidate.current_hp_fraction
            if hp_frac <= hazard_dmg + 0.05: return -9999 

            score = 0.0
            if hp_frac >= 0.7: score += 200
            elif hp_frac >= 0.35: score += 100
            else: score += 50
                
            role = self.get_role(candidate)
            if role == Role.TANK: score += 100

            if opponent:
                cand_spe = self.estimate_stat(candidate, 'spe')
                opp_spe = self.estimate_stat(opponent, 'spe')
                has_weakness = False
                
                # --- RESISTÊNCIAS E IMUNIDADES CAPADAS EM 300 ---
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
                    'GRASS': ['sapsipper'], 'FIRE': ['flashfire'], 'ELECTRIC': ['voltabsorb', 'lightningrod']
                }
                threat_types_str = [str(t).split('.')[-1].upper() for t in opp_types_obj]
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

            return score

        return max(candidates, key=get_score)

    def get_offensive_switch(self, battle, history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not candidates: return None
        
        active_weather = battle.weather
        opp_types_obj = [t for t in opponent.types if t] if opponent else []
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0] if opponent else []

        def get_score(candidate):
            hazard_dmg = self._get_hazard_damage(candidate, battle)
            hp_frac = candidate.current_hp_fraction
            if hp_frac <= hazard_dmg + 0.05: return -9999 

            score = 0.0
            if hp_frac >= 0.7: score += 200
            elif hp_frac >= 0.35: score += 100
            else: score += 50

            role = self.get_role(candidate)
            if role == Role.SWEEPER: score += 100
            
            cand_abi = str(candidate.ability).lower() if candidate.ability else ""
            weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration']
            
            if active_weather:
                weather_start = history.get('weather_start_turn', battle.turn) if history and 'weather_start_turn' in history else battle.turn
                if cand_abi in weather_abusers and (battle.turn - weather_start) < 4: 
                    score += 200 
                
            if opponent:
                cand_spe = self.estimate_stat(candidate, 'spe')
                opp_spe = self.estimate_stat(opponent, 'spe')
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

                if cand_spe > opp_spe: score += 150
                
                has_se_move = False
                for m in candidate.moves.values():
                    if m.base_power > 0:
                        mult = opponent.damage_multiplier(m)
                        if mult > 1.0:
                            score += 100 * mult  
                            has_se_move = True
                if has_se_move: score += 150

            return score
        
        return max(candidates, key=get_score)

    def get_post_faint_switch(self, battle, history=None):
        opponent = battle.opponent_active_pokemon
        candidates = battle.available_switches
        if not opponent or not candidates:
            return None

        best_offense_score = -9999
        best_defense_score = -9999

        opp_atk = self.estimate_stat(opponent, 'atk')
        opp_spa = self.estimate_stat(opponent, 'spa')
        opp_spe = self.estimate_stat(opponent, 'spe')
        opp_is_physical = opp_atk > opp_spa

        opp_types_obj = [t for t in opponent.types if t]
        threat_types_str = [str(t).split('.')[-1].upper() for t in opp_types_obj]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]

        type_absorb_map = {
            'WATER': ['waterabsorb', 'dryskin', 'stormdrain'],
            'GROUND': ['levitate'],
            'GRASS': ['sapsipper'],
            'FIRE': ['flashfire'],
            'ELECTRIC': ['voltabsorb', 'lightningrod', 'motordrive']
        }

        for cand in candidates:
            off_score = 0
            def_score = 0
            
            cand_spe = self.estimate_stat(cand, 'spe')
            matchup = self.get_matchup_state(cand, opponent)
            cand_abi = str(cand.ability).lower() if cand.ability else ""

            # --- LÓGICA DEFENSIVA HERDADA COM RESISTÊNCIAS CAPADAS ---
            has_weakness = False
            for opp_type in opp_types_obj:
                mult = cand.damage_multiplier(opp_type)
                if mult > 1.0:
                    def_score -= 150 * mult
                    off_score -= 50 * mult
                    has_weakness = True
                elif mult < 1.0:
                    def_score += min(300, 50 / max(mult, 0.1))

            for move in known_opp_moves:
                mult = cand.damage_multiplier(move)
                if mult > 1.0:
                    def_score -= 200 * mult
                    off_score -= 100 * mult
                    has_weakness = True
                elif mult < 1.0:
                    def_score += min(300, 75 / max(mult, 0.1))

            if cand_spe > opp_spe:
                if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
                    off_score += 150
                
                has_se_move = any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in cand.moves.values())
                if has_se_move: off_score += 150
                    
                if has_weakness: off_score -= 50
            else:
                if has_weakness:
                    off_score -= 200
                    def_score -= 200

            if opponent.boosts:
                boost_sum = max(opponent.boosts.get('atk', 0), opponent.boosts.get('spa', 0)) + opponent.boosts.get('spe', 0)
                if boost_sum > 0: def_score += (boost_sum * 100)

            for t_str in threat_types_str:
                if t_str in type_absorb_map and cand_abi in type_absorb_map[t_str]:
                    def_score += 200 

            cand_def = cand.base_stats.get('def', 0)
            cand_spd = cand.base_stats.get('spd', 0)
            if opp_is_physical and cand_def > cand_spd: def_score += 80
            elif not opp_is_physical and cand_spd > cand_def: def_score += 80

            if off_score > best_offense_score: best_offense_score = off_score
            if def_score > best_defense_score: best_defense_score = def_score

        if best_offense_score > best_defense_score:
            return self.get_offensive_switch(battle, history)
        else:
            return self.get_defensive_switch(battle, history)
       
    def _select_best_move_in_category(self, candidates, cat, active, opponent, battle):
        """
        Heurística inteligente para escolher o melhor movimento dentre múltiplas opções 
        da mesma categoria, evitando seleções cegas (next) do motor.
        """
        if not candidates: return None

        if cat == MoveCategory.HAZARD:
            # Prioriza Hazards de maior impacto global
            priority = {'stealthrock': 4, 'stickyweb': 3, 'spikes': 2, 'toxicspikes': 1}
            return max(candidates, key=lambda m: priority.get(m.id, 0))

        if cat == MoveCategory.STATUS:
            # Prioriza precisão e impacto tático imediato
            def status_score(m):
                s = float(m.accuracy) if isinstance(m.accuracy, (int, float)) else 100.0
                if m.id in ['spore', 'sleeppowder', 'yawn']: s += 50
                elif m.id in ['willowisp', 'thunderwave', 'glare']: s += 30
                elif m.id in ['toxic']: s += 20
                return s
            return max(candidates, key=status_score)

        if cat == MoveCategory.ATTACK_TECH:
            # Organizado: Avalia golpes técnicos com base no estado do campo
            def tech_score(m):
                s = float(m.base_power)
                if m.id in ['rapidspin', 'mortalspin'] and self.get_hazard_state(battle.side_conditions) == "SET":
                    s += 500
                elif m.id in ['nuzzle', 'scald', 'discharge', 'lavaplume'] and self.get_status_state(opponent) == "CLEAN":
                    s += 300
                elif m.id == 'knockoff':
                    s += 200
                return s
            return max(candidates, key=tech_score)

        if cat == MoveCategory.ATTACK_PIVOT:
            # U-turn vs Volt Switch: prefere o com STAB ou maior base power
            return max(candidates, key=lambda m: m.base_power * (1.5 if m.type in active.types else 1.0))

        # Fallback genérico: Pega o primeiro válido (Heal, Protect, Phaze, etc.)
        return candidates[0]


    def get_best_execution_object(self, base_action, battle, history=None):
        if isinstance(base_action, list): base_action = base_action[0]

        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon
        
        if active and opponent:
            is_threat = self.is_threatening(active, opponent)
            if is_threat and active.current_hp_fraction < 0.45:
                if base_action in ["BUFF", "HAZARD", "STATUS", "DEBUFF", "FIELD_CONTROL"]:
                    base_action = "ATTACK_STRONG" 

        try:
            cat = MoveCategory[base_action]
            
            # --- Heurística determinística para categorias secundárias ---
            non_offensive = [
                MoveCategory.STATUS, MoveCategory.BUFF, MoveCategory.DEBUFF,
                MoveCategory.HAZARD, MoveCategory.HEAL, MoveCategory.FIELD_CONTROL,
                MoveCategory.CLEAN_HAZARD, MoveCategory.PROTECT, MoveCategory.ATTACK_PIVOT,
                MoveCategory.ATTACK_TECH, MoveCategory.STAT_CLEAN, MoveCategory.HEAL_STATUS, MoveCategory.PHAZE
            ]
            
            if cat in non_offensive:
                # 1. Filtra lixos instantaneamente (imunidades, substitutos, etc)
                candidates = [
                    m for m in battle.available_moves 
                    if self.classify_move(m) == cat and not self.is_move_useless(m, opponent, battle)
                ]
                
                # 2. Varredura estrita extra para Hazards
                if cat == MoveCategory.HAZARD:
                    candidates = [m for m in candidates if not self.is_hazard_already_set(m, battle)]

                # 3. Manda para a máquina de desempate lógico
                if candidates:
                    best = self._select_best_move_in_category(candidates, cat, active, opponent, battle)
                    if best:
                        return best
                        
                # 4. Fallback: Se o Cérebro pediu STATUS mas não há golpes viáveis (ou alvo imune), ataca!
                base_action = "ATTACK_STRONG"
                
        except KeyError:
            pass 

        if base_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "SWITCH"]:
            if active and opponent and battle.available_switches:
                my_spe = self.estimate_stat(active, 'spe')
                opp_spe = self.estimate_stat(opponent, 'spe')
                is_faster = my_spe > opp_spe
                
                pivot_moves = [m for m in battle.available_moves if m.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']]
                
                if pivot_moves and is_faster:
                    return pivot_moves[0]

            if base_action == "SWITCH_DEFENSIVE":
                switch = self.get_defensive_switch(battle, history)
            else:
                switch = self.get_offensive_switch(battle, history)
                
            if switch: return switch

        # ==========================================================
        # BLINDAGEM DE ESCOPO: BLOCO DE ATAQUE CORRIGIDO
        # ==========================================================
        if active and opponent:
            if base_action in ["ATTACK_STRONG", "ATTACK_PREDICTIVE"]:
                valid_moves = [m for m in battle.available_moves if self.classify_move(m) in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT]]
                valid_moves = [m for m in valid_moves if not self.is_move_useless(m, opponent, battle)]
                
                if not valid_moves: 
                    valid_moves = [m for m in battle.available_moves if m.base_power > 0 and opponent.damage_multiplier(m) > 0]
                    if not valid_moves: 
                        valid_moves = [m for m in battle.available_moves if m.base_power > 0]
                
                if valid_moves:
                    # Garantimos que strong_move sempre terá um valor aqui
                    strong_move = None
                    max_strong_score = -9999
                    
                    for m in valid_moves:
                        score = self.estimate_damage_percent(m, active, opponent)
                        m_priority = getattr(m, 'priority', 0)
                        if m_priority > 0 and score >= opponent.current_hp_fraction:
                            score += 2.0 
                        if hasattr(m, 'secondary') and m.secondary: score += 0.05
                        
                        if score > max_strong_score:
                            max_strong_score = score
                            strong_move = m

                    # Fallback imediato caso a matemática empate em tudo
                    if not strong_move:
                        strong_move = valid_moves[0]

                    if base_action == "ATTACK_STRONG":
                        return strong_move
                    
                    if base_action == "ATTACK_PREDICTIVE":
                        benched_opponents = [m for m in battle.opponent_team.values() if not m.fainted and not m.active]
                        
                        if benched_opponents:
                            predictive_candidates = [m for m in valid_moves if m.type != strong_move.type]
                            
                            if predictive_candidates:
                                best_pred_move = None
                                max_pred_score = -9999
                                
                                for m in predictive_candidates:
                                    avg_bench_dmg = sum(self.estimate_damage_percent(m, active, bench_mon) for bench_mon in benched_opponents) / len(benched_opponents)
                                    score = avg_bench_dmg
                                    
                                    if m.id in ['knockoff', 'scald', 'nuzzle', 'saltcure', 'uturn', 'voltswitch', 'flipturn']:
                                        score += 0.20 
                                        
                                    if hasattr(m, 'secondary') and m.secondary: score += 0.05
                                    
                                    if score > max_pred_score:
                                        max_pred_score = score
                                        best_pred_move = m
                                        
                                if best_pred_move:
                                    return best_pred_move
                        
                        # Se não achou predição boa, bate com o forte
                        return strong_move
        
        # Fallback geral do sistema caso tudo acima falhe
        if battle.available_switches:
            switch = self.get_defensive_switch(battle, history)
            if switch: return switch
            
        if battle.available_moves:
            damaging = [m for m in battle.available_moves if m.base_power > 0 and opponent.damage_multiplier(m) > 0]
            if damaging: 
                return max(damaging, key=lambda m: m.base_power)
            return random.choice(battle.available_moves)
        
        return None