import traceback
import random
from enum import Enum
from poke_env.player import Player

# =============================================================================
# 1. DEFINIÇÕES DE DADOS TÁTICOS
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

class TacticalMode(Enum):
    PRESS = 1
    CONTEST = 2
    GRIND = 3
    ESCAPE = 4
    LEAD = 5
    WALLBREAK = 6

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
    BARRIER = 16
    SWITCH_DEFENSIVE = 17
    SWITCH_OFFENSIVE = 18
    UNKNOWN = 19

# =============================================================================
# 2. O CÉREBRO HEURÍSTICO (INSTINCT BOT)
# =============================================================================

class InstinctBot(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mode_templates = {
            TacticalMode.PRESS: [
                "ATTACK_PREDICTIVE", "ATTACK_STRONG", "BUFF", "ATTACK_TECH",
                "HAZARD", "FIELD_CONTROL", "ATTACK_PIVOT", "CLEAN_HAZARD", 
                "STATUS", "DEBUFF", "HEAL", "HEAL_STATUS", "STAT_CLEAN", 
                "PHAZE", "PROTECT", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE"
            ],
            TacticalMode.CONTEST: [
                "ATTACK_STRONG", "ATTACK_TECH", "PROTECT", "ATTACK_PIVOT",
                "STATUS", "BUFF", "HEAL", "HAZARD", "CLEAN_HAZARD",
                "DEBUFF", "FIELD_CONTROL", "ATTACK_PREDICTIVE", "STAT_CLEAN", 
                "PHAZE", "HEAL_STATUS", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE"
            ],
            TacticalMode.GRIND: [
                "HAZARD", "STATUS", "HEAL", "PROTECT", "DEBUFF", 
                "CLEAN_HAZARD", "PHAZE", "STAT_CLEAN", "HEAL_STATUS", 
                "BUFF", "FIELD_CONTROL", "ATTACK_TECH", "ATTACK_PIVOT", 
                "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE", "ATTACK_STRONG", "ATTACK_PREDICTIVE"
            ],
            TacticalMode.ESCAPE: [
                "SWITCH_DEFENSIVE", "ATTACK_PIVOT", "PROTECT", "SWITCH_OFFENSIVE",
                "ATTACK_TECH", "STATUS", "DEBUFF", "STAT_CLEAN", "PHAZE", 
                "HEAL", "CLEAN_HAZARD", "HEAL_STATUS", "FIELD_CONTROL", 
                "HAZARD", "BUFF", "ATTACK_STRONG", "ATTACK_PREDICTIVE"
            ],
            TacticalMode.LEAD: [
                "HAZARD", "FIELD_CONTROL", "ATTACK_PIVOT", "DEBUFF", 
                "ATTACK_STRONG", "BUFF", "STATUS", "PROTECT", 
                "CLEAN_HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
                "ATTACK_PREDICTIVE", "ATTACK_TECH", "STAT_CLEAN", "HEAL_STATUS", "PHAZE"
            ],
            TacticalMode.WALLBREAK: [
                "ATTACK_TECH", "STATUS", "BUFF", "ATTACK_PIVOT",
                "DEBUFF", "HAZARD", "ATTACK_STRONG", "ATTACK_PREDICTIVE",
                "HEAL", "CLEAN_HAZARD", "PROTECT", "SWITCH_OFFENSIVE",
                "STAT_CLEAN", "HEAL_STATUS", "PHAZE", "FIELD_CONTROL", "SWITCH_DEFENSIVE"
            ]
        }

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
    # PERCEPÇÃO E MEMÓRIA
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
            if stat_name == 'hp': val = int(base * 2 + 204)
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

        if stat_name == 'spe': val *= self._get_speed_mod(pokemon)
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

    def estimate_damage_percent(self, move, attacker, defender, battle=None):
        if move.category.name == "STATUS" or move.base_power == 0: return 0.0
            
        bp = float(move.base_power)
        level = float(getattr(attacker, 'level', 100))
        
        multi_hit_moves = ['iciclespear', 'rockblast', 'bulletseed', 'tailslap', 'pinmissile', 'boneclub', 'scaleshot', 'watershuriken']
        if move.id in multi_hit_moves:
            attacker_ability = str(getattr(attacker, 'ability', '')).lower()
            bp *= 5.0 if attacker_ability == 'skilllink' else 3.0 

        if move.category.name == "PHYSICAL":
            atk = self.estimate_stat(attacker, 'def') if move.id == 'bodypress' else self.estimate_stat(attacker, 'atk')
            defense = self.estimate_stat(defender, 'def')
        else:
            atk = self.estimate_stat(attacker, 'spa')
            defense = self.estimate_stat(defender, 'def') if move.id in ['psyshock', 'psystrike', 'secretsword'] else self.estimate_stat(defender, 'spd')
                
        if defense <= 0: defense = 1
        base_dmg = ((((2 * level / 5) + 2) * atk * bp / defense) / 50) + 2
        stab = 1.5 if move.type in attacker.types else 1.0
        type_mod = defender.damage_multiplier(move)
        margin = 0.95
        
        charge_moves = ['fly', 'bounce', 'dig', 'dive', 'phantomforce', 'shadowforce', 'solarbeam', 'solarblade', 'skullbash', 'meteorbeam']
        recharge_moves = ['hyperbeam', 'gigaimpact', 'rockwrecker', 'roaroftime', 'frenzyplant', 'blastburn', 'hydrocannon']
        item_str = str(getattr(attacker, 'item', '')).lower()
        weather = next(iter(battle.weather)).name if battle and battle.weather else "CLEAR"
        known_opp_moves = [m.id for m in defender.moves.values()]
        
        if move.id in charge_moves:
            is_instant = item_str == 'powerherb' or (move.id in ['solarbeam', 'solarblade'] and weather in ['SUNNYDAY', 'DESOLATELAND'])
            if not is_instant:
                margin *= 0.4
                if move.id == 'dig' and 'earthquake' in known_opp_moves: margin *= 0.1
                elif move.id in ['fly', 'bounce'] and any(m in known_opp_moves for m in ['thunder', 'hurricane']): margin *= 0.1
        elif move.id in recharge_moves: margin *= 0.45

        ignores_screens = move.id in ['brickbreak', 'psychicfangs'] or str(getattr(attacker, 'ability', '')).lower() == 'infiltrator'
        if battle and not ignores_screens:
            side_to_check = battle.side_conditions if defender in battle.team.values() else battle.opponent_side_conditions
            active_screens = [str(k).upper() for k in side_to_check.keys()]
            if move.category.name == "PHYSICAL" and any(s in active_screens for s in ['REFLECT', 'AURORA_VEIL']): margin *= 0.5 
            elif move.category.name == "SPECIAL" and any(s in active_screens for s in ['LIGHT_SCREEN', 'AURORA_VEIL']): margin *= 0.5 
        
        final_dmg = base_dmg * stab * type_mod * margin
        max_hp = max(1, self.estimate_stat(defender, 'hp'))
        return final_dmg / max_hp

    def classify_move(self, move):
        move_id = move.id
        if move_id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']: return MoveCategory.ATTACK_PIVOT
        tech_moves = ['knockoff', 'foulplay', 'thief', 'nuzzle', 'scald', 'discharge', 'lavaplume', 'saltcure', 'superfang', 'naturesmadness', 'ruination', 'seismictoss', 'nightshade', 'icywind', 'electroweb', 'rocktomb', 'bulldoze', 'snarl', 'mysticalfire', 'strugglebug', 'fakeout', 'brickbreak', 'psychicfangs', 'bodypress']
        if move_id in tech_moves: return MoveCategory.ATTACK_TECH
        if move_id in ['haze', 'clearsmog']: return MoveCategory.STAT_CLEAN
        if move_id in ['aromatherapy', 'healbell', 'junglehealing']: return MoveCategory.HEAL_STATUS
        if move_id in ['roar', 'whirlwind', 'dragontail', 'circlethrow']: return MoveCategory.PHAZE
        if move_id in ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape', 'trickroom', 'tailwind', 'electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain']: return MoveCategory.FIELD_CONTROL
        if move_id in ['defog', 'rapidspin', 'mortalspin', 'courtchange']: return MoveCategory.CLEAN_HAZARD
        if move_id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']: return MoveCategory.HAZARD
        if move_id in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark', 'silktrap', 'obstruct', 'endure']: return MoveCategory.PROTECT
        if move_id in ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun', 'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap']: return MoveCategory.HEAL
        if move_id in ['reflect', 'lightscreen', 'auroraveil']: return MoveCategory.BARRIER

        if move.category.name == "STATUS":
            if getattr(move, 'heal', 0): return MoveCategory.HEAL
            if getattr(move, 'status', None): return MoveCategory.STATUS
            if getattr(move, 'boosts', None):
                if any(v > 0 for v in move.boosts.values()): return MoveCategory.BUFF
                if any(v < 0 for v in move.boosts.values()): return MoveCategory.DEBUFF
            return MoveCategory.STATUS 
        if move.category.name in ["PHYSICAL", "SPECIAL"] and move.base_power > 0: return MoveCategory.ATTACK_STRONG
        return MoveCategory.UNKNOWN

    def is_move_useless(self, move, opponent, battle, history=None):
        if not move: return True
        active = battle.active_pokemon
        if not active or not opponent: return True

        opp_types = [t.name for t in opponent.types if t]
        opp_abilities = []
        if opponent.ability:
            opp_abilities = [str(opponent.ability).lower()]
        elif opponent.possible_abilities:
            opp_abilities = [str(a).lower() for a in opponent.possible_abilities]
        
        if move.base_power > 0:
            if opponent.damage_multiplier(move) == 0: return True
            if 'wonderguard' in opp_abilities and opponent.damage_multiplier(move) < 2 and move.id != 'struggle': return True
            
        if move.type and move.type.name == 'GROUND' and move.id != 'thousandarrows' and opponent.item and str(opponent.item).lower() == 'airballoon': return True

        if move.id in ['defog', 'rapidspin', 'mortalspin', 'tidyup', 'courtchange']:
            has_hazard = False
            for cond in battle.side_conditions:
                if cond.name in ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']:
                    has_hazard = True
                    break
            if not has_hazard and move.id in ['defog', 'courtchange']:
                for cond in battle.opponent_side_conditions:
                    if cond.name in ['REFLECT', 'LIGHT_SCREEN', 'AURORA_VEIL', 'STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']:
                        has_hazard = True
                        break
            if not has_hazard: return True

        move_priority = getattr(move, 'priority', 0)
        if move.id in ['fakeout', 'firstimpression'] and not getattr(active, 'first_turn', False): return True
        if move_priority > 0:
            if any(ab in opp_abilities for ab in ['dazzling', 'queenlymajesty', 'armortail']): return True
            if 'psychicsurge' in opp_abilities or any('psychicterrain' in str(f).lower() for f in battle.fields.keys()):
                if 'FLYING' not in opp_types and not (opponent.item and str(opponent.item).lower() == 'airballoon') and 'levitate' not in opp_abilities: return True
        
        if move.category.name == "STATUS":
            if active.ability == 'prankster' and 'DARK' in opp_types: return True
            if move.id in ['spore', 'sleeppowder', 'stunspore', 'poisonpowder', 'ragepowder'] and ('GRASS' in opp_types or 'overcoat' in opp_abilities): return True
            if move.id == 'thunderwave' and ('GROUND' in opp_types or 'ELECTRIC' in opp_types): return True
            if move.id == 'leechseed' and 'GRASS' in opp_types: return True
            if any(ab in opp_abilities for ab in ['magicbounce']) and getattr(move, 'target', '') in ['normal', 'allAdjacentFoes', 'foeSide']: return True
            if any(ab in opp_abilities for ab in ['goodasgold', 'magicguard']): return True 
            if move.id in ['confuseray', 'swagger'] and any(ab in opp_abilities for ab in ['owntempo', 'oblivious']): return True

            if move.status:
                if opponent.status: return True 
                
                active_fields = [str(f).upper() for f in battle.fields.keys()]
                grounded_opp = 'FLYING' not in opp_types and not (opponent.item and str(opponent.item).lower() == 'airballoon') and 'levitate' not in opp_abilities
                if grounded_opp:
                    if 'MISTY_TERRAIN' in active_fields: return True
                    if 'ELECTRIC_TERRAIN' in active_fields and move.status.name == 'SLP': return True

                if 'synchronize' in opp_abilities:
                    my_types = [t.name for t in active.types if t]
                    if move.status.name in ['TOX', 'PSN'] and 'POISON' not in my_types and 'STEEL' not in my_types: return True
                    if move.status.name == 'BRN' and 'FIRE' not in my_types: return True
                    if move.status.name == 'PRZ' and 'ELECTRIC' not in my_types and 'GROUND' not in my_types: return True

                if move.status.name in ['TOX', 'PSN']:
                    if 'immunity' in opp_abilities: return True
                    if ('POISON' in opp_types or 'STEEL' in opp_types) and active.ability != 'corrosion': return True
                elif move.status.name == 'BRN' and ('FIRE' in opp_types or any(ab in opp_abilities for ab in ['waterveil', 'waterbubble'])): return True
                elif move.status.name == 'PRZ' and ('ELECTRIC' in opp_types or 'limber' in opp_abilities): return True
                elif move.status.name == 'SLP' and any(ab in opp_abilities for ab in ['insomnia', 'vitalspirit', 'sweetveil']): return True

        if move.category.name == "STATUS":
            boosts = getattr(move, 'boosts', None) or getattr(move, 'self_boost', None)
            if boosts:
                target_str = str(getattr(move, 'target', '')).lower()
                if 'self' in target_str:
                    is_useful = False
                    for stat, boost_amount in boosts.items():
                        if (boost_amount > 0 and active.boosts.get(stat, 0) < 6) or boost_amount < 0:
                            is_useful = True
                            break 
                    if not is_useful: return True
                elif 'normal' in target_str or 'foe' in target_str:
                    if any(ab in opp_abilities for ab in ['clearbody', 'whitesmoke', 'fullmetalbody']) and any(b < 0 for b in boosts.values()): return True

        if move.id == 'substitute':
            if active.effects and any('substitute' in str(e).lower() for e in active.effects): return True
            if active.current_hp_fraction <= 0.25: return True

        if move.id == 'leechseed':
            if opponent.effects and any('leechseed' in str(e).lower() for e in opponent.effects): return True

        current_weather = next(iter(battle.weather)).name if battle.weather else "CLEAR"
        my_side = [str(k).upper() for k in battle.side_conditions.keys()]
        if move.id == 'reflect' and 'REFLECT' in my_side: return True
        if move.id == 'lightscreen' and 'LIGHT_SCREEN' in my_side: return True
        if move.id == 'safeguard' and 'SAFEGUARD' in my_side: return True
        if move.id == 'tailwind' and 'TAILWIND' in my_side: return True
        if move.id == 'auroraveil' and ('AURORA_VEIL' in my_side or current_weather not in ['HAIL', 'SNOW', 'SNOWSCAPE']): return True

        if move.id in ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape']:
            if move.id == 'raindance' and current_weather in ['RAINDANCE', 'PRIMORDIALSEA']: return True
            if move.id == 'sunnyday' and current_weather in ['SUNNYDAY', 'DESOLATELAND']: return True
            if move.id == 'sandstorm' and current_weather == 'SANDSTORM': return True
            if move.id in ['hail', 'snowscape'] and current_weather in ['HAIL', 'SNOW', 'SNOWSCAPE']: return True

        if move.id in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark', 'silktrap', 'obstruct', 'endure']:
            if history and 'last_action' in history:
                last_action_tuple = history.get('last_action')
                if last_action_tuple and "PROTECT" in last_action_tuple[0]: return True

        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        if move.id in ['roar', 'whirlwind', 'dragontail', 'circlethrow'] and (opp_alive <= 1 or 'suctioncups' in opp_abilities): return True 
        if move.id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb'] and opp_alive <= 1: return True

        if move.category in ["Physical", "Special"]:
            opp_species = str(opponent.species).lower()
            if move.type.name == "WATER" and opp_species in ['vaporeon', 'gastrodon', 'seismitoad', 'toxicroak', 'mantine', 'clodsire', 'volcanion']: return True
            elif move.type.name == "FIRE" and opp_species in ['heatran', 'chandelure', 'arcanine', 'ceruledge', 'houndoom', 'dachsbun']: return True
            elif move.type.name == "ELECTRIC" and opp_species in ['jolteon', 'thundurus', 'thundurustherian', 'zeraora', 'electivire', 'raichu', 'marowakalola']: return True
            elif move.type.name == "GRASS" and opp_species in ['azumarill', 'goodra', 'bouffalant']: return True
            elif move.type.name == "GROUND" and opp_species in ['rotom', 'rotomwash', 'rotomheat', 'rotommow', 'latios', 'latias', 'hydreigon', 'cresselia', 'weezing', 'orthworm']: return True
            
        if move.id in ['recover', 'roost', 'slackoff', 'softboiled', 'milkdrink', 'shoreup', 'moonlight', 'morningsun', 'synthesis', 'healorder', 'wish'] or (hasattr(move, 'heal') and move.heal and move.category == "Status"):
            if active.current_hp_fraction >= 0.95: return True
        if move.id in ['aromatherapy', 'healbell'] and not any(m.status is not None for m in battle.team.values()): return True

        return False

    def get_macro_context(self, battle):
        my_alive = len([m for m in battle.team.values() if not m.fainted])
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        total_alive = my_alive + opp_alive
        piece_advantage = my_alive - opp_alive
        
        if total_alive >= 10: return "OPENING"
        if piece_advantage >= 2: return "DOMINATING"
        if piece_advantage <= -2: return "RECOVERING"
        return "CLUTCH" if total_alive <= 5 else "BRAWL"

    def get_matchup_state(self, my_mon, opp_mon) -> MatchupState:
        if not my_mon or not opp_mon: return MatchupState.NEUTRAL
        my_moves = [m for m in my_mon.moves.values() if m.base_power > 0]
        my_best_mult = max([opp_mon.damage_multiplier(move) for move in my_moves]) if my_moves else 0.0

        opp_best_mult = 0.0
        for type_ in opp_mon.types:
             if type_: opp_best_mult = max(opp_best_mult, my_mon.damage_multiplier(type_))
        
        known_opp_moves = [m for m in opp_mon.moves.values() if m.base_power > 0]
        for move in known_opp_moves:
             opp_best_mult = max(opp_best_mult, my_mon.damage_multiplier(move))

        my_se, my_neutral, my_nve = my_best_mult > 1.0, my_best_mult == 1.0, my_best_mult < 1.0
        opp_se, opp_neutral, opp_nve = opp_best_mult > 1.0, opp_best_mult == 1.0, opp_best_mult < 1.0

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
        if my_mon.current_hp_fraction < 0.45 and self.estimate_stat(opp_mon, 'spe') > self.estimate_stat(my_mon, 'spe'):
            if max(self.estimate_stat(opp_mon, 'atk'), self.estimate_stat(opp_mon, 'spa')) > 250: return True
        return False

    def is_hazard_already_set(self, move, battle):
        hazard_types = ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']
        target_hazard = {'stealthrock': 'STEALTH_ROCK', 'stickyweb': 'STICKY_WEB', 'spikes': 'SPIKES', 'toxicspikes': 'TOXIC_SPIKES'}.get(move.id)
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
            dmg += 0.041 * int(battle.side_conditions.get('spikes', 1))
        return dmg

    def get_available_actions(self, battle):
        available = set()
        if battle.available_switches:
            available.update([MoveCategory.SWITCH_DEFENSIVE.name, MoveCategory.SWITCH_OFFENSIVE.name])
        if battle.available_moves:
            damaging_types = set()  
            for move in battle.available_moves:
                if self.is_move_useless(move, battle.opponent_active_pokemon, battle): continue
                cat = self.classify_move(move)
                if cat != MoveCategory.UNKNOWN:
                    if cat == MoveCategory.HAZARD and self.is_hazard_already_set(move, battle): continue
                    available.add(cat.name)
                    if cat in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT] and move.type:
                        damaging_types.add(move.type)
            if MoveCategory.ATTACK_STRONG.name in available and len(damaging_types) > 1 and any(not m.fainted and not m.active for m in battle.opponent_team.values()):
                available.add(MoveCategory.ATTACK_PREDICTIVE.name)
        return list(available) or (["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"] if battle.available_switches else ["ATTACK_STRONG"])

    # =========================================================================
    # HELPERS DE ROLE
    # =========================================================================

    def _is_physical(self, pokemon): return self.estimate_stat(pokemon, 'atk') > self.estimate_stat(pokemon, 'spa') if pokemon else True
    def _has_recovery(self, pokemon): return any(m.id in ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun', 'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap'] for m in pokemon.moves.values()) if pokemon else False
    def _has_move(self, pokemon, move_ids): return any(m.id in move_ids for m in pokemon.moves.values()) if pokemon else False
    def _opponent_can_setup(self, opponent): return any(m.id in ['swordsdance', 'dragondance', 'nastyplot', 'quiverdance', 'shellsmash', 'shiftgear', 'calmmind', 'bulkup', 'workup', 'coil'] for m in opponent.moves.values()) if opponent else False
    
    def _is_active_best_remaining(self, active, opponent, battle):
        if not battle.available_switches: return True
        active_score = self._get_survival_score(active, opponent, battle, is_active=True)
        best_bench_score = max(self._get_survival_score(m, opponent, battle, is_active=False) for m in battle.available_switches)
        return best_bench_score <= active_score + 50

    def _get_survival_score(self, candidate, opponent, battle, is_active=False):
        if not candidate: return -9999
        hp_frac = candidate.current_hp_fraction
        if not is_active and hp_frac <= self._get_hazard_damage(candidate, battle) + 0.05: return -9999 
                
        score = 150 if hp_frac >= 0.7 else 50 if hp_frac >= 0.4 else -100
        if not opponent: return score

        opp_types_obj = [t for t in opponent.types if t]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]
        
        has_weakness = False
        for obj in opp_types_obj + known_opp_moves:
            mult = candidate.damage_multiplier(obj)
            if mult > 1.0:
                score -= (100 if type(obj) != str else 150) * mult
                has_weakness = True
            elif mult < 1.0:
                score += (50 if type(obj) != str else 75) / max(mult, 0.1)

        if self.estimate_stat(candidate, 'spe') > self.estimate_stat(opponent, 'spe'):
            score += 100
            if any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in candidate.moves.values()): score += 150
        elif has_weakness: score -= 200

        matchup = self.get_matchup_state(candidate, opponent)
        if matchup == MatchupState.DOMINANT: score += 200
        elif matchup == MatchupState.DEFENSIVE_ADV: score += 100
        elif matchup == MatchupState.CRITICAL_DIS: score -= 300
        return score

    # =========================================================================
    # CAMADAS 1 E 3: MATCHUPS E MODIFICADORES TÁTICOS
    # =========================================================================

    def _get_tactical_mode(self, matchup, my_role, opp_role, is_faster, my_hp_frac, opp_hp_frac, is_threat, active, opponent):
        if my_role == Role.TANK and opp_role == Role.SWEEPER:
            opp_is_physical = self._is_physical(opponent)
            my_def, my_spd = active.base_stats.get('def', 0), active.base_stats.get('spd', 0)
            if not ((opp_is_physical and my_def >= my_spd) or (not opp_is_physical and my_spd > my_def)):
                return TacticalMode.ESCAPE
        if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]: return TacticalMode.PRESS
        if matchup in [MatchupState.VOLATILE, MatchupState.NEUTRAL]: return TacticalMode.CONTEST
        if matchup in [MatchupState.STALEMATE, MatchupState.DEFENSIVE_ADV]: return TacticalMode.GRIND
        return TacticalMode.ESCAPE

    def _mod_sweeper_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        if is_faster:
            if "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
            if "ATTACK_PREDICTIVE" in mod: mod.insert(1, mod.pop(mod.index("ATTACK_PREDICTIVE")))
            if "BUFF" in mod: mod.append(mod.pop(mod.index("BUFF")))
        else:
            if is_threat:
                if "SWITCH_DEFENSIVE" in mod: mod.insert(0, "SWITCH_DEFENSIVE")
                if "ATTACK_TECH" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_TECH")))
            elif "BUFF" in mod: mod.insert(0, mod.pop(mod.index("BUFF")))
        return mod

    def _mod_sweeper_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        tank_defense_high = (self._is_physical(active) and opponent.base_stats.get('def', 0) >= 100) or (not self._is_physical(active) and opponent.base_stats.get('spd', 0) >= 100)
        if tank_defense_high and "BUFF" in mod: mod.insert(0, mod.pop(mod.index("BUFF")))
        elif "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))

        if "DEBUFF" in mod: mod.insert(0, mod.pop(mod.index("DEBUFF")))
        if "BUFF" in mod and "DEBUFF" in mod and mod.index("BUFF") < mod.index("DEBUFF"):
            mod[mod.index("BUFF")], mod[mod.index("DEBUFF")] = mod[mod.index("DEBUFF")], mod[mod.index("BUFF")]

        if my_hp_frac <= 0.5 and "ATTACK_PIVOT" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_PIVOT")))
        return mod

    def _mod_sweeper_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        if "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
        if opp_hp_frac >= 0.7 and "BUFF" in mod: mod.insert(0, mod.pop(mod.index("BUFF")))
        if not is_faster and is_threat and "SWITCH_DEFENSIVE" in mod: mod.insert(0, "SWITCH_DEFENSIVE")
        return mod

    def _mod_tank_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        opp_is_physical = self._is_physical(opponent)
        if not ((opp_is_physical and active.base_stats.get('def', 0) >= active.base_stats.get('spd', 0)) or (not opp_is_physical and active.base_stats.get('spd', 0) > active.base_stats.get('def', 0))):
            if "SWITCH_DEFENSIVE" in mod: mod.insert(0, "SWITCH_DEFENSIVE")
            if "PROTECT" in mod: mod.insert(0, "PROTECT")
            return mod

        if "STATUS" in mod: mod.insert(0, mod.pop(mod.index("STATUS")))
        if "ATTACK_TECH" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_TECH")))
        if "HEAL" in mod: mod.insert(0, mod.pop(mod.index("HEAL")))
        if self._has_recovery(opponent) and "ATTACK_PIVOT" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_PIVOT")))
        return mod

    def _mod_tank_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        if self._has_move(active, ['futuresight', 'doomdesire']) and self._has_move(active, ['uturn', 'voltswitch', 'flipturn', 'teleport']):
            if "ATTACK_TECH" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_TECH")))
            if "ATTACK_PIVOT" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_PIVOT")))
            return mod

        for m in ["HAZARD", "STATUS", "HEAL"]:
            if m in mod: mod.insert(0, mod.pop(mod.index(m)))
        if self._has_move(active, ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker']) and "PROTECT" in mod: mod.insert(0, mod.pop(mod.index("PROTECT")))
        if "SWITCH_OFFENSIVE" in mod: mod.insert(0, mod.pop(mod.index("SWITCH_OFFENSIVE")))
        if "ATTACK_TECH" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_TECH")))
        return mod

    def _mod_tank_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        if "HAZARD" in mod: mod.insert(0, mod.pop(mod.index("HAZARD")))
        if "STATUS" in mod: mod.insert(0, mod.pop(mod.index("STATUS")))
        if self._has_move(opponent, ['uturn', 'voltswitch', 'flipturn', 'teleport']) and "ATTACK_TECH" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_TECH")))
        return mod

    def _mod_utility_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        if is_threat and not is_faster:
            if "SWITCH_DEFENSIVE" in mod: mod.insert(0, "SWITCH_DEFENSIVE")
            if "ATTACK_TECH" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_TECH")))
        if self._opponent_can_setup(opponent) and "HAZARD" in mod: mod.insert(0, mod.pop(mod.index("HAZARD")))
        elif "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
        if is_faster and opp_hp_frac <= 0.4 and "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
        return mod

    def _mod_utility_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        for m in ["STATUS", "ATTACK_PIVOT", "DEBUFF", "HAZARD"]:
            if m in mod: mod.insert(0, mod.pop(mod.index(m)))
        if "ATTACK_STRONG" in mod and "SWITCH_DEFENSIVE" in mod: mod.remove("ATTACK_STRONG")
        return mod

    def _mod_utility_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        mod = base.copy()
        for m in ["HAZARD", "STATUS", "FIELD_CONTROL", "ATTACK_PIVOT", "DEBUFF"]:
            if m in mod: mod.insert(0, mod.pop(mod.index(m)))
        return mod

    def _mod_escape(self, base, active, opponent, is_faster, my_role, battle):
        mod = base.copy()
        if self._is_active_best_remaining(active, opponent, battle):
            for m in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]:
                if m in mod: mod.remove(m)
            if is_faster:
                if "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
                if "ATTACK_TECH" in mod: mod.insert(1, mod.pop(mod.index("ATTACK_TECH")))
            elif my_role == Role.TANK:
                opp_is_physical = self._is_physical(opponent)
                if ((opp_is_physical and active.base_stats.get('def', 0) >= active.base_stats.get('spd', 0)) or (not opp_is_physical and active.base_stats.get('spd', 0) > active.base_stats.get('def', 0))) and "STATUS" in mod:
                    mod.insert(0, mod.pop(mod.index("STATUS")))
                elif "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
            elif "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
            mod.extend(["ATTACK_PIVOT", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"])
        return mod

    def _mod_lead(self, base, active, opponent, battle, is_faster):
        mod = base.copy()
        my_team = list(battle.team.values())
        needs_field_control = any(str(m.ability).lower() in ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration', 'drought', 'drizzle', 'sandstream', 'snowwarning'] for m in my_team) or (sum(m.base_stats.get('spe', 50) for m in my_team) / len(my_team) < 70)
        weather_active = battle.weather is not None and len(battle.weather) > 0
        matchup = self.get_matchup_state(active, opponent)

        if matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]:
            if is_faster and "ATTACK_PIVOT" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_PIVOT")))
            elif not is_faster and "SWITCH_DEFENSIVE" in mod: mod.insert(0, mod.pop(mod.index("SWITCH_DEFENSIVE")))
            return mod

        if needs_field_control and not weather_active and "FIELD_CONTROL" in mod: mod.insert(0, mod.pop(mod.index("FIELD_CONTROL")))
        elif weather_active and matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV] and "HAZARD" in mod: mod.insert(0, mod.pop(mod.index("HAZARD")))
        if weather_active and is_faster and "ATTACK_STRONG" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_STRONG")))
        return mod

    def _mod_wallbreak(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac):
        mod = base.copy()
        has_rec = self._has_recovery(opponent)
        if has_rec:
            if "STATUS" in mod: mod.insert(0, mod.pop(mod.index("STATUS")))
            if "ATTACK_TECH" in mod: mod.insert(1, mod.pop(mod.index("ATTACK_TECH")))
        if "BUFF" in mod and my_hp_frac >= 0.60: mod.insert(2 if has_rec else 0, mod.pop(mod.index("BUFF")))
        if my_hp_frac < 0.40 and "ATTACK_PIVOT" in mod: mod.insert(0, mod.pop(mod.index("ATTACK_PIVOT")))
        return mod

    # =========================================================================
    # ÁRVORE TÁTICA & ACTION MASKING
    # =========================================================================

    def get_instinct_profile(self, battle):
        candidate_mask = self.get_available_actions(battle)
        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            return ["SWITCH_DEFENSIVE"] if battle.available_switches else ["ATTACK_STRONG"]

        active, opp = battle.active_pokemon, battle.opponent_active_pokemon
        my_role, opp_role = self.get_role(active), self.get_role(opp)
        matchup = self.get_matchup_state(active, opp)
        is_faster = self.estimate_stat(active, 'spe') > self.estimate_stat(opp, 'spe')
        my_hp_frac, opp_hp_frac = active.current_hp_fraction, opp.current_hp_fraction
        is_threat = self.is_threatening(active, opp)
        macro_context = self.get_macro_context(battle)
        
        mode = TacticalMode.LEAD if macro_context == "OPENING" else self._get_tactical_mode(matchup, my_role, opp_role, is_faster, my_hp_frac, opp_hp_frac, is_threat, active, opp)
        base_priorities = self.mode_templates[mode].copy()

        if mode == TacticalMode.LEAD: priorities = self._mod_lead(base_priorities, active, opp, battle, is_faster)
        elif mode == TacticalMode.WALLBREAK: priorities = self._mod_wallbreak(base_priorities, active, opp, is_faster, my_hp_frac, opp_hp_frac)
        elif mode == TacticalMode.ESCAPE: priorities = self._mod_escape(base_priorities, active, opp, is_faster, my_role, battle)
        else: priorities = self.role_modifiers.get((my_role, opp_role), lambda b, *args: b)(base_priorities, active, opp, is_faster, my_hp_frac, opp_hp_frac, is_threat)

        ranking_list = []
        for intent in priorities:
            if intent in candidate_mask:
                if intent == "HEAL" and my_hp_frac >= 0.85: continue
                if intent == "BUFF" and my_hp_frac <= 0.35: continue
                if intent == "STATUS" and opp.status is not None: continue
                if intent not in ranking_list: ranking_list.append(intent)

        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        if opp_alive <= 2 and "HAZARD" in ranking_list:
            ranking_list.remove("HAZARD")
            ranking_list.append("HAZARD")

        opp_side_conds = [str(k).upper() for k in battle.opponent_side_conditions.keys()]
        phys_blk, spec_blk = 'REFLECT' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds, 'LIGHT_SCREEN' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds
        
        if phys_blk or spec_blk:
            benched = [m for m in battle.team.values() if not m.fainted and not m.active]
            can_bypass = (phys_blk and not spec_blk and any(self.get_role(m) == Role.SWEEPER and not self._is_physical(m) for m in benched)) or (spec_blk and not phys_blk and any(self.get_role(m) == Role.SWEEPER and self._is_physical(m) for m in benched))
            boost_intents = ["CLEAN_HAZARD"] + (["SWITCH_OFFENSIVE", "ATTACK_PIVOT", "BUFF"] if can_bypass else ["SWITCH_DEFENSIVE", "STATUS", "HEAL", "PROTECT", "DEBUFF"])
            for b_intent in reversed(boost_intents):
                if b_intent in ranking_list: 
                    ranking_list.remove(b_intent)
                    ranking_list.insert(0, b_intent)

        # Lethal Check
        has_lethal = False
        if "ATTACK_STRONG" in candidate_mask or "ATTACK_PREDICTIVE" in candidate_mask:
            for m in battle.available_moves:
                if m.base_power > 0 and not self.is_move_useless(m, opp, battle):
                    dmg = self.estimate_damage_percent(m, active, opp, battle)
                    if dmg >= opp_hp_frac:
                        has_lethal = True
                        break
        
        if has_lethal:
            for atk in reversed(["ATTACK_PREDICTIVE", "ATTACK_STRONG"]):
                if atk in ranking_list:
                    ranking_list.remove(atk)
                    ranking_list.insert(0, atk)

        if not ranking_list:
            atk_options = [a for a in candidate_mask if "ATTACK" in a]
            ranking_list.append(atk_options[0] if atk_options else (candidate_mask[0] if candidate_mask else "ATTACK_STRONG"))

        return ranking_list

    # =========================================================================
    # EXECUTORES
    # =========================================================================

    def teampreview(self, battle):
        try:
            my_team, opp_team = list(battle.team.values()), list(battle.opponent_team.values())
            if not opp_team: return "/team 123456"
            best_lead = None
            
            weather_setters = ['drought', 'drizzle', 'sandstream', 'snowwarning']
            my_weather_setter = next((m for m in my_team if str(m.ability) in weather_setters), None)
            opp_has_weather = any(str(m.ability) in weather_setters for m in opp_team)
            
            if my_weather_setter and opp_has_weather: best_lead = my_weather_setter
            
            if not best_lead:
                for m in my_team:
                    has_hazard = any(move.id in ['stealthrock', 'spikes', 'stickyweb'] for move in m.moves.values())
                    fast_or_sash = str(m.item) == 'focussash' or m.base_stats.get('spe', 0) > 105
                    if has_hazard and fast_or_sash:
                        best_lead = m
                        break

            if not best_lead:
                pivots = [m for m in my_team if any(move.id in ['uturn', 'voltswitch', 'flipturn'] for move in m.moves.values())]
                if pivots: best_lead = max(pivots, key=lambda m: m.base_stats.get('spe', 0))
                
            if not best_lead:
                avg_speed = sum(m.base_stats.get('spe', 50) for m in my_team) / len(my_team)
                if avg_speed > 85: best_lead = max(my_team, key=lambda m: m.base_stats.get('spe', 50))
                else: best_lead = max(my_team, key=lambda m: m.base_stats.get('hp', 50) + m.base_stats.get('def', 50) + m.base_stats.get('spd', 50))

            try: lead_index = my_team.index(best_lead) + 1
            except ValueError: lead_index = 1
                
            rest_indices = [str(i + 1) for i in range(len(my_team)) if i + 1 != lead_index]
            return f"/team {lead_index}" + "".join(rest_indices)
        except Exception: return "/team 123456"

    def get_defensive_switch(self, battle):
        candidates = battle.available_switches
        if not candidates: return None
        opponent = battle.opponent_active_pokemon
        opp_types_obj = [t for t in opponent.types if t] if opponent else []
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0] if opponent else []
        opp_is_physical = self._is_physical(opponent) if opponent else True
        
        def get_score(cand):
            if cand.current_hp_fraction <= self._get_hazard_damage(cand, battle) + 0.05: return -9999 
            score = 200 if cand.current_hp_fraction >= 0.7 else 100 if cand.current_hp_fraction >= 0.35 else 50
            if self.get_role(cand) == Role.TANK: score += 100

            if opponent:
                has_weakness = False
                for obj in opp_types_obj + known_opp_moves:
                    mult = cand.damage_multiplier(obj)
                    if mult > 1.0:
                        score -= (150 if type(obj) != str else 200) * mult  
                        has_weakness = True
                    elif mult < 1.0:
                        score += min(300, (50 if type(obj) != str else 75) / max(mult, 0.1))

                if self.estimate_stat(opponent, 'spe') > self.estimate_stat(cand, 'spe') and has_weakness: score -= 300

                cand_abi = str(cand.ability).lower() if cand.ability else ""
                type_absorb_map = {'WATER': ['waterabsorb', 'dryskin', 'stormdrain'], 'GROUND': ['levitate'], 'GRASS': ['sapsipper'], 'FIRE': ['flashfire'], 'ELECTRIC': ['voltabsorb', 'lightningrod']}
                for t_str in [t.name for t in opp_types_obj]:
                    if t_str in type_absorb_map and cand_abi in type_absorb_map[t_str]: score += 500 
                            
                cand_matchup = self.get_matchup_state(cand, opponent)
                score += {MatchupState.DOMINANT: 300, MatchupState.DEFENSIVE_ADV: 200, MatchupState.STALEMATE: 100, MatchupState.NEUTRAL: 50, MatchupState.DEFENSIVE_DIS: -150, MatchupState.CRITICAL_DIS: -300}.get(cand_matchup, 0)
                if (opp_is_physical and cand.base_stats.get('def', 0) > cand.base_stats.get('spd', 0)) or (not opp_is_physical and cand.base_stats.get('spd', 0) > cand.base_stats.get('def', 0)): score += 100
            return score
        return max(candidates, key=get_score)

    def get_offensive_switch(self, battle):
        candidates = battle.available_switches
        if not candidates: return None
        opponent = battle.opponent_active_pokemon
        opp_types_obj = [t for t in opponent.types if t] if opponent else []
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0] if opponent else []

        def get_score(cand):
            if cand.current_hp_fraction <= self._get_hazard_damage(cand, battle) + 0.05: return -9999 
            score = 200 if cand.current_hp_fraction >= 0.7 else 100 if cand.current_hp_fraction >= 0.35 else 50
            if self.get_role(cand) == Role.SWEEPER: score += 100
            
            if battle.weather and (str(cand.ability).lower() if cand.ability else "") in ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration']: score += 200 
            if opponent:
                has_weakness = False
                for obj in opp_types_obj + known_opp_moves:
                    mult = cand.damage_multiplier(obj)
                    if mult > 1.0:
                        score -= (150 if type(obj) != str else 200) * mult 
                        has_weakness = True
                    elif mult < 1.0:
                        score += min(150, (25 if type(obj) != str else 50) / max(mult, 0.1))

                if self.estimate_stat(opponent, 'spe') > self.estimate_stat(cand, 'spe') and has_weakness: score -= 300
                if self.estimate_stat(cand, 'spe') > self.estimate_stat(opponent, 'spe'): score += 150
                if any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.0 for m in cand.moves.values()): score += 150
            return score
        return max(candidates, key=get_score)

    def get_post_faint_switch(self, battle):
        candidates = battle.available_switches
        opponent = battle.opponent_active_pokemon
        if not opponent or not candidates: return candidates[0] if candidates else None

        def get_general_score(cand):
            score = 150 if cand.current_hp_fraction >= 0.7 else 50 if cand.current_hp_fraction >= 0.4 else -100
            has_weakness = False
            for obj in [t for t in opponent.types if t] + [m for m in opponent.moves.values() if m.base_power > 0]:
                mult = cand.damage_multiplier(obj)
                if mult > 1.0:
                    score -= (100 if type(obj) != str else 150) * mult
                    has_weakness = True
                elif mult < 1.0:
                    score += (50 if type(obj) != str else 75) / max(mult, 0.1)

            if self.estimate_stat(cand, 'spe') > self.estimate_stat(opponent, 'spe'):
                score += 100
                if any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in cand.moves.values()): score += 150
            elif has_weakness: score -= 200
                    
            score += {MatchupState.DOMINANT: 200, MatchupState.DEFENSIVE_ADV: 100, MatchupState.CRITICAL_DIS: -300}.get(self.get_matchup_state(cand, opponent), 0)
            return score
        return max(candidates, key=get_general_score)
       
    def _select_best_move_in_category(self, candidates, cat, active, opponent, battle):
        if not candidates: return None
        if cat == MoveCategory.HAZARD: return max(candidates, key=lambda m: {'stealthrock': 4, 'stickyweb': 3, 'spikes': 2, 'toxicspikes': 1}.get(m.id, 0))
        if cat == MoveCategory.STATUS:
            def status_score(m):
                s = float(m.accuracy) if isinstance(m.accuracy, (int, float)) else 100.0
                return s + (50 if m.id in ['spore', 'sleeppowder', 'yawn'] else 30 if m.id in ['willowisp', 'thunderwave', 'glare'] else 20 if m.id == 'toxic' else 0)
            return max(candidates, key=status_score)
        if cat == MoveCategory.ATTACK_TECH:
            def tech_score(m):
                s = float(m.base_power)
                has_hazard = any(cond.name in ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB'] for cond in battle.side_conditions)
                if m.id in ['rapidspin', 'mortalspin'] and has_hazard: s += 500
                elif m.id in ['nuzzle', 'scald', 'discharge', 'lavaplume'] and not opponent.status: s += 300
                elif m.id == 'knockoff': s += 200
                return s
            return max(candidates, key=tech_score)
        if cat == MoveCategory.ATTACK_PIVOT: return max(candidates, key=lambda m: m.base_power * (1.5 if m.type in active.types else 1.0))
        return candidates[0]

    def get_best_execution_object(self, base_action, battle):
        active, opponent = battle.active_pokemon, battle.opponent_active_pokemon
        if active and opponent and self.is_threatening(active, opponent) and active.current_hp_fraction < 0.45 and base_action in ["BUFF", "HAZARD", "STATUS", "DEBUFF", "FIELD_CONTROL"]:
            base_action = "ATTACK_STRONG" 

        try:
            cat = MoveCategory[base_action]
            if cat in [MoveCategory.STATUS, MoveCategory.BUFF, MoveCategory.DEBUFF, MoveCategory.HAZARD, MoveCategory.HEAL, MoveCategory.FIELD_CONTROL, MoveCategory.CLEAN_HAZARD, MoveCategory.PROTECT, MoveCategory.ATTACK_PIVOT, MoveCategory.ATTACK_TECH, MoveCategory.STAT_CLEAN, MoveCategory.HEAL_STATUS, MoveCategory.PHAZE]:
                cands = [m for m in battle.available_moves if self.classify_move(m) == cat and not self.is_move_useless(m, opponent, battle)]
                if cat == MoveCategory.HAZARD: cands = [m for m in cands if not self.is_hazard_already_set(m, battle)]
                if cands: return self._select_best_move_in_category(cands, cat, active, opponent, battle)
                base_action = "ATTACK_STRONG"
        except KeyError: pass 

        if base_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "SWITCH"]:
            if active and opponent and battle.available_switches and self.estimate_stat(active, 'spe') > self.estimate_stat(opponent, 'spe'):
                pivot = next((m for m in battle.available_moves if m.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']), None)
                if pivot: return pivot
            return self.get_defensive_switch(battle) if base_action == "SWITCH_DEFENSIVE" else self.get_offensive_switch(battle)

        if active and opponent and base_action in ["ATTACK_STRONG", "ATTACK_PREDICTIVE"]:
            valid_moves = [m for m in battle.available_moves if self.classify_move(m) in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT] and not self.is_move_useless(m, opponent, battle)]
            if not valid_moves: valid_moves = [m for m in battle.available_moves if m.base_power > 0 and opponent.damage_multiplier(m) > 0] or [m for m in battle.available_moves if m.base_power > 0]
            
            if valid_moves:
                strong_move, max_strong_score = None, -9999
                opp_hp_frac, opp_alive = opponent.current_hp_fraction, len([m for m in battle.opponent_team.values() if not m.fainted])
                benched_opps = [m for m in battle.opponent_team.values() if not m.fainted and not m.active]
                
                for m in valid_moves:
                    score = self.estimate_damage_percent(m, active, opponent, battle)
                    if getattr(m, 'priority', 0) > 0 and score >= opp_hp_frac: score += 5.0 
                    
                    if m.id in ['bravebird', 'flareblitz', 'doubleedge', 'woodhammer', 'wildcharge']:
                        if score >= opp_hp_frac and opp_alive > 1:
                            if any(self.get_matchup_state(active, b) in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV] for b in benched_opps) and any(om.id != m.id and self.estimate_damage_percent(om, active, opponent, battle) >= opp_hp_frac for om in valid_moves): 
                                score -= 2.0
                                
                    if m.id in ['closecombat', 'superpower', 'dracometeor', 'leafstorm', 'overheat', 'makeitrain', 'fleurcannon']:
                        if score < opp_hp_frac: score -= 0.3
                        if (m.category.name == "SPECIAL" and active.boosts.get('spa', 0) < 0) or (m.category.name == "PHYSICAL" and active.boosts.get('atk', 0) < 0): 
                            score -= 1.5
                            
                    if m.id in ['iciclespear', 'rockblast', 'bulletseed', 'tailslap', 'pinmissile', 'watershuriken']: score += 0.2 
                    if getattr(m, 'secondary', None): score += 0.05
                    
                    if score > max_strong_score: max_strong_score, strong_move = score, m

                strong_move = strong_move or valid_moves[0]
                if base_action == "ATTACK_STRONG": return strong_move
                
                if base_action == "ATTACK_PREDICTIVE" and benched_opps:
                    pred_cands = [m for m in valid_moves if m.type != strong_move.type]
                    if pred_cands:
                        best_pred, max_pred = None, -9999
                        for m in pred_cands:
                            score = sum(self.estimate_damage_percent(m, active, b, battle) for b in benched_opps) / len(benched_opps)
                            if m.id in ['knockoff', 'scald', 'nuzzle', 'saltcure', 'uturn', 'voltswitch', 'flipturn']: score += 0.20 
                            if getattr(m, 'secondary', None): score += 0.05
                            if score > max_pred: max_pred, best_pred = score, m
                        if best_pred: return best_pred
                return strong_move
        
        return battle.available_switches[0] if battle.available_switches else (battle.available_moves[0] if battle.available_moves else None)

    def choose_move(self, battle):
        try:
            if battle.force_switch or (battle.active_pokemon and battle.active_pokemon.fainted):
                switch = self.get_post_faint_switch(battle)
                return self.create_order(switch) if switch else self.choose_random_move(battle)

            if not battle.active_pokemon or not battle.opponent_active_pokemon:
                return self.choose_random_move(battle)

            intents = self.get_instinct_profile(battle)
            for intent in intents:
                obj = self.get_best_execution_object(intent, battle)
                if obj: return self.create_order(obj)
                
            return self.choose_random_move(battle)

        except Exception:
            traceback.print_exc()
            return self.choose_random_move(battle)