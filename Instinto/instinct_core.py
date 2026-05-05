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

        # =====================================================================
        # CAMADA 3: MODIFICADORES POR ROLE
        # Cada função recebe o template base e o contexto completo.
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
        if hp >= 0.7: return "FULL"
        if hp >= 0.35: return "MED"
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
        
        # Uso elegante do Enum (t.name)
        cand_types_str = [t.name for t in candidate.types if t]
        
        if 'STEALTH_ROCK' in cond_keys: 
            for t in candidate.types:
                if t:
                    PokemonTypeEnum = type(t)
                    rock_enum = getattr(PokemonTypeEnum, 'ROCK', None)
                    if rock_enum: 
                        dmg += 0.125 * candidate.damage_multiplier(rock_enum)
                        break
            
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

    def _is_active_best_remaining(self, active, opponent, battle):
        if not battle.available_switches:
            return True

        # Pontua o quão "ferrado" o Pokémon ativo está
        active_score = self._get_survival_score(active, opponent, battle, is_active=True)

        best_bench_score = -9999
        for bench_mon in battle.available_switches:
            bench_score = self._get_survival_score(bench_mon, opponent, battle, is_active=False)
            if bench_score > best_bench_score:
                best_bench_score = bench_score

        # A Regra de Ouro: Só fugimos se o banco for CONSIDERAVELMENTE mais seguro.
        # Uma margem de 50 pontos evita que o bot troque de um Pokémon ruim para outro ruim.
        if best_bench_score > active_score + 50:
            return False

        return True

    def get_state(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active or not opponent:
            return ("UNKNOWN",) * 15
        
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
            self.get_mechanic_state(battle),
            self.get_macro_context(battle)
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

    def estimate_damage_percent(self, move, attacker, defender, battle=None):
        if move.category.name == "STATUS" or move.base_power == 0: return 0.0
            
        bp = float(move.base_power)
        level = float(getattr(attacker, 'level', 100))
        
        # --- 1. ATUALIZAÇÃO MULTI-HIT ---
        multi_hit_moves = ['iciclespear', 'rockblast', 'bulletseed', 'tailslap', 'pinmissile', 'boneclub', 'scaleshot', 'watershuriken']
        if move.id in multi_hit_moves:
            attacker_ability = str(getattr(attacker, 'ability', '')).lower()
            if attacker_ability == 'skilllink':
                bp *= 5.0  # Dano máximo garantido
            else:
                bp *= 3.0  # Dano médio esperado (3 hits)

        # Identificação de Atributos Ofensivos/Defensivos
        if move.category.name == "PHYSICAL":
            atk = self.estimate_stat(attacker, 'atk')
            if move.id == 'bodypress': atk = self.estimate_stat(attacker, 'def')
            defense = self.estimate_stat(defender, 'def')
        else:
            atk = self.estimate_stat(attacker, 'spa')
            defense = self.estimate_stat(defender, 'spd')
            if move.id in ['psyshock', 'psystrike', 'secretsword']: 
                defense = self.estimate_stat(defender, 'def')
                
        if defense <= 0: defense = 1
        
        base_dmg = ((((2 * level / 5) + 2) * atk * bp / defense) / 50) + 2
        
        stab = 1.5 if move.type in attacker.types else 1.0
        type_mod = defender.damage_multiplier(move)
        
        margin = 0.95
        
        # --- 2. ATUALIZAÇÃO: GOLPES DE 2 TURNOS E HERB ---
        charge_moves = ['fly', 'bounce', 'dig', 'dive', 'phantomforce', 'shadowforce', 'solarbeam', 'solarblade', 'skullbash', 'meteorbeam']
        recharge_moves = ['hyperbeam', 'gigaimpact', 'rockwrecker', 'roaroftime', 'frenzyplant', 'blastburn', 'hydrocannon']
        
        item_str = str(getattr(attacker, 'item', '')).lower()
        weather = next(iter(battle.weather)).name if battle and battle.weather else "CLEAR"
        known_opp_moves = [m.id for m in defender.moves.values()]
        
        if move.id in charge_moves:
            is_instant = False
            # Checa a Power Herb
            if item_str == 'powerherb':
                is_instant = True
            # Checa o Clima para Solar Beam/Blade
            elif move.id in ['solarbeam', 'solarblade'] and weather in ['SUNNYDAY', 'DESOLATELAND']:
                is_instant = True
                
            if not is_instant:
                # Debuff severo por dar um turno livre ao oponente para setup/troca
                margin *= 0.4
                
                # Fraquezas táticas extremas durante o turno de invulnerabilidade
                if move.id == 'dig' and 'earthquake' in known_opp_moves:
                    margin *= 0.1 # Punição extrema (Leva dobro de dano no subsolo)
                elif move.id in ['fly', 'bounce'] and any(m in known_opp_moves for m in ['thunder', 'hurricane']):
                    margin *= 0.1 # Punição extrema (Leva dano garantido no ar)
                    
        elif move.id in recharge_moves:
            # Golpes que travam o usuário APÓS o ataque sempre recebem penalidade tática pesada
            # (Power Herb não afeta recargas!)
            margin *= 0.45

        ignores_screens = move.id in ['brickbreak', 'psychicfangs'] or str(getattr(attacker, 'ability', '')).lower() == 'infiltrator'
        
        if battle and not ignores_screens:
            if defender in battle.team.values():
                side_to_check = battle.side_conditions
            else:
                side_to_check = battle.opponent_side_conditions
                
            active_screens = [str(k).upper() for k in side_to_check.keys()]
            
            if move.category.name == "PHYSICAL":
                if 'REFLECT' in active_screens or 'AURORA_VEIL' in active_screens:
                    margin *= 0.5 
            elif move.category.name == "SPECIAL":
                if 'LIGHT_SCREEN' in active_screens or 'AURORA_VEIL' in active_screens:
                    margin *= 0.5 
        
        final_dmg = base_dmg * stab * type_mod * margin
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

        if move.category.name in ["PHYSICAL", "SPECIAL"] and move.base_power > 0: 
            return MoveCategory.ATTACK_STRONG

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
        
        # --- 1. FILTRO DE DANO E IMUNIDADE ---
        if move.base_power > 0:
            if opponent.damage_multiplier(move) == 0:
                return True
            if 'wonderguard' in opp_abilities and opponent.damage_multiplier(move) < 2:
                if move.id != 'struggle': return True
                
        # Golpe Terrestre vs Air Balloon (Exceção: Thousand Arrows)
        if move.type and move.type.name == 'GROUND' and move.id != 'thousandarrows':
            if opponent.item and str(opponent.item).lower() == 'airballoon':
                return True

        # --- 2. LIMPEZA DE HAZARDS ---
        if move.id in ['defog', 'rapidspin', 'mortalspin', 'tidyup', 'courtchange']:
            my_side = battle.side_conditions
            opp_side = battle.opponent_side_conditions
            
            has_hazard = False
            target_hazards = ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']
            target_screens = ['REFLECT', 'LIGHT_SCREEN', 'AURORA_VEIL']
            
            for cond in my_side:
                if cond.name in target_hazards:
                    has_hazard = True
                    break
                    
            if not has_hazard and move.id in ['defog', 'courtchange']:
                for cond in opp_side:
                    if cond.name in target_screens or cond.name in target_hazards:
                        has_hazard = True
                        break
                        
            if not has_hazard:
                return True
        # --- 3. PRIORIDADE, PRIMEIRO TURNO E FLINCH ---
        move_priority = 0
        try:
            move_priority = move.priority
        except (KeyError, AttributeError):
            move_priority = 0

        if move.id in ['fakeout', 'firstimpression']:
            if not getattr(active, 'first_turn', False):
                return True

        if move_priority > 0:
            if any(ab in opp_abilities for ab in ['dazzling', 'queenlymajesty', 'armortail']):
                return True
            if 'psychicsurge' in opp_abilities or any('psychicterrain' in str(f).lower() for f in battle.fields.keys()):
                if 'FLYING' not in opp_types and not (opponent.item and str(opponent.item).lower() == 'airballoon') and 'levitate' not in opp_abilities:
                    return True
        
        # --- 4. FILTRO DE STATUS E IMUNIDADES ---
        if move.category.name == "STATUS":
            # Prankster vs Noturnos
            if active.ability == 'prankster' and 'DARK' in opp_types: return True
                
            # Powders vs Planta / Overcoat
            if move.id in ['spore', 'sleeppowder', 'stunspore', 'poisonpowder', 'ragepowder']:
                if 'GRASS' in opp_types or 'overcoat' in opp_abilities: return True
                    
            if move.id == 'thunderwave' and ('GROUND' in opp_types or 'ELECTRIC' in opp_types): return True
            if move.id == 'leechseed' and 'GRASS' in opp_types: return True

            # Refletores e Imunidades Totais (Magic Bounce / Good as Gold)
            if any(ab in opp_abilities for ab in ['magicbounce']) and getattr(move, 'target', '') in ['normal', 'allAdjacentFoes', 'foeSide']: return True
            if any(ab in opp_abilities for ab in ['goodasgold', 'magicguard']): return True 
            if move.id in ['confuseray', 'swagger'] and any(ab in opp_abilities for ab in ['owntempo', 'oblivious']): return True

            # Aplicação de Doenças
            if move.status:
                if opponent.status: return True # Já tem status
                
                # Prevenção de Suicídio via Synchronize
                if 'synchronize' in opp_abilities:
                    my_types = [t.name for t in active.types if t]
                    if move.status.name in ['TOX', 'PSN'] and 'POISON' not in my_types and 'STEEL' not in my_types: return True
                    if move.status.name == 'BRN' and 'FIRE' not in my_types: return True
                    if move.status.name == 'PRZ' and 'ELECTRIC' not in my_types and 'GROUND' not in my_types: return True

                if move.status.name in ['TOX', 'PSN']:
                    if 'immunity' in opp_abilities: return True
                    if 'POISON' in opp_types or 'STEEL' in opp_types:
                        if active.ability != 'corrosion': return True
                elif move.status.name == 'BRN':
                    if 'FIRE' in opp_types or any(ab in opp_abilities for ab in ['waterveil', 'waterbubble']): return True
                elif move.status.name == 'PRZ':
                    if 'ELECTRIC' in opp_types or 'limber' in opp_abilities: return True
                elif move.status.name == 'SLP':
                    if any(ab in opp_abilities for ab in ['insomnia', 'vitalspirit', 'sweetveil']): return True

        # --- 5. BUFFS MAXIMIZADOS E DEBUFFS ---
        if move.category.name == "STATUS" and move.boosts:
            # Pega o alvo padrão. Se for buff próprio ('self' ou equivalente)
            if getattr(move, 'target', 'self') == 'self':
                is_useful = False
                for stat, boost_amount in move.boosts.items():
                    current_stage = active.boosts.get(stat, 0)
                    if boost_amount > 0: 
                        # O limite hardcoded da mecânica do jogo é +6
                        if current_stage < 6:
                            is_useful = True
                            break 
                    elif boost_amount < 0: 
                        is_useful = True 
                        
                if not is_useful: 
                    return True
                    
            # Debuffs aplicados no oponente (ex: Screech)
            elif getattr(move, 'target', 'normal') in ['normal', 'allAdjacentFoes']:
                if any(ab in opp_abilities for ab in ['clearbody', 'whitesmoke', 'fullmetalbody']):
                    if any(b < 0 for b in move.boosts.values()):
                        return True

        # --- 6. BARREIRAS, CONDIÇÕES DE CAMPO E CLIMA ---
        current_weather = next(iter(battle.weather)).name if battle.weather else "CLEAR"
        
        if move.id in ['reflect', 'lightscreen', 'auroraveil', 'safeguard', 'tailwind']:
            my_side = [str(k).upper() for k in battle.side_conditions.keys()]
            if move.id == 'reflect' and 'REFLECT' in my_side: return True
            if move.id == 'lightscreen' and 'LIGHT_SCREEN' in my_side: return True
            if move.id == 'safeguard' and 'SAFEGUARD' in my_side: return True
            if move.id == 'tailwind' and 'TAILWIND' in my_side: return True
            if move.id == 'auroraveil':
                if 'AURORA_VEIL' in my_side: return True
                if current_weather not in ['HAIL', 'SNOW', 'SNOWSCAPE']: return True

        weather_moves = ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape']
        if move.id in weather_moves:
            if move.id == 'raindance' and current_weather in ['RAINDANCE', 'PRIMORDIALSEA']: return True
            if move.id == 'sunnyday' and current_weather in ['SUNNYDAY', 'DESOLATELAND']: return True
            if move.id == 'sandstorm' and current_weather == 'SANDSTORM': return True
            if move.id in ['hail', 'snowscape'] and current_weather in ['HAIL', 'SNOW', 'SNOWSCAPE']: return True

        # --- 7. PREVENÇÃO DE PROTECT CONSECUTIVO ---
        if move.id in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark', 'silktrap', 'obstruct', 'endure']:
            if history and 'last_action' in history:
                last_action_tuple = history.get('last_action')
                if last_action_tuple and "PROTECT" in last_action_tuple[0]:
                    return True

        # --- 8. CENÁRIO DE ÚLTIMO POKÉMON E PHAZING ---
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        
        if move.id in ['roar', 'whirlwind', 'dragontail', 'circlethrow']:
            if opp_alive <= 1: return True
            if 'suctioncups' in opp_abilities: return True 

        if move.id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb'] and opp_alive <= 1:
            return True

        # --- 9. PREDICT DE HABILIDADES (ABILITY IMMUNITY) ---
        # Bloqueia ataques diretos contra espécies que sabidamente possuem habilidades de absorção/imunidade no competitivo
        if move.category in ["Physical", "Special"]:
            opp_species = str(opponent.species).lower()
            
            # Water Absorb / Storm Drain / Dry Skin (Imunidade a Água)
            if move.type.name == "WATER":
                if opp_species in ['vaporeon', 'gastrodon', 'seismitoad', 'toxicroak', 'mantine', 'clodsire', 'volcanion']:
                    return True
                    
            # Flash Fire / Well-Baked Body (Imunidade a Fogo)
            elif move.type.name == "FIRE":
                if opp_species in ['heatran', 'chandelure', 'arcanine', 'ceruledge', 'houndoom', 'dachsbun']:
                    return True
                    
            # Volt Absorb / Motor Drive / Lightning Rod (Imunidade a Elétrico)
            elif move.type.name == "ELECTRIC":
                if opp_species in ['jolteon', 'thundurus', 'thundurustherian', 'zeraora', 'electivire', 'raichu', 'marowakalola']:
                    return True
                    
            # Sap Sipper (Imunidade a Planta)
            elif move.type.name == "GRASS":
                if opp_species in ['azumarill', 'goodra', 'bouffalant']:
                    return True
                    
            # Levitate / Earth Eater (Imunidade a Terra)
            elif move.type.name == "GROUND":
                if opp_species in ['rotom', 'rotomwash', 'rotomheat', 'rotommow', 'latios', 'latias', 'hydreigon', 'cresselia', 'weezing', 'orthworm']:
                    return True
            
        # --- 10. CURA DESNECESSÁRIA (OVERHEAL) ---
        # A. Bloqueia cura de HP se a vida estiver em 100% (ou muito próxima disso, >= 95%)
        healing_moves = [
            'recover', 'roost', 'slackoff', 'softboiled', 'milkdrink', 
            'shoreup', 'moonlight', 'morningsun', 'synthesis', 'healorder', 'wish'
        ]
        if move.id in healing_moves or (hasattr(move, 'heal') and move.heal and move.category == "Status"):
            if active.current_hp_fraction >= 0.95:
                return True
                
        # B. Bloqueia cura de Status da equipe se ninguém estiver com Status negativo
        if move.id in ['aromatherapy', 'healbell']:
            team_has_status = any(m.status is not None for m in battle.team.values())
            if not team_has_status:
                return True

        return False

    def get_macro_context(self, battle):
        """
        Funde o tempo de jogo e a contagem de peças (Vantagem) em 5 contextos vitais,
        ignorando a flutuação de HP para evitar ruído.
        """
        my_alive = len([m for m in battle.team.values() if not m.fainted])
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        total_alive = my_alive + opp_alive
        
        piece_advantage = my_alive - opp_alive
        
        # 1. EARLY GAME (12 a 10 vivos)
        # O jogo acabou de começar. Perder 1 Pokémon aqui não dita o jogo, é fase de mapeamento e Hazards.
        if total_alive >= 10:
            return "OPENING"
            
        # 2. VANTAGEM OU DESVANTAGEM NUMÉRICA CLARA
        # O peso de ter peças a mais/menos sobrepõe qualquer noção de "Early/Mid/Late".
        if piece_advantage >= 2:
            return "DOMINATING"  # Bot tem 2+ Pokémon de vantagem. Foco em pressionar e não dar turnos livres.
        elif piece_advantage <= -2:
            return "RECOVERING"  # Bot tem 2+ Pokémon a menos. Jogo desesperado, focar em predict e setups de risco.
            
        # 3. JOGO PARELHO (Diferença de 1 peça ou menos)
        if total_alive <= 5:
            return "CLUTCH"      # Final de jogo equilibrado (ex: 3v2, 2v2, 1v2). Foco total em Checkmate.
        else:
            return "BRAWL"       # Mid-game sangrento (ex: 4v4, 5v4). Foco em quebrar os tanks e controle de campo.
    
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

    def _is_active_best_remaining(self, active, opponent, battle):

        if not battle.available_switches:
            return True

        # 1. Calcula o score de sobrevivência do Pokémon que já está em campo
        # (is_active=True indica que ele não sofrerá dano de entrada de Hazards)
        active_score = self._get_survival_score(active, opponent, battle, is_active=True)

        # 2. Avalia quem é o melhor candidato entre os Pokémon no banco
        best_bench_score = -9999
        for bench_mon in battle.available_switches:
            bench_score = self._get_survival_score(bench_mon, opponent, battle, is_active=False)
            if bench_score > best_bench_score:
                best_bench_score = bench_score

        # 3. Decisão Tática:
        # Só consideramos que o ativo NÃO é a melhor opção se houver alguém 
        # no banco com uma segurança (score) consideravelmente maior (+50).
        # Isso evita trocas infinitas entre dois Pokémon ruins.
        if best_bench_score > active_score + 50:
            return False

        return True

    def _get_survival_score(self, candidate, opponent, battle, is_active=False):

        if not candidate: return -9999
        hp_frac = candidate.current_hp_fraction
        
        # Penalidade de entrada para quem está no banco (Hazards)
        if not is_active:
            hazard_dmg = self._get_hazard_damage(candidate, battle)
            if hp_frac <= hazard_dmg + 0.05:
                return -9999 # Morte certa na entrada
                
        score = 0.0
        if hp_frac >= 0.7: score += 150
        elif hp_frac >= 0.4: score += 50
        else: score -= 100
        
        if not opponent: return score

        opp_types_obj = [t for t in opponent.types if t]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]
        
        # Avaliação de fraquezas contra o oponente atual
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

        # Avaliação de Velocidade e Matchup
        cand_spe = self.estimate_stat(candidate, 'spe')
        opp_spe = self.estimate_stat(opponent, 'spe')
        
        if cand_spe > opp_spe:
            score += 100
            # Bônus se tiver um golpe super efetivo contra o oponente
            has_se_move = any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in candidate.moves.values())
            if has_se_move: score += 150
        else:
            if has_weakness: score -= 200

        matchup = self.get_matchup_state(candidate, opponent)
        if matchup == MatchupState.DOMINANT: score += 200
        elif matchup == MatchupState.DEFENSIVE_ADV: score += 100
        elif matchup == MatchupState.CRITICAL_DIS: score -= 300

        return score

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

    def _mod_escape(self, base, active, opponent, is_faster, my_role, battle):
        modified = base.copy()
        
        # Condição: Se a função acima disser que o banco não é seguro, nós ficamos para lutar!
        if self._is_active_best_remaining(active, opponent, battle):
            
            # Removemos a obrigatoriedade de troca do topo
            if "SWITCH_DEFENSIVE" in modified: modified.remove("SWITCH_DEFENSIVE")
            if "SWITCH_OFFENSIVE" in modified: modified.remove("SWITCH_OFFENSIVE")
            if "ATTACK_PIVOT" in modified: modified.remove("ATTACK_PIVOT")
            
            # 1. Se formos mais rápidos: prioridade máxima em causar dano antes de cair (Kamikaze)
            if is_faster:
                if "ATTACK_STRONG" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
                if "ATTACK_TECH" in modified: modified.insert(1, modified.pop(modified.index("ATTACK_TECH")))
            
            # 2. Se formos um Tank: tentamos inutilizar o oponente com status
            elif my_role == Role.TANK:
                opp_is_physical = self._is_physical(opponent)
                my_def = active.base_stats.get('def', 0)
                my_spd = active.base_stats.get('spd', 0)
                is_right_def = (opp_is_physical and my_def >= my_spd) or (not opp_is_physical and my_spd > my_def)
                
                # Só joga status se a defesa do tank estiver alinhada ao atacante inimigo
                if is_right_def and "STATUS" in modified:
                    modified.insert(0, modified.pop(modified.index("STATUS")))
                elif "ATTACK_STRONG" in modified:
                    modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
                    
            # 3. Fallback genérico: Bater o mais forte possível
            else:
                if "ATTACK_STRONG" in modified: modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
                
            # Recoloca as ações de fuga no fim da lista como último recurso absoluto
            modified.extend(["ATTACK_PIVOT", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"])
            
        return modified

    def _mod_lead(self, base, active, opponent, battle, is_faster):
        modified = base.copy()
        my_team = list(battle.team.values())
        
        # 1. Análise da dependência do time
        weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce', 'solarpower', 'hydration', 'drought', 'drizzle', 'sandstream', 'snowwarning']
        team_needs_weather = any(str(m.ability).lower() in weather_abusers for m in my_team)
        
        avg_speed = sum(m.base_stats.get('spe', 50) for m in my_team) / len(my_team)
        team_needs_tr = avg_speed < 70 # Time lento pede Trick Room (Field Control)
        
        needs_field_control = team_needs_weather or team_needs_tr
        weather_active = battle.weather is not None and len(battle.weather) > 0
        
        matchup = self.get_matchup_state(active, opponent)
        matchup_lost = matchup in [MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS]
        matchup_won = matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]

        # 2. Execução das Condicionais
        
        # Condição C: Perdemos o matchup
        if matchup_lost:
            if is_faster and "ATTACK_PIVOT" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
            elif not is_faster and "SWITCH_DEFENSIVE" in modified:
                modified.insert(0, modified.pop(modified.index("SWITCH_DEFENSIVE")))
            return modified

        # Condição A: Dependência de Clima/Trick Room (Prioridade 1)
        if needs_field_control and not weather_active:
            if "FIELD_CONTROL" in modified:
                modified.insert(0, modified.pop(modified.index("FIELD_CONTROL")))
                
        # Condição B: Clima Ativo + Vantagem de Matchup
        elif weather_active and matchup_won:
            if "HAZARD" in modified:
                modified.insert(0, modified.pop(modified.index("HAZARD")))
                
        # Condição Extra: Clima Ativo + Vantagem de Speed
        if weather_active and is_faster:
            if "ATTACK_STRONG" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))

        return modified

    def _mod_wallbreak(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac):
        modified = base.copy()
        
        # 1. Se o oponente tem capacidade de cura, priorizamos ataques técnicos (Taunt/Knock Off) e Status
        opponent_has_recovery = self._has_recovery(opponent)
        if opponent_has_recovery:
            if "STATUS" in modified:
                modified.insert(0, modified.pop(modified.index("STATUS")))
            if "ATTACK_TECH" in modified:
                modified.insert(1, modified.pop(modified.index("ATTACK_TECH")))
                
        # 2. Se temos HP seguro e o oponente é passivo, é a janela perfeita para Buff
        if "BUFF" in modified and my_hp_frac >= 0.60:
            # Coloca o BUFF na frente se não for prioridade de status
            insert_idx = 2 if opponent_has_recovery else 0
            modified.insert(insert_idx, modified.pop(modified.index("BUFF")))
            
        # 3. Se nosso HP está baixo, não tentamos quebrar a wall, fazemos pivot para outro pokemon
        if my_hp_frac < 0.40 and "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))

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

        macro_context = self.get_macro_context(battle)
        
        # CAMADA 1: Matchup -> TacticalMode
        if macro_context == "OPENING":
            mode = TacticalMode.LEAD
        else:
            mode = self._get_tactical_mode(matchup, my_role, opp_role, 
                                           is_faster, my_hp_frac, opp_hp_frac, is_threat, 
                                           active, opp)

        # CAMADA 2: Template base do modo
        base_priorities = self.mode_templates[mode].copy()

        # CAMADA 3: Modificador de Role ajusta a lista
        if mode == TacticalMode.LEAD:
            priorities = self._mod_lead(base_priorities, active, opp, battle, is_faster)
        elif mode == TacticalMode.WALLBREAK:
            priorities = self._mod_wallbreak(base_priorities, active, opp, is_faster, my_hp_frac, opp_hp_frac)
        elif mode == TacticalMode.ESCAPE:
            priorities = self._mod_escape(base_priorities, active, opp, is_faster, my_role, battle)
        else:
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

        # --- INFLUÊNCIA DE BARREIRAS (SCREENS) NA MACRO-ESTRATÉGIA ---
        opp_side_conds = [str(k).upper() for k in battle.opponent_side_conditions.keys()]
        
        physical_blocked = 'REFLECT' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds
        special_blocked = 'LIGHT_SCREEN' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds
        
        if physical_blocked or special_blocked:
            benched_mons = [m for m in battle.team.values() if not m.fainted and not m.active]
            can_bypass = False
            
            # Se apenas uma barreira está ativa, tentamos achar um Sweeper do tipo de ataque oposto no banco
            if physical_blocked and not special_blocked:
                # Procura um Sweeper Especial
                if any(self.get_role(m) == Role.SWEEPER and not self._is_physical(m) for m in benched_mons):
                    can_bypass = True
            elif special_blocked and not physical_blocked:
                # Procura um Sweeper Físico
                if any(self.get_role(m) == Role.SWEEPER and self._is_physical(m) for m in benched_mons):
                    can_bypass = True
            
            # O Clean Hazard (Defog/Court Change/Brick Break) quebra barreiras e deve sempre ser top tier aqui
            base_priorities = ["CLEAN_HAZARD"]
            
            if can_bypass:
                # TEMOS A RESPOSTA! A barreira dele é inútil contra o nosso banco.
                # Eleva a prioridade de trazer o atacante correto e amassar o oponente.
                boost_intents = base_priorities + ["SWITCH_OFFENSIVE", "ATTACK_PIVOT", "BUFF"]
            else:
                # NÃO TEMOS RESPOSTA (ou é Aurora Veil bloqueando ambos).
                # Entra em Modo Stall: Trocas defensivas, cura, status e perda de tempo até a barreira cair.
                boost_intents = base_priorities + ["SWITCH_DEFENSIVE", "STATUS", "HEAL", "PROTECT", "DEBUFF"]
            
            # Puxa as ações decididas para o topo da lista de ranqueamento, respeitando a ordem
            for b_intent in reversed(boost_intents):
                if b_intent in ranking_list:
                    ranking_list.remove(b_intent)
                    ranking_list.insert(0, b_intent)

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
            my_team = list(battle.team.values())
            opp_team = list(battle.opponent_team.values())
            
            if not opp_team: return "/team 123456"
            best_lead = None
            
            # Árvore 1: Guerra de Climas
            weather_setters = ['drought', 'drizzle', 'sandstream', 'snowwarning']
            my_weather_setter = next((m for m in my_team if str(m.ability) in weather_setters), None)
            opp_has_weather = any(str(m.ability) in weather_setters for m in opp_team)
            
            if my_weather_setter and opp_has_weather:
                best_lead = my_weather_setter
            
            # Árvore 2: Suicide/Dedicated Hazard Setter (Sash + Hazard ou Speed Alta)
            if not best_lead:
                for m in my_team:
                    has_hazard = any(move.id in ['stealthrock', 'spikes', 'stickyweb'] for move in m.moves.values())
                    fast_or_sash = str(m.item) == 'focussash' or m.base_stats.get('spe', 0) > 105
                    if has_hazard and fast_or_sash:
                        best_lead = m
                        break
                        
            # Árvore 3: Fast Pivot (Garantir momentum no turno 1)
            if not best_lead:
                pivots = [m for m in my_team if any(move.id in ['uturn', 'voltswitch', 'flipturn'] for move in m.moves.values())]
                if pivots:
                    best_lead = max(pivots, key=lambda m: m.base_stats.get('spe', 0))
                    
            # Árvore 4: Fallback Base (Baseado no arquétipo do nosso time)
            if not best_lead:
                avg_speed = sum(m.base_stats.get('spe', 50) for m in my_team) / len(my_team)
                if avg_speed > 85: # Time Hyper Offense
                    best_lead = max(my_team, key=lambda m: m.base_stats.get('spe', 50))
                else:              # Time Bulky/Stall
                    best_lead = max(my_team, key=lambda m: m.base_stats.get('hp', 50) + m.base_stats.get('def', 50) + m.base_stats.get('spd', 50))

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
                threat_types_str = [t.name for t in opp_types_obj]
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

        opp_spe = self.estimate_stat(opponent, 'spe')
        opp_types_obj = [t for t in opponent.types if t]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]
        
        def get_general_score(cand):
            score = 0.0
            cand_spe = self.estimate_stat(cand, 'spe')
            hp_frac = cand.current_hp_fraction
            
            # 1. Conservação de HP Base
            if hp_frac >= 0.7: score += 150
            elif hp_frac >= 0.4: score += 50
            else: score -= 100
            
            # 2. Resistências e Fraquezas (Avaliando a segurança da entrada)
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

            # 3. Pressão de Revenge Kill (Sou mais rápido e ameaço matar?)
            if cand_spe > opp_spe:
                score += 100
                has_se_move = any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in cand.moves.values())
                if has_se_move: score += 150
            else:
                if has_weakness: 
                    score -= 200
                    
            # 4. Impacto do Matchup Abstrato
            matchup = self.get_matchup_state(cand, opponent)
            if matchup == MatchupState.DOMINANT: score += 200
            elif matchup == MatchupState.DEFENSIVE_ADV: score += 100
            elif matchup == MatchupState.CRITICAL_DIS: score -= 300
            
            return score

        return max(candidates, key=get_general_score)
       
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
        # BLINDAGEM DE ESCOPO: BLOCO DE ATAQUE CORRIGIDO (V3 - COMPETITIVE MIND)
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
                    strong_move = None
                    max_strong_score = -9999
                    
                    opp_hp_frac = opponent.current_hp_fraction
                    opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
                    benched_opponents = [m for m in battle.opponent_team.values() if not m.fainted and not m.active]
                    
                    for m in valid_moves:
                        score = self.estimate_damage_percent(m, active, opponent, battle)
                        m_priority = getattr(m, 'priority', 0)
                        
                        # --- 1. MALÍCIA DE PRIORIDADE ---
                        if m_priority > 0 and score >= opp_hp_frac:
                            score += 5.0 
                            
                        # --- 2. INTELIGÊNCIA COMPLEXA DE RECUO (RECOIL) ---
                        has_recoil = m.id in ['bravebird', 'flareblitz', 'doubleedge', 'woodhammer', 'wildcharge']
                        if has_recoil:
                            if score >= opp_hp_frac: # Vai matar?
                                # Se NÃO for o último Pokémon do oponente, avaliamos o sacrifício
                                if opp_alive > 1:
                                    future_utility = False
                                    # Meu Pokémon é valioso contra o resto do time dele?
                                    for b_opp in benched_opponents:
                                        if self.get_matchup_state(active, b_opp) in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
                                            future_utility = True
                                            break
                                            
                                    if future_utility:
                                        # Meu Pokémon é importante. Tem outro golpe seguro que também mata agora?
                                        for other_m in valid_moves:
                                            if other_m.id != m.id and self.estimate_damage_percent(other_m, active, opponent, battle) >= opp_hp_frac:
                                                score -= 2.0 # Preserve-se para varrer o resto do time! Use o outro golpe.
                                                break
                                                
                        # --- 3. CONSCIÊNCIA DE DEBUFFS (SELF-DROPS) ---
                        self_drop = m.id in ['closecombat', 'superpower', 'dracometeor', 'leafstorm', 'overheat', 'makeitrain', 'fleurcannon']
                        if self_drop:
                            if score < opp_hp_frac:
                                score -= 0.3 # Punição leve: evite sujar seus stats se não for o golpe de misericórdia
                                
                            # Agravante Crítico: Se já está debufado (usou antes), a utilidade do golpe despenca
                            if m.category.name == "SPECIAL" and active.boosts.get('spa', 0) < 0:
                                score -= 1.5
                            elif m.category.name == "PHYSICAL" and active.boosts.get('atk', 0) < 0:
                                score -= 1.5
                            # Isso forçará o score lá para baixo, fazendo o Agente acatar os golpes de Pivot ou Switch!
                            
                        # --- 4. BÔNUS PARA MULTI-HIT ---
                        multi_hit = m.id in ['iciclespear', 'rockblast', 'bulletseed', 'tailslap', 'pinmissile', 'watershuriken']
                        if multi_hit:
                            score += 0.2 

                        if hasattr(m, 'secondary') and m.secondary: score += 0.05
                        
                        if score > max_strong_score:
                            max_strong_score = score
                            strong_move = m

                    if not strong_move:
                        strong_move = valid_moves[0]

                    if base_action == "ATTACK_STRONG":
                        return strong_move
                    
                    if base_action == "ATTACK_PREDICTIVE":
                        if benched_opponents:
                            predictive_candidates = [m for m in valid_moves if m.type != strong_move.type]
                            if predictive_candidates:
                                best_pred_move = None
                                max_pred_score = -9999
                                
                                for m in predictive_candidates:
                                    avg_bench_dmg = sum(self.estimate_damage_percent(m, active, bench_mon, battle) for bench_mon in benched_opponents) / len(benched_opponents)
                                    score = avg_bench_dmg
                                    
                                    if m.id in ['knockoff', 'scald', 'nuzzle', 'saltcure', 'uturn', 'voltswitch', 'flipturn']:
                                        score += 0.20 
                                    if hasattr(m, 'secondary') and m.secondary: score += 0.05
                                    
                                    if score > max_pred_score:
                                        max_pred_score = score
                                        best_pred_move = m
                                        
                                if best_pred_move:
                                    return best_pred_move
                        
                        return strong_move
        
        if battle.available_switches:
            return battle.available_switches[0]
        if battle.available_moves:
            return battle.available_moves[0]
            
        return None