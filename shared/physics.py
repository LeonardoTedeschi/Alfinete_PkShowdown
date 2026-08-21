"""
Camada 1 — Física do Jogo (GamePhysics).

Motor determinístico de física competitiva de Pokémon. Dado um Pokémon e/ou um
golpe, responde a perguntas de baixo nível: qual o papel tático, qual o valor
estimado de um atributo, quanto dano um golpe causa, e a que categoria funcional
um golpe pertence.

Este componente é a base da hierarquia do domínio: NÃO depende de nenhum outro
componente do projeto (apenas de `definitions`). Todos os componentes acima
(StateParser, ActionMasker, InstinctPolicy) recebem uma instância desta classe por
injeção. Não conhece o conceito de "estado da Q-table" nem de "instinto tático" —
é puramente o modelo de regras do jogo.

Reutilização na pesquisa:
- Usado por TODOS os agentes que precisam raciocinar sobre dano/stats.
- Isolá-lo permite testar a precisão da física independentemente da política.
"""

from shared.definitions import Role, MoveCategory


class GamePhysics:
    """Motor de física do jogo. Sem estado interno; métodos são funções puras sobre
    os objetos `pokemon`/`move`/`battle` do poke-env."""

    # -- Papel tático -------------------------------------------------------

    def get_role(self, pokemon) -> Role:
        """Classifica o papel do Pokémon a partir dos base stats."""
        if not pokemon:
            return Role.UTILITY
        b_atk = pokemon.base_stats.get('atk', 0)
        b_spa = pokemon.base_stats.get('spa', 0)
        b_hp = pokemon.base_stats.get('hp', 0)
        b_def = pokemon.base_stats.get('def', 0)
        b_spd = pokemon.base_stats.get('spd', 0)
        b_spe = pokemon.base_stats.get('spe', 0)

        # CLASSIFICACAO POR PONTUACAO COMPARATIVA (substitui a cascata de limiares).
        #
        # A versao anterior era: se atk>=100 ou spa>=100 -> SWEEPER, senao se
        # hp>=80 e (def>=100 ou spd>=100) -> TANK, senao UTILITY. Tinha dois defeitos:
        #   1. O primeiro IF capturava tudo antes de olhar para a defesa, logo
        #      Tyranitar, Heatran e Slowbro (ofensivos E defensivos) eram sempre
        #      SWEEPER e o instinto tratava-os como fragis.
        #   2. Limiares rigidos deixavam de fora paredes classicas: Skarmory (def 140
        #      mas hp 65), Toxapex (def 152/spd 142 mas hp 50), Ferrothorn e Gliscor
        #      caiam todos em UTILITY.
        # Em 14 Pokemon do meta, 6 ficavam mal classificados.
        #
        # Agora comparam-se DUAS pontuacoes e vence a maior, com a velocidade a
        # desempatar (um ofensivo lento e mais wallbreaker que sweeper, mas continua
        # a ser tratado como ofensivo).
        ofensiva = max(b_atk, b_spa) + 0.35 * b_spe
        # A defesa efetiva depende do HP: uma defesa alta com pouco HP vale menos.
        defensiva = (b_def + b_spd) * 0.5 + 0.9 * b_hp

        if ofensiva >= defensiva * 1.05:
            return Role.SWEEPER
        if defensiva >= ofensiva * 1.05:
            return Role.TANK
        # Zona de indiferenca (<5% de diferenca): nem claramente ofensivo nem
        # claramente defensivo -> papel de apoio.
        return Role.UTILITY

    # -- Modificador de velocidade -----------------------------------------

    def _get_speed_mod(self, pokemon):
        """Modificador de velocidade por paralisia e boosts de estágio."""
        mod = 1.0
        if pokemon.status and 'PAR' in str(pokemon.status).upper():
            mod *= 0.5
        stage = pokemon.boosts.get('spe', 0)
        if stage > 0:
            mod *= (1 + 0.5 * stage)
        elif stage < 0:
            mod *= (2 / (2 + abs(stage)))
        return mod

    # -- Estimativa de atributo --------------------------------------------

    def estimate_stat(self, pokemon, stat_name):
        """Estima o valor efetivo de um atributo, incluindo boosts, status e itens.
        Usa stats reais quando disponíveis; caso contrário infere a partir do base
        stat e do papel presumido (investimento típico competitivo)."""
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
                    if stat_name == 'spe':
                        val = calc_boosted
                    elif stat_name == 'atk' and pokemon.base_stats.get('atk', 0) >= pokemon.base_stats.get('spa', 0):
                        val = calc_invested
                    elif stat_name == 'spa' and pokemon.base_stats.get('spa', 0) > pokemon.base_stats.get('atk', 0):
                        val = calc_invested
                    else:
                        val = calc_uninvested
                elif role == Role.TANK:
                    base_def = pokemon.base_stats.get('def', 0)
                    base_spd = pokemon.base_stats.get('spd', 0)
                    best_def = 'def' if base_def >= base_spd else 'spd'
                    if stat_name == best_def:
                        val = calc_boosted
                    elif stat_name in ['def', 'spd']:
                        val = calc_invested
                    else:
                        val = calc_uninvested
                else:
                    if stat_name == 'spe':
                        val = calc_boosted
                    elif stat_name in ['def', 'spd']:
                        val = calc_invested
                    else:
                        val = calc_uninvested

        if stat_name == 'spe':
            val *= self._get_speed_mod(pokemon)
        else:
            modifier = pokemon.boosts.get(stat_name, 0)
            if modifier > 0:
                val *= (1 + 0.5 * modifier)
            elif modifier < 0:
                val *= (2 / (2 + abs(modifier)))

        item_str = str(pokemon.item).lower() if pokemon.item else ""
        item_mod = 1.0

        if stat_name == 'spe' and item_str == 'choicescarf':
            item_mod = 1.5
        elif stat_name == 'atk' and item_str == 'choiceband':
            item_mod = 1.5
        elif stat_name == 'spa' and item_str == 'choicespecs':
            item_mod = 1.5
        elif stat_name == 'spd' and item_str in ['assaultvest', 'eviolite']:
            item_mod = 1.5

        return int(val * item_mod)

    def _is_physical(self, pokemon):
        """True se o Pokémon é predominantemente físico (atk > spa estimado)."""
        if not pokemon:
            return True
        return self.estimate_stat(pokemon, 'atk') > self.estimate_stat(pokemon, 'spa')

    # -- Estimativa de dano -------------------------------------------------

    def estimate_damage_percent(self, move, attacker, defender, battle=None):
        """Estima o dano de um golpe como fração do HP máximo do defensor.
        Considera STAB, tipos, itens, habilidades, clima, terreno, barreiras e
        golpes de carga. Retorna 0.0 para golpes de status."""
        if move.category.name == "STATUS" or move.base_power == 0:
            return 0.0

        bp = float(move.base_power)
        level = float(getattr(attacker, 'level', 100))
        attacker_ability = str(getattr(attacker, 'ability', '')).lower()
        item_str = str(getattr(attacker, 'item', '')).lower()

        # 0. TECHNICIAN
        if attacker_ability == 'technician' and bp <= 60:
            bp *= 1.5

        # 1. MULTI-HIT
        multi_hit_moves = ['iciclespear', 'rockblast', 'bulletseed', 'tailslap', 'pinmissile', 'boneclub', 'scaleshot', 'watershuriken', 'dualwingbeat', 'bonemerang']
        if move.id in multi_hit_moves:
            if attacker_ability == 'skilllink':
                bp *= 5.0
            elif move.id in ['dualwingbeat', 'bonemerang']:
                bp *= 2.0
            else:
                bp *= 3.0

        # 2. MODIFICADORES DE BASE POWER (habilidades de categoria)
        move_flags = getattr(move, 'flags', {})
        if attacker_ability == 'ironfist' and 'punch' in move_flags:
            bp *= 1.2
        elif attacker_ability == 'strongjaw' and 'bite' in move_flags:
            bp *= 1.5
        elif attacker_ability == 'sharpness' and 'slicing' in move_flags:
            bp *= 1.5
        elif attacker_ability == 'toughclaws' and 'contact' in move_flags:
            bp *= 1.3
        elif attacker_ability == 'megalauncher' and 'pulse' in move_flags:
            bp *= 1.5
        elif attacker_ability == 'sheerforce' and getattr(move, 'secondary', False):
            bp *= 1.3
        elif attacker_ability == 'waterbubble' and move.type and move.type.name == 'WATER':
            bp *= 2.0
        elif attacker_ability == 'transistor' and move.type and move.type.name == 'ELECTRIC':
            bp *= 1.3
        elif attacker_ability == 'dragonsmaw' and move.type and move.type.name == 'DRAGON':
            bp *= 1.5

        # 3. ATRIBUTO OFENSIVO E DEFENSIVO + modificadores de status
        if move.category.name == "PHYSICAL":
            atk = self.estimate_stat(attacker, 'atk')
            if item_str == 'choiceband':
                atk *= 1.5
            if attacker_ability in ['hugepower', 'purepower']:
                atk *= 2.0
            if attacker_ability == 'hustle':
                atk *= 1.5
            if attacker_ability == 'guts' and attacker.status:
                atk *= 1.5
            if move.id == 'bodypress':
                atk = self.estimate_stat(attacker, 'def')
            defense = self.estimate_stat(defender, 'def')
        else:
            atk = self.estimate_stat(attacker, 'spa')
            if item_str == 'choicespecs':
                atk *= 1.5
            defense = self.estimate_stat(defender, 'spd')
            if move.id in ['psyshock', 'psystrike', 'secretsword']:
                defense = self.estimate_stat(defender, 'def')

        if defense <= 0:
            defense = 1

        base_dmg = ((((2 * level / 5) + 2) * atk * bp / defense) / 50) + 2

        # 4. STAB, tipo, item de dano final
        stab_multiplier = 2.0 if attacker_ability == 'adaptability' else 1.5
        stab = stab_multiplier if move.type in attacker.types else 1.0
        type_mod = defender.damage_multiplier(move)
        if attacker_ability == 'tintedlens' and type_mod < 1.0:
            type_mod *= 2.0

        item_mod = 1.0
        if item_str == 'lifeorb':
            item_mod = 1.3
        elif item_str == 'expertbelt' and type_mod > 1.0:
            item_mod = 1.2
        elif item_str == 'muscleband' and move.category.name == "PHYSICAL":
            item_mod = 1.1
        elif item_str == 'wiseglasses' and move.category.name == "SPECIAL":
            item_mod = 1.1

        margin = 0.95

        # 5. GOLPES DE 2 TURNOS E HERB
        charge_moves = ['fly', 'bounce', 'dig', 'dive', 'phantomforce', 'shadowforce', 'solarbeam', 'solarblade', 'skullbash', 'meteorbeam']
        recharge_moves = ['hyperbeam', 'gigaimpact', 'rockwrecker', 'roaroftime', 'frenzyplant', 'blastburn', 'hydrocannon']
        weather = next(iter(battle.weather)).name.upper() if battle and battle.weather else "CLEAR"
        known_opp_moves = [m.id for m in defender.moves.values()]

        if move.id in charge_moves:
            is_instant = False
            if item_str == 'powerherb':
                is_instant = True
            elif move.id in ['solarbeam', 'solarblade'] and weather in ['SUNNYDAY', 'DESOLATELAND']:
                is_instant = True
            if not is_instant:
                margin *= 0.4
                if move.id == 'dig' and 'earthquake' in known_opp_moves:
                    margin *= 0.1
                elif move.id in ['fly', 'bounce'] and any(m in known_opp_moves for m in ['thunder', 'hurricane']):
                    margin *= 0.1
        elif move.id in recharge_moves:
            margin *= 0.45

        # 6. BARREIRAS (SCREENS)
        ignores_screens = move.id in ['brickbreak', 'psychicfangs'] or attacker_ability == 'infiltrator'
        if battle and not ignores_screens:
            side_to_check = battle.side_conditions if defender in battle.team.values() else battle.opponent_side_conditions
            active_screens = [str(k).upper() for k in side_to_check.keys()]
            if move.category.name == "PHYSICAL" and ('REFLECT' in active_screens or 'AURORA_VEIL' in active_screens):
                margin *= 0.5
            elif move.category.name == "SPECIAL" and ('LIGHT_SCREEN' in active_screens or 'AURORA_VEIL' in active_screens):
                margin *= 0.5

        # 7. CLIMA E TERRENO
        weather_mod = 1.0
        terrain_mod = 1.0
        if battle:
            move_type = move.type.name if move.type else ""
            current_fields = [str(f).upper() for f in battle.fields.keys()]

            if weather in ["RAINDANCE", "PRIMORDIALSEA"]:
                if move_type == "WATER":
                    weather_mod = 1.5
                elif move_type == "FIRE":
                    weather_mod = 0.5
            elif weather in ["SUNNYDAY", "DESOLATELAND"]:
                if move_type == "FIRE":
                    weather_mod = 1.5
                elif move_type == "WATER":
                    weather_mod = 0.5
            elif weather == "SANDSTORM" and attacker_ability == 'sandforce' and move_type in ['ROCK', 'GROUND', 'STEEL']:
                weather_mod = 1.3

            def is_grounded(pokemon):
                if "FLYING" in [t.name for t in pokemon.types if t]:
                    return False
                if str(getattr(pokemon, 'ability', '')).lower() == "levitate":
                    return False
                if str(getattr(pokemon, 'item', '')).lower() == "airballoon":
                    return False
                return True

            attacker_grounded = is_grounded(attacker)
            defender_grounded = is_grounded(defender)

            if "ELECTRIC_TERRAIN" in current_fields and move_type == "ELECTRIC" and attacker_grounded:
                terrain_mod = 1.3
            elif "GRASSY_TERRAIN" in current_fields:
                if move_type == "GRASS" and attacker_grounded:
                    terrain_mod = 1.3
                if move.id in ["earthquake", "bulldoze", "magnitude"] and defender_grounded:
                    terrain_mod = 0.5
            elif "PSYCHIC_TERRAIN" in current_fields and move_type == "PSYCHIC" and attacker_grounded:
                terrain_mod = 1.3
            elif "MISTY_TERRAIN" in current_fields and move_type == "DRAGON" and defender_grounded:
                terrain_mod = 0.5

        final_dmg = base_dmg * stab * type_mod * item_mod * margin * weather_mod * terrain_mod
        max_hp = max(1, self.estimate_stat(defender, 'hp'))
        return final_dmg / max_hp

    # -- Classificação de golpes -------------------------------------------

    def classify_move(self, move) -> MoveCategory:
        """Mapeia um golpe para a sua categoria funcional (usada pelo masking e
        pela política)."""
        move_id = move.id

        if move_id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']:
            return MoveCategory.ATTACK_PIVOT

        tech_moves = [
            'knockoff', 'foulplay', 'thief', 'nuzzle', 'scald', 'discharge', 'lavaplume', 'saltcure',
            'superfang', 'naturesmadness', 'ruination', 'seismictoss', 'nightshade', 'icywind', 'electroweb',
            'rocktomb', 'bulldoze', 'snarl', 'mysticalfire', 'strugglebug', 'fakeout', 'brickbreak',
            'psychicfangs', 'bodypress'
        ]
        if move_id in tech_moves:
            return MoveCategory.ATTACK_TECH

        if move_id in ['haze', 'clearsmog']:
            return MoveCategory.STAT_CLEAN
        if move_id in ['aromatherapy', 'healbell', 'junglehealing']:
            return MoveCategory.HEAL_STATUS
        if move_id in ['roar', 'whirlwind', 'dragontail', 'circlethrow']:
            return MoveCategory.PHAZE
        if move_id in ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape', 'trickroom', 'tailwind', 'electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain']:
            return MoveCategory.FIELD_CONTROL
        if move_id in ['defog', 'rapidspin', 'mortalspin', 'courtchange']:
            return MoveCategory.CLEAN_HAZARD
        if move_id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']:
            return MoveCategory.HAZARD
        if move_id in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark', 'silktrap', 'obstruct', 'endure']:
            return MoveCategory.PROTECT
        if move_id in ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun', 'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap']:
            return MoveCategory.HEAL
        if move_id in ['reflect', 'lightscreen', 'auroraveil']:
            return MoveCategory.BARRIER
        if move.id in ['taunt', 'torment', 'encore', 'disable']:
            return MoveCategory.DISRUPTION

        if move.category.name == "STATUS":
            if getattr(move, 'heal', 0):
                return MoveCategory.HEAL
            if getattr(move, 'status', None):
                return MoveCategory.STATUS
            if getattr(move, 'boosts', None):
                if any(v > 0 for v in move.boosts.values()):
                    return MoveCategory.BUFF
                if any(v < 0 for v in move.boosts.values()):
                    return MoveCategory.DEBUFF
            return MoveCategory.STATUS

        if move.category.name in ["PHYSICAL", "SPECIAL"] and move.base_power > 0:
            return MoveCategory.ATTACK_STRONG

        return MoveCategory.UNKNOWN
