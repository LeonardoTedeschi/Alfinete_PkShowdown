"""
Camada 2 — Leitura de Estado (StateParser).

Traduz um objeto `battle` do poke-env numa tupla de estado DISCRETA e de baixa
cardinalidade, que serve de chave para a Q-table. Esta é a peça partilhada por
TODOS os agentes de aprendizado (Green/Q-puro, Blue/Híbrido, Red/DQN), porque
todos precisam de perceber o estado do jogo da mesma forma para que a comparação
entre eles seja justa.

Depende da Camada 1 (GamePhysics), recebida por injeção no construtor, porque a
leitura de velocidade (get_speed_tier) e de papéis usa a física para estimar stats.
NÃO depende do masking nem da política — não sabe o que é uma "ação boa".

Contrato de saída (get_state): tupla de EXATAMENTE STATE_DIM elementos (strings).
A dimensão é fixa e garantida em todos os caminhos, incluindo o fallback, porque o
agente Red (DQN) vetoriza esta tupla e exige comprimento constante.

Índices da tupla de estado:
  0  my_role            8  my_status
  1  opp_role           9  opp_status
  2  matchup           10  my_boost
  3  my_hp             11  opp_boost
  4  opp_hp            12  my_hazards
  5  weather/field     13  opp_hazards
  6  speed_tier        14  macro_context
  7  mechanic
"""

from shared.definitions import MatchupState

STATE_DIM = 15  # dimensão fixa da tupla de estado (ver contrato acima)


class StateParser:
    """Componente de leitura de estado. Recebe a física por injeção."""

    def __init__(self, physics):
        # physics: instância de GamePhysics. Injeção explícita torna a dependência
        # visível e permite testar o parser com uma física falsa se necessário.
        self.physics = physics

    # -- Buckets de HP ------------------------------------------------------

    def get_hp_bucket(self, pokemon):
        if not pokemon or pokemon.fainted:
            return "CRIT"
        hp = pokemon.current_hp_fraction
        if hp >= 0.85:
            return "FULL"
        if hp >= 0.50:
            return "SAFE"
        if hp >= 0.25:
            return "DANGER"
        return "CRIT"

    # Aliás explícito: a lógica do oponente é idêntica à própria, mas mantemos
    # o método separado para permitir divergência futura (ex.: incerteza de HP).
    def get_opp_hp_bucket(self, pokemon):
        return self.get_hp_bucket(pokemon)

    # -- Clima / campo ------------------------------------------------------

    def get_weather_state(self, battle):
        active = battle.active_pokemon
        if not active:
            return "NORMAL"

        current_weather = next(iter(battle.weather)).name.upper() if battle.weather else "CLEAR"
        current_fields = [str(f).upper() for f in battle.fields.keys()]
        my_side = [str(k).upper() for k in battle.side_conditions.keys()]

        my_types = [t.name for t in active.types if t]
        my_ability = str(active.ability).lower() if active.ability else ""
        my_spe = active.base_stats.get('spe', 100)

        synergies = []

        # 1. POWER
        if current_weather in ["RAINDANCE", "PRIMORDIALSEA"] and "WATER" in my_types:
            synergies.append("POWER")
        elif current_weather in ["SUNNYDAY", "DESOLATELAND"] and "FIRE" in my_types:
            synergies.append("POWER")
        elif "ELECTRIC_TERRAIN" in current_fields and "ELECTRIC" in my_types:
            synergies.append("POWER")
        elif "GRASSY_TERRAIN" in current_fields and "GRASS" in my_types:
            synergies.append("POWER")
        elif "PSYCHIC_TERRAIN" in current_fields and "PSYCHIC" in my_types:
            synergies.append("POWER")
        elif "MISTY_TERRAIN" in current_fields and "FAIRY" in my_types:
            synergies.append("POWER")
        elif my_ability in ['sandforce', 'solarpower']:
            synergies.append("POWER")

        # 2. SPEED
        if "TAILWIND" in my_side:
            synergies.append("SPEED")
        elif current_weather in ["RAINDANCE", "PRIMORDIALSEA"] and my_ability == 'swiftswim':
            synergies.append("SPEED")
        elif current_weather in ["SUNNYDAY", "DESOLATELAND"] and my_ability == 'chlorophyll':
            synergies.append("SPEED")
        elif current_weather == "SANDSTORM" and my_ability == 'sandrush':
            synergies.append("SPEED")
        elif current_weather in ["HAIL", "SNOW", "SNOWSCAPE"] and my_ability == 'slushrush':
            synergies.append("SPEED")
        elif "ELECTRIC_TERRAIN" in current_fields and my_ability == 'surgesurfer':
            synergies.append("SPEED")
        elif "TRICK_ROOM" in current_fields and (my_spe <= 65):
            synergies.append("SPEED")

        # 3. DEFENSE / SUSTAIN
        if current_weather in ["RAINDANCE", "PRIMORDIALSEA"] and my_ability in ['raindish', 'dryskin', 'hydration']:
            synergies.append("DEFENSE")
        elif current_weather == "SANDSTORM" and ("ROCK" in my_types or my_ability in ['sandveil']):
            synergies.append("DEFENSE")
        elif current_weather in ["HAIL", "SNOW", "SNOWSCAPE"] and ("ICE" in my_types or my_ability in ['snowcloak', 'icebody']):
            synergies.append("DEFENSE")
        elif current_weather in ["SUNNYDAY", "DESOLATELAND"] and my_ability == 'leafguard':
            synergies.append("DEFENSE")

        # 4. HOSTILE
        hostile = False
        if current_weather in ["RAINDANCE", "PRIMORDIALSEA"] and "FIRE" in my_types:
            hostile = True
        elif current_weather in ["SUNNYDAY", "DESOLATELAND"] and ("WATER" in my_types or my_ability == 'dryskin'):
            hostile = True
        elif current_weather == "SANDSTORM" and not ("ROCK" in my_types or "GROUND" in my_types or "STEEL" in my_types or "magicguard" in my_ability or "overcoat" in my_ability):
            hostile = True
        elif current_weather == "HAIL" and not ("ICE" in my_types or "magicguard" in my_ability or "overcoat" in my_ability):
            hostile = True
        elif "TRICK_ROOM" in current_fields and my_spe >= 90:
            hostile = True

        if synergies:
            if "POWER" in synergies and "SPEED" in synergies:
                return "FIELD_SWEEP"
            if "POWER" in synergies:
                return "FIELD_POWER"
            if "SPEED" in synergies:
                return "FIELD_SPEED"
            if "DEFENSE" in synergies:
                return "FIELD_DEFENSE"
        elif hostile:
            return "FIELD_HOSTILE"
        elif current_weather not in ["CLEAR", "NONE"] or current_fields:
            return "FIELD_NEUTRAL"

        return "NORMAL"

    # -- Velocidade (usa a física) -----------------------------------------

    def get_speed_tier(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not active or not opponent:
            return "SLOWER"
        my_speed = self.physics.estimate_stat(active, 'spe')
        opp_speed = self.physics.estimate_stat(opponent, 'spe')
        if my_speed > opp_speed:
            return "FASTER"
        return "SLOWER"

    # -- Status -------------------------------------------------------------

    def get_status_state(self, pokemon):
        if not pokemon or pokemon.fainted:
            return "CLEAN"
        if pokemon.status:
            return "AFFLICTED"
        return "CLEAN"

    # -- Boosts (inclui status como nerf) ----------------------------------

    def get_boost_state(self, pokemon):
        if not pokemon or pokemon.fainted:
            return "NEUTRAL"
        state = "NEUTRAL"

        if pokemon.boosts:
            relevant_boosts = [v for k, v in pokemon.boosts.items() if k in ['atk', 'def', 'spa', 'spd', 'spe']]
            if any(v > 0 for v in relevant_boosts):
                state = "BUFFED"
            elif any(v < 0 for v in relevant_boosts):
                state = "DEBUFF"

        if pokemon.status:
            s_name = pokemon.status.name
            if s_name == 'BRN' and self.physics._is_physical(pokemon):
                state = "DEBUFF" if state == "NEUTRAL" else state + "_DEBUFF"
            elif s_name == 'PAR':
                state = "DEBUFF" if state == "NEUTRAL" else state + "_DEBUFF"

        return state

    # -- Hazards ------------------------------------------------------------

    def get_hazard_state(self, side_conditions):
        if not side_conditions:
            return "CLEAR"
        cond_strings = [str(k).upper() for k in side_conditions.keys()]
        hazards = ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']
        if any(h in cond for cond in cond_strings for h in hazards):
            return "SET"
        return "CLEAR"

    # -- Mecânica (tera/mega/z/dynamax disponível?) ------------------------

    def get_mechanic_state(self, battle):
        if battle.can_tera or battle.can_mega_evolve or battle.can_z_move or battle.can_dynamax:
            return "MEC_AVAIL"
        return "MEC_USED"

    # -- Matchup (tipos) ----------------------------------------------------

    def get_matchup_state(self, my_mon, opp_mon) -> MatchupState:
        if not my_mon or not opp_mon:
            return MatchupState.NEUTRAL

        my_moves = [m for m in my_mon.moves.values() if m.base_power > 0]
        if my_moves:
            my_best_mult = max([opp_mon.damage_multiplier(move) for move in my_moves])
        else:
            my_best_mult = 0.0

        opp_best_mult = 0.0
        for type_ in opp_mon.types:
            if type_:
                multiplier = my_mon.damage_multiplier(type_)
                if multiplier > opp_best_mult:
                    opp_best_mult = multiplier

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
            if opp_se:
                return MatchupState.VOLATILE
            if opp_neutral:
                return MatchupState.OFFENSIVE_ADV
            if opp_nve:
                return MatchupState.DOMINANT
        if my_neutral:
            if opp_se:
                return MatchupState.DEFENSIVE_DIS
            if opp_neutral:
                return MatchupState.NEUTRAL
            if opp_nve:
                return MatchupState.DEFENSIVE_ADV
        if my_nve:
            if opp_se:
                return MatchupState.CRITICAL_DIS
            if opp_neutral:
                return MatchupState.OFFENSIVE_DIS
            if opp_nve:
                return MatchupState.STALEMATE

        return MatchupState.NEUTRAL

    # -- Contexto macro (fase de jogo x vantagem de peças) -----------------

    def get_macro_context(self, battle):
        """Funde tempo de jogo e contagem de peças em 5 contextos, ignorando
        flutuação de HP para evitar ruído."""
        my_alive = len([m for m in battle.team.values() if not m.fainted])
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        total_alive = my_alive + opp_alive
        piece_advantage = my_alive - opp_alive

        if total_alive >= 10:
            return "OPENING"
        if piece_advantage >= 2:
            return "DOMINATING"
        elif piece_advantage <= -2:
            return "RECOVERING"
        if total_alive <= 5:
            return "CLUTCH"
        else:
            return "BRAWL"

    # -- Estado completo ----------------------------------------------------

    def get_state(self, battle):
        """Produz a tupla de estado de dimensão fixa STATE_DIM.

        CORREÇÃO face ao monólito original: o fallback agora devolve exatamente
        STATE_DIM elementos (o original devolvia 16, divergindo do caso normal de
        15). Dimensão constante é obrigatória para o encoding do agente Red (DQN).
        """
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active or not opponent:
            return ("UNKNOWN",) * STATE_DIM

        my_role = self.physics.get_role(active).name
        opp_role = self.physics.get_role(opponent).name
        matchup = self.get_matchup_state(active, opponent).name

        state = (
            my_role, opp_role, matchup,
            self.get_hp_bucket(active), self.get_opp_hp_bucket(opponent),
            self.get_weather_state(battle), self.get_speed_tier(battle),
            self.get_mechanic_state(battle),
            self.get_status_state(active), self.get_status_state(opponent),
            self.get_boost_state(active), self.get_boost_state(opponent),
            self.get_hazard_state(battle.side_conditions),
            self.get_hazard_state(battle.opponent_side_conditions),
            self.get_macro_context(battle),
        )

        # Garantia de contrato: a tupla tem sempre STATE_DIM elementos.
        assert len(state) == STATE_DIM, f"StateParser produziu {len(state)} dims, esperado {STATE_DIM}"
        return state
