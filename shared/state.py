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
  7  mechanic          15  bench            <-- ACRESCENTADA 30/08/2026

A DIMENSAO 15 FOI ACRESCENTADA NO FIM, DE PROPOSITO. O `brain._calculate_potential`
le a tupla por INDICE FIXO (state[0], state[2], state[5]...). Inserir a meio deslocava
todos e reproduzia o bug de indices deslocados que a propria funcao ja documenta ter
tido duas vezes. Acrescentar no fim deixa os catorze indices existentes intactos.
"""

from shared.definitions import MatchupState

STATE_DIM = 16  # dimensão fixa da tupla de estado (ver contrato acima)


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
        # CORRIGIDO 26/08/2026 — BUG SILENCIOSO DE 5 TESTES.
        #
        # `str(Field.ELECTRIC_TERRAIN).upper()` devolve "FIELD.ELECTRIC_TERRAIN", nao
        # "ELECTRIC_TERRAIN". Como os testes abaixo usam PERTENCA A LISTA
        # (`"ELECTRIC_TERRAIN" in current_fields`) e nao subcadeia, davam SEMPRE
        # False. Ficavam permanentemente invisiveis ao agente:
        #
        #     Electric / Grassy / Psychic / Misty Terrain   (FIELD_POWER)
        #     Tailwind                                      (FIELD_SPEED)
        #     Surge Surfer com terreno eletrico             (FIELD_SPEED)
        #     Trick Room, dos dois lados                    (FIELD_SPEED / HOSTILE)
        #
        # Consequencia medida nos cerebros v7/v4: FIELD_SPEED com 0,8% das visitas.
        # O pool de treino TEM um time de Trick Room, e o agente nunca soube que
        # estava em Trick Room — jogou essas batalhas com a leitura de velocidade
        # invertida e sem sinal nenhum no estado.
        #
        # O `get_hazard_state` (mais abaixo) ja fazia o teste correto, por subcadeia.
        # Eram dois estilos diferentes para o mesmo problema no mesmo ficheiro.
        #
        # `.split(".")[-1]` fica com o nome do membro do enum, que e o que os testes
        # comparam.
        # MIGRADO PARA `physics.nomes_de` EM 03/09/2026. A forma antiga devolvia
        # 'ELECTRIC_TERRAIN (FIELD) OBJECT', logo TODAS as sinergias de terreno e
        # o TAILWIND abaixo eram sempre falsos. A dimensao de clima da tupla de
        # estado nunca viu terreno nenhum. Ver `nomes_de` em `shared/physics.py`.
        current_fields = self.physics.nomes_de(battle.fields)
        my_side = self.physics.nomes_de(battle.side_conditions)

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
        elif "TRICK_ROOM" in current_fields and self.physics.mais_rapido(
                active, battle.opponent_active_pokemon, battle):
            # COMPARACAO REAL, NAO LIMIAR (30/08/2026). Antes era `my_spe <= 65`
            # sobre o atributo BASE, com um espelho em `>= 90` no bloco hostile e uma
            # ZONA MORTA entre 66 e 89 onde o Trick Room nao era nem bom nem mau.
            #
            # O limiar responde "sou lento em abstrato". A pergunta que decide e "ajo
            # antes DESTE adversario NESTE turno", e o `mais_rapido` responde essa,
            # ja com boosts, paralisia, Choice Scarf, Tailwind e a propria inversao do
            # Trick Room. Um Torkoal de 20 de velocidade contra um Ferrothorn de 20
            # nao ganha nada com Trick Room; o limiar dizia que ganhava.
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
        # IMUNIDADE AO DANO DE CLIMA (30/08/2026): delegada a fisica.
        # A lista aqui estava incompleta. Faltavam Sand Rush, Sand Force, Sand
        # Veil, Ice Body, Snow Cloak e as Safety Goggles — um Excadrill com Sand
        # Rush era lido como se apanhasse da areia da propria equipa. Delegar
        # evita que a lista volte a divergir da do calculo de dano.
        # `battle` e nao `current_weather`: a fisica le o clima do proprio
        # `battle`. Passar a string fazia o `except` interno devolver False
        # SEMPRE, e o `hostile` ficava mudo para areia e granizo sem dar erro.
        elif current_weather in ("SANDSTORM", "HAIL") and \
                self.physics.sofre_dano_de_clima(active, battle):
            hostile = True
        elif "TRICK_ROOM" in current_fields and not self.physics.mais_rapido(
                active, battle.opponent_active_pokemon, battle):
            # Espelho exato do bloco de sinergia: sob Trick Room, se NAO agimos
            # primeiro, o campo esta a favorecer o outro lado. Sem zona morta.
            hostile = True

        # ==============================================================
        # 5. O LADO DELES (30/08/2026)
        # ==============================================================
        # ATE AQUI ESTA FUNCAO SO OLHAVA PARA `battle.active_pokemon`: os nossos
        # tipos, a nossa habilidade, a nossa velocidade. `FIELD_HOSTILE` so disparava
        # quando o campo nos magoava DIRETAMENTE.
        #
        # O buraco: chuva com um Kingdra do lado de la e nada nosso para aproveitar
        # caia em `FIELD_NEUTRAL`, quando e um dos piores estados possiveis. O
        # adversario ser favorecido E sermos desfavorecidos.
        #
        # A LEITURA E POR TIPOS, NAO POR HABILIDADES, e isso e deliberado. A
        # habilidade do adversario raramente esta revelada; inferir dai repetiria o
        # defeito do `opp_has_weather` (6.31 item 13), que lia `m.ability` do
        # adversario e dava quase sempre falso. Uma dimensao que quase nunca sai do
        # valor por omissao custa estados e nao discrimina nada. Tipos sao visiveis
        # desde o primeiro turno. A habilidade so entra se JA estiver confirmada.
        #
        # Cobertura parcial e assumida: um Barraskewda com Swift Swim por revelar nao
        # conta. E menos do que gostariamos e mais do que tinhamos.
        deles = False
        opponent = battle.opponent_active_pokemon
        if opponent:
            opp_types = [t.name for t in opponent.types if t]
            opp_ability = str(opponent.ability).lower() if opponent.ability else ""
            opp_spe = opponent.base_stats.get('spe', 100)
            if current_weather in ["RAINDANCE", "PRIMORDIALSEA"]:
                deles = "WATER" in opp_types or opp_ability in ('swiftswim', 'raindish', 'dryskin')
            elif current_weather in ["SUNNYDAY", "DESOLATELAND"]:
                deles = "FIRE" in opp_types or opp_ability in ('chlorophyll', 'solarpower', 'leafguard')
            elif current_weather == "SANDSTORM":
                deles = "ROCK" in opp_types or opp_ability in ('sandrush', 'sandforce', 'sandveil')
            elif current_weather in ["HAIL", "SNOW", "SNOWSCAPE"]:
                deles = "ICE" in opp_types or opp_ability in ('slushrush', 'icebody', 'snowcloak')
            if not deles and current_fields:
                if "ELECTRIC_TERRAIN" in current_fields:
                    deles = "ELECTRIC" in opp_types or opp_ability == 'surgesurfer'
                elif "GRASSY_TERRAIN" in current_fields:
                    deles = "GRASS" in opp_types
                elif "PSYCHIC_TERRAIN" in current_fields:
                    deles = "PSYCHIC" in opp_types
                elif "MISTY_TERRAIN" in current_fields:
                    deles = "FAIRY" in opp_types
                if "TRICK_ROOM" in current_fields and not self.physics.mais_rapido(
                        active, opponent, battle):
                    deles = True

        # ==============================================================
        # BALDES: de 7 para 5 (30/08/2026)
        # ==============================================================
        # Antes: FIELD_SWEEP, FIELD_POWER, FIELD_SPEED, FIELD_DEFENSE, FIELD_HOSTILE,
        #        FIELD_NEUTRAL, NORMAL. (7)
        # Agora: NORMAL, FIELD_SWEEP, FIELD_OURS_OFF, FIELD_OURS_DEF, FIELD_THEIRS,
        #        FIELD_SHARED. (6)
        #
        # REVISTO PARA 6 EM 30/08/2026, depois de medir a distribuicao real de visitas
        # nos cerebros de 400k. A versao de 5 baldes fundia ofensivo com defensivo, e
        # essa distincao pede jogadas OPOSTAS. Cobertura ponderada projectada: 81,0%
        # com 6 baldes, contra 82,8% com 5 e 79,4% com os 7 antigos. Custa 1,8 pontos
        # face aos 5 e devolve a leitura que mais decide.
        #
        # O QUE SE GANHA: `FIELD_THEIRS` passa a existir. A vantagem do adversario era
        # invisivel ao cerebro, e e metade da informacao que o campo carrega.
        #
        # O QUE SE PERDE, declarado: POWER, SPEED e DEFENSE fundem-se em `FIELD_OURS`.
        # A distincao entre "o campo da-me dano" e "o campo da-me resistencia" deixa
        # de estar no estado. Aceita-se porque a dimensao 6 (speed_tier), corrigida
        # em 30/08, ja carrega o efeito de velocidade de forma diretamente acionavel,
        # que era a parte mais decisiva das tres.
        #
        # FIELD_SWEEP fica separado por ser o unico que decide partidas sozinho.
        nosso = bool(synergies)
        sweep = "POWER" in synergies and "SPEED" in synergies

        if sweep and not deles:
            return "FIELD_SWEEP"
        if nosso and not deles:
            # OFENSIVO (POWER ou SPEED) separado de DEFENSIVO. Sao respostas opostas:
            # com vantagem ofensiva pressiona-se, com vantagem defensiva aguenta-se.
            # SPEED sozinho cai no ofensivo, e nao num balde proprio, porque a
            # dimensao 6 (`speed_tier`) ja carrega o efeito de velocidade de forma
            # directamente accionavel desde a correccao de 30/08. Um balde
            # FIELD_SPEED duplicaria essa informacao e custaria 1/6 do espaco.
            if "POWER" in synergies or "SPEED" in synergies:
                return "FIELD_OURS_OFF"
            return "FIELD_OURS_DEF"
        if (deles or hostile) and not nosso:
            return "FIELD_THEIRS"
        if nosso and deles:
            return "FIELD_SHARED"          # espelho: chuva com abusadores dos dois lados
        if current_weather not in ["CLEAR", "NONE"] or current_fields:
            return "FIELD_SHARED"          # existe e nao decide nada

        return "NORMAL"

    # -- Banco (dimensão 15, acrescentada 30/08/2026) -----------------------

    def get_bench_state(self, battle):
        """O banco oferece alguma coisa? BANCO_VAZIO / DEF / OFF / AMBOS.

        PORQUE ESTA DIMENSAO EXISTE. O espaco de accoes tem SWITCH_DEFENSIVE e
        SWITCH_OFFENSIVE, e ate aqui o estado nao dizia NADA sobre o banco. A mesma
        tupla de estado tanto podia ter um contra-tanque saudavel a espera como cinco
        Pokemon mortos, e o cerebro apostava as cegas nas duas. Era a unica accao do
        espaco cujo valor dependia de informacao que o estado nao continha.

        DUAS PERGUNTAS INDEPENDENTES, DOIS BITS. Nao se colapsa num ordinal de tres
        valores porque `SWITCH_DEFENSIVE` e `SWITCH_OFFENSIVE` perguntam coisas
        diferentes: "ha quem aguente" e "ha quem bata". Ter as duas nao e o mesmo que
        ter a melhor das duas.

        A FONTE E `battle.available_switches`, e nao a equipa viva. E a leitura
        decisoria correta: se estamos presos (Shadow Tag, Arena Trap, Ingrain), nao ha
        troca nenhuma disponivel e o estado deve dizer isso. Confundir "preso" com
        "banco vazio" e aceitavel porque a consequencia para a decisao e a mesma.

        DEF  o candidato RESISTE ao melhor golpe conhecido do adversario (<1x)
        OFF  o candidato tem golpe SUPER EFETIVO contra o adversario (>1x)

        Usa-se o mesmo criterio do `get_matchup_state` de proposito: dois criterios
        diferentes para a mesma pergunta produziriam estado incoerente com o matchup.
        """
        opponent = battle.opponent_active_pokemon
        candidatos = getattr(battle, "available_switches", None) or []
        if not candidatos or not opponent:
            return "BANCO_VAZIO"

        # Ameaca do adversario: tipos sempre visiveis, mais os golpes ja revelados.
        ameacas = [t for t in opponent.types if t]
        ameacas += [m for m in opponent.moves.values() if m.base_power > 0]

        tem_def = False
        tem_off = False
        for cand in candidatos:
            if cand.fainted:
                continue
            if not tem_def and ameacas:
                pior = max(cand.damage_multiplier(a) for a in ameacas)
                if pior < 1.0:
                    tem_def = True
            if not tem_off:
                golpes = [m for m in cand.moves.values() if m.base_power > 0]
                if golpes and max(opponent.damage_multiplier(m) for m in golpes) > 1.0:
                    tem_off = True
            if tem_def and tem_off:
                break

        if tem_def and tem_off:
            return "BANCO_AMBOS"
        if tem_def:
            return "BANCO_DEF"
        if tem_off:
            return "BANCO_OFF"
        return "BANCO_VAZIO"

    # -- Velocidade (usa a física) -----------------------------------------

    def get_speed_tier(self, battle):
        """FASTER se agimos primeiro NESTE turno, contando clima, campo e Trick Room.

        CORRIGIDO EM 30/08/2026. Antes chamava `estimate_stat(mon, 'spe')` sem
        `battle`, e o `_get_speed_mod` da fisica so conhecia paralisia e boosts. Nove
        fontes ficavam de fora: Swift Swim, Chlorophyll, Sand Rush, Slush Rush, Surge
        Surfer, Tailwind, Slow Start, Unburden e Trick Room.

        O ABSURDO ERA INTERNO A ESTA TUPLA. O `get_weather_state` logo acima DETETA
        Swift Swim com chuva, Tailwind e Trick Room, e devolve `FIELD_SPEED` na
        dimensao 5. A dimensao 6 saia daqui e dizia `SLOWER` na mesma situacao. Duas
        dimensoes do mesmo estado, a afirmar o contrario uma da outra, e os agentes a
        aprender por cima disso. Com Trick Room era pior que inconsistente: era
        exatamente ao contrario da verdade.

        A comparacao passa para `physics.mais_rapido`, que aplica os modificadores aos
        dois lados e inverte o teste sob Trick Room. Empate continua a dar SLOWER.
        """
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not active or not opponent:
            return "SLOWER"
        return "FASTER" if self.physics.mais_rapido(active, opponent, battle) else "SLOWER"

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

        # QUEIMADURA E PARALISIA SAO SEMPRE DEBUFF (30/08/2026).
        #
        # Antes a queimadura so contava como debuff se `_is_physical(pokemon)` fosse
        # verdadeiro. Um atacante especial queimado aparecia como NEUTRAL, apesar de
        # estar a perder 1/16 do HP por turno — dano residual e debuff, mesmo quando o
        # corte de Ataque nao se aplica.
        #
        # A concatenacao (BUFFED_DEBUFF, DEBUFF_DEBUFF) e DELIBERADA e mantem-se. Os
        # dois estatutos aparecem em DUAS dimensoes de proposito: na 8/9 como
        # AFFLICTED (ha status) e aqui como debuff (o status penaliza). E assim que o
        # cerebro pode aprender a FUNCAO do status e nao so a sua presenca — "estou
        # com status" e "estou pior por causa dele" sao perguntas diferentes, e a
        # resposta a segunda e o que decide entre curar, trocar e continuar.
        if pokemon.status:
            s_name = pokemon.status.name
            if s_name in ('BRN', 'PAR'):
                state = "DEBUFF" if state == "NEUTRAL" else state + "_DEBUFF"

        return state

    # -- Hazards ------------------------------------------------------------

    def get_hazard_state(self, side_conditions):
        if not side_conditions:
            return "CLEAR"
        # Normalizacao igual a de get_weather_state (26/08/2026): um so estilo no
        # ficheiro. O teste anterior era por SUBCADEIA e ja funcionava; passa a
        # pertenca a lista sobre o nome do membro, que e mais estrito e nao muda o
        # resultado (SPIKES deixa de casar dentro de TOXIC_SPIKES por acidente).
        # MIGRADO PARA `physics.tem` EM 03/09/2026. ESTE ERA O PIOR DOS 22: com a
        # forma antiga `get_hazard_state` devolvia SEMPRE "CLEAR", logo as
        # dimensoes 12 e 13 da tupla de estado (my_hazard, opp_hazard) eram
        # CONSTANTES em todos os treinos ja feitos. Blue e Green aprenderam sem
        # nunca ver um hazard no estado.
        if self.physics.tem(side_conditions,
                            'STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB'):
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
        # O roster adversario e revelado no team preview, mas `opponent_team` pode
        # passar a representar apenas os Pokemon efetivamente revelados depois do
        # arranque. Usa a fusao preview+revelados da fisica para nao transformar
        # turno 1 em BRAWL/DOMINATING por subcontagem do adversario.
        opp_alive, _ = self.physics.equipa_adversaria(battle)
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

        # `battle` passado de proposito: sob Trick Room o papel inverte-se com a
        # velocidade. Ver `get_role` na fisica.
        my_role = self.physics.get_role(active, battle).name
        opp_role = self.physics.get_role(opponent, battle).name
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
            self.get_bench_state(battle),          # 15 — acrescentada no FIM
        )

        # Garantia de contrato: a tupla tem sempre STATE_DIM elementos.
        assert len(state) == STATE_DIM, f"StateParser produziu {len(state)} dims, esperado {STATE_DIM}"
        return state
