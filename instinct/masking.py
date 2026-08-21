"""
Camada 3 — Action Masking (ActionMasker).

Poda ações taticamente inválidas ANTES de a decisão chegar ao aprendizado. É o
coração do conceito "filtrado por instinto" do projeto ALFINETE: em vez de deixar
a Q-table descobrir sozinha que atacar um imune é inútil (gastando milhares de
episódios), o masking remove essas ações do espaço logo à partida, focando o
aprendizado nas decisões que de facto importam.

Depende da Camada 1 (GamePhysics, para classify_move). Recebe-a por injeção.
NÃO depende do parser de estado nem da política — só olha para a legalidade e a
utilidade tática de cada ação isolada.

Contrato:
- get_available_actions(battle) -> lista de nomes de MoveCategory jogáveis.
- is_move_useless(move, opponent, battle, history) -> bool (True = podar).
- is_hazard_already_set(move, battle) -> bool.

NOTA DE AUDITORIA: dois bugs do monólito original foram PRESERVADOS aqui (marcados
com `# BUG:`) para manter equivalência de comportamento durante a refatoração. Devem
ser decididos explicitamente — corrigir muda a estratégia de jogo, por isso não foi
feito automaticamente.
"""

from shared.definitions import MoveCategory


class ActionMasker:
    """Componente de poda de ações. Recebe a física por injeção."""

    def __init__(self, physics):
        self.physics = physics

    # ======================================================================
    # ORQUESTRADOR: que categorias de ação estão disponíveis neste turno
    # ======================================================================

    def get_available_actions(self, battle):
        """Constrói o conjunto de MoveCategory jogáveis, já podadas.

        Entra: battle. Sai: lista de nomes de categoria (strings).
        Regra especial: se houver golpes de tipos diferentes E oponentes no banco,
        habilita ATTACK_PREDICTIVE (prever a troca do oponente).
        """
        available = set()

        if battle.available_switches:
            available.add(MoveCategory.SWITCH_DEFENSIVE.name)
            available.add(MoveCategory.SWITCH_OFFENSIVE.name)

        if battle.available_moves:
            damaging_types = set()
            for move in battle.available_moves:
                if self.is_move_useless(move, battle.opponent_active_pokemon, battle):
                    continue

                cat = self.physics.classify_move(move)
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

        # Rede de segurança: nunca devolver lista vazia (o agente precisa de opções).
        if not available_list:
            if battle.available_switches:
                return [MoveCategory.SWITCH_DEFENSIVE.name, MoveCategory.SWITCH_OFFENSIVE.name]
            else:
                if battle.available_moves:
                    cats = [self.physics.classify_move(m).name for m in battle.available_moves]
                    if MoveCategory.PROTECT.name in cats:
                        return [MoveCategory.PROTECT.name]
                    return list(set(c for c in cats if c != "UNKNOWN")) or ["ATTACK_STRONG"]
                return ["ATTACK_STRONG"]

        return available_list

    # ======================================================================
    # HAZARDS: já estão colocados? (evita repor stealth rock, etc.)
    # ======================================================================

    def is_hazard_already_set(self, move, battle):
        """True se o hazard do golpe JÁ atingiu o seu limite no lado do oponente
        (poda a ação por ser redundante).

        REGRA DE JOGO (corrigida): os quatro hazards PODEM coexistir todos no mesmo
        lado (o chamado "full hazard stack"). Não há restrição de combinações dois-a-
        dois. O único limite é o número de CAMADAS por hazard:
            - STEALTH_ROCK : não empilha (1 camada máx.)
            - STICKY_WEB   : não empilha (1 camada máx.)
            - SPIKES       : até 3 camadas
            - TOXIC_SPIKES : até 2 camadas
        A versão anterior tinha uma regra `len(current_hazards) >= 2 -> podar` que
        assumia (erradamente) coexistência limitada; ela podava indevidamente um 3.º
        ou 4.º hazard legal. Foi REMOVIDA.

        Sobre o poke-env: `battle.opponent_side_conditions` é um dict
        {SideCondition: valor}. Para SPIKES e TOXIC_SPIKES o valor é o número de
        camadas (int); para STEALTH_ROCK e STICKY_WEB é tipicamente 1 (presença).
        Distinguimos SPIKES de TOXIC_SPIKES pelo nome da condição, porque ambos
        contêm a substring 'SPIKES'.
        """
        move_to_hazard = {
            'stealthrock': 'STEALTH_ROCK',
            'stickyweb': 'STICKY_WEB',
            'spikes': 'SPIKES',
            'toxicspikes': 'TOXIC_SPIKES',
        }
        target_hazard = move_to_hazard.get(move.id)
        if not target_hazard:
            return False

        # Lê as camadas atuais de cada hazard no lado do oponente.
        current = {}
        for condition, layers in battle.opponent_side_conditions.items():
            cond_str = str(condition).upper()
            if 'TOXIC_SPIKES' in cond_str or ('TOXIC' in cond_str and 'SPIKES' in cond_str):
                current['TOXIC_SPIKES'] = int(layers) if isinstance(layers, int) else 1
            elif 'SPIKES' in cond_str:
                current['SPIKES'] = int(layers) if isinstance(layers, int) else 1
            elif 'STEALTH_ROCK' in cond_str:
                current['STEALTH_ROCK'] = 1
            elif 'STICKY_WEB' in cond_str:
                current['STICKY_WEB'] = 1

        # Poda apenas se ESTE hazard específico já está no seu limite de camadas.
        if target_hazard == 'STEALTH_ROCK':
            return 'STEALTH_ROCK' in current
        if target_hazard == 'STICKY_WEB':
            return 'STICKY_WEB' in current
        if target_hazard == 'SPIKES':
            return current.get('SPIKES', 0) >= 3
        if target_hazard == 'TOXIC_SPIKES':
            return current.get('TOXIC_SPIKES', 0) >= 2
        return False

    # ======================================================================
    # A REGRA PESADA: um golpe é inútil neste contexto?
    # (13 filtros, do mais absoluto ao mais situacional)
    # ======================================================================

    def is_move_useless(self, move, opponent, battle, history=None):
        if not move:
            return True
        active = battle.active_pokemon
        if not active or not opponent:
            return True

        # 0. IMUNIDADE DE TIPO ABSOLUTA (multiplicador 0)
        if move.category.name != "STATUS":
            if opponent.damage_multiplier(move) == 0:
                return True

        opp_types = [t.name for t in opponent.types if t]
        opp_abilities = [str(opponent.ability).lower()] if opponent.ability else []
        if opponent.possible_abilities:
            opp_abilities.extend([str(a).lower() for a in opponent.possible_abilities])

        move_type = move.type.name if move.type else ""

        # 1. IMUNIDADES POR HABILIDADE (água/elétrico/fogo/planta/terra)
        if move.category.name != "STATUS" and move.base_power > 0:
            if move_type == "WATER" and any(ab in opp_abilities for ab in ['waterabsorb', 'dryskin', 'stormdrain']):
                return True
            if move_type == "ELECTRIC" and any(ab in opp_abilities for ab in ['voltabsorb', 'motordrive', 'lightningrod']):
                return True
            if move_type == "FIRE" and any(ab in opp_abilities for ab in ['flashfire', 'wellbakedbody']):
                return True
            if move_type == "GRASS" and any(ab in opp_abilities for ab in ['sapsipper']):
                return True
            if move_type == "GROUND" and any(ab in opp_abilities for ab in ['levitate', 'eartheater']):
                return True

        # 2. MAGIC BOUNCE / GOOD AS GOLD (golpes de status dirigidos ao oponente)
        if move.category.name == "STATUS" or move.base_power == 0:
            targets_opponent = str(move.target).lower() not in ['self', 'allyside', 'allyteam', 'adjacentally']
            if targets_opponent:
                if any(ab in opp_abilities for ab in ['magicbounce']):
                    return True
                if any(ab in opp_abilities for ab in ['goodasgold']):
                    return True

        # 3. PRIORIDADE, PRIMEIRO TURNO, FLINCH
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

        # 4. STATUS: imunidades e aplicação redundante
        if move.category.name == "STATUS":
            if active.ability == 'prankster' and 'DARK' in opp_types:
                return True
            if move.id in ['spore', 'sleeppowder', 'stunspore', 'poisonpowder', 'ragepowder']:
                if 'GRASS' in opp_types or 'overcoat' in opp_abilities:
                    return True
            if move.id == 'thunderwave' and ('GROUND' in opp_types or 'ELECTRIC' in opp_types):
                return True
            if move.id == 'leechseed' and 'GRASS' in opp_types:
                return True
            if any(ab in opp_abilities for ab in ['magicbounce']) and getattr(move, 'target', '') in ['normal', 'allAdjacentFoes', 'foeSide']:
                return True
            if any(ab in opp_abilities for ab in ['goodasgold', 'magicguard']):
                return True
            if move.id in ['confuseray', 'swagger'] and any(ab in opp_abilities for ab in ['owntempo', 'oblivious']):
                return True

            if move.status:
                if opponent.status:
                    return True
                if 'synchronize' in opp_abilities:
                    my_types = [t.name for t in active.types if t]
                    if move.status.name in ['TOX', 'PSN'] and 'POISON' not in my_types and 'STEEL' not in my_types:
                        return True
                    if move.status.name == 'BRN' and 'FIRE' not in my_types:
                        return True
                    if move.status.name == 'PRZ' and 'ELECTRIC' not in my_types and 'GROUND' not in my_types:
                        return True
                if move.status.name in ['TOX', 'PSN']:
                    if 'immunity' in opp_abilities:
                        return True
                    if 'POISON' in opp_types or 'STEEL' in opp_types:
                        if active.ability != 'corrosion':
                            return True
                elif move.status.name == 'BRN':
                    if 'FIRE' in opp_types or any(ab in opp_abilities for ab in ['waterveil', 'waterbubble']):
                        return True
                elif move.status.name == 'PRZ':
                    if 'ELECTRIC' in opp_types or 'limber' in opp_abilities:
                        return True
                elif move.status.name == 'SLP':
                    if any(ab in opp_abilities for ab in ['insomnia', 'vitalspirit', 'sweetveil']):
                        return True

        # 5. BUFFS já maximizados / DEBUFFS contra clear body
        if move.category.name == "STATUS":
            boosts = getattr(move, 'boosts', None) or getattr(move, 'self_boost', None)
            if boosts:
                target_str = str(getattr(move, 'target', '')).lower()
                if 'self' in target_str:
                    is_useful = False
                    for stat, boost_amount in boosts.items():
                        current_stage = active.boosts.get(stat, 0)
                        if boost_amount > 0 and current_stage < 6:
                            is_useful = True
                            break
                        elif boost_amount < 0:
                            is_useful = True
                    if not is_useful:
                        return True
                elif 'normal' in target_str or 'foe' in target_str:
                    if any(ab in opp_abilities for ab in ['clearbody', 'whitesmoke', 'fullmetalbody']):
                        if any(b < 0 for b in boosts.values()):
                            return True

        # 5.5. LOOPS DE SUBSTITUTE / LEECH SEED
        if move.id == 'substitute':
            if active.effects and any('substitute' in str(e).lower() for e in active.effects):
                return True
            if active.current_hp_fraction <= 0.25:
                return True
        if move.id == 'leechseed':
            if opponent.effects and any('leechseed' in str(e).lower() for e in opponent.effects):
                return True

        # 6. BARREIRAS / CLIMA já ativos
        current_weather = next(iter(battle.weather)).name if battle.weather else "CLEAR"
        if move.id in ['reflect', 'lightscreen', 'auroraveil', 'safeguard', 'tailwind']:
            my_side = [str(k).upper() for k in battle.side_conditions.keys()]
            if move.id == 'reflect' and 'REFLECT' in my_side:
                return True
            if move.id == 'lightscreen' and 'LIGHT_SCREEN' in my_side:
                return True
            if move.id == 'safeguard' and 'SAFEGUARD' in my_side:
                return True
            if move.id == 'tailwind' and 'TAILWIND' in my_side:
                return True
            if move.id == 'auroraveil':
                if 'AURORA_VEIL' in my_side:
                    return True
                if current_weather not in ['HAIL', 'SNOW', 'SNOWSCAPE']:
                    return True

        weather_moves = ['raindance', 'sunnyday', 'sandstorm', 'hail', 'snowscape']
        if move.id in weather_moves:
            if move.id == 'raindance' and current_weather in ['RAINDANCE', 'PRIMORDIALSEA']:
                return True
            if move.id == 'sunnyday' and current_weather in ['SUNNYDAY', 'DESOLATELAND']:
                return True
            if move.id == 'sandstorm' and current_weather == 'SANDSTORM':
                return True
            if move.id in ['hail', 'snowscape'] and current_weather in ['HAIL', 'SNOW', 'SNOWSCAPE']:
                return True

        # 6.5. LIMPEZA DE HAZARDS DESNECESSÁRIA
        if move.id in ['defog', 'rapidspin', 'mortalspin', 'courtchange']:
            my_side = [str(k).upper() for k in battle.side_conditions.keys()]
            opp_side = [str(k).upper() for k in battle.opponent_side_conditions.keys()]
            hazards_list = ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB']
            my_hazards = any(h in cond for cond in my_side for h in hazards_list)

            if move.id in ['rapidspin', 'mortalspin']:
                is_trapped = active.effects and any(e in str(active.effects).lower() for e in ['leechseed', 'bind', 'wrap', 'firespin', 'magmastorm'])
                if not my_hazards and not is_trapped:
                    return True
            elif move.id == 'defog':
                opp_screens = any(s in cond for cond in opp_side for s in ['REFLECT', 'LIGHT_SCREEN', 'AURORA_VEIL', 'SAFEGUARD'])
                if not my_hazards and not opp_screens:
                    return True
            elif move.id == 'courtchange':
                if not my_hazards and not opp_side:
                    return True

        # 7. PROTECT CONSECUTIVO
        if move.id in ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark', 'silktrap', 'obstruct', 'endure']:
            if history:
                prev_act = history.get('prev_action')
                last_act = history.get('last_action')
                str_prev = str(prev_act[0]) if isinstance(prev_act, tuple) else str(prev_act)
                str_last = str(last_act[0]) if isinstance(last_act, tuple) else str(last_act)
                if "PROTECT" in str_prev or "PROTECT" in str_last:
                    return True

        # 8. ÚLTIMO POKÉMON: phazing e hazards inúteis
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        if move.id in ['roar', 'whirlwind', 'dragontail', 'circlethrow']:
            if opp_alive <= 1:
                return True
            if 'suctioncups' in opp_abilities:
                return True
        if move.id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb'] and opp_alive <= 1:
            return True

        # 9. PREDICT DE IMUNIDADES POR ESPÉCIE
        # CORRIGIDO: o poke-env expõe move.category como enum MoveCategory
        # (PHYSICAL/SPECIAL/STATUS). A versão antiga comparava o enum com as strings
        # ["Physical","Special"], o que dá SEMPRE False -> o bloco nunca executava.
        # Agora comparamos por .name em maiúsculas, consistente com os outros filtros.
        if move.category.name in ["PHYSICAL", "SPECIAL"]:
            opp_species = str(opponent.species).lower()
            if move.type.name == "WATER":
                if opp_species in ['vaporeon', 'gastrodon', 'seismitoad', 'toxicroak', 'mantine', 'clodsire', 'volcanion']:
                    return True
            elif move.type.name == "FIRE":
                if opp_species in ['heatran', 'chandelure', 'arcanine', 'ceruledge', 'houndoom', 'dachsbun']:
                    return True
            elif move.type.name == "ELECTRIC":
                if opp_species in ['jolteon', 'thundurus', 'thundurustherian', 'zeraora', 'electivire', 'raichu', 'marowakalola']:
                    return True
            elif move.type.name == "GRASS":
                if opp_species in ['azumarill', 'goodra', 'bouffalant']:
                    return True
            elif move.type.name == "GROUND":
                if opp_species in ['rotom', 'rotomwash', 'rotomheat', 'rotommow', 'latios', 'latias', 'hydreigon', 'cresselia', 'weezing', 'orthworm']:
                    return True

        # 10. CURA DESNECESSÁRIA (overheal)
        healing_moves = ['recover', 'roost', 'slackoff', 'softboiled', 'milkdrink', 'shoreup', 'moonlight', 'morningsun', 'synthesis', 'healorder', 'wish']
        # CORRIGIDO: mesma questão de enum-vs-string do Filtro 9. Agora a segunda
        # condição (qualquer golpe de status com heal > 0) também dispara, cobrindo
        # golpes de cura que não estejam na lista explícita healing_moves.
        if move.id in healing_moves or (getattr(move, 'heal', 0) and move.category.name == "STATUS"):
            if active.current_hp_fraction >= 0.95:
                return True
        if move.id in ['aromatherapy', 'healbell', 'junglehealing']:
            team_needs_heal = any(m.status is not None and not m.fainted for m in battle.team.values())
            if not team_needs_heal:
                return True

        # 11. TERRENOS que bloqueiam status (Misty impede todos, Electric impede sleep)
        # NOTA: a verificação redundante `if opponent.status: return True` foi REMOVIDA
        # daqui — já é feita no Filtro 4. Mantém-se apenas a lógica única de terrenos.
        if move.status:
            active_fields = [str(f).upper() for f in battle.fields.keys()]
            grounded_opp = 'FLYING' not in opp_types and not (opponent.item and str(opponent.item).lower() == 'airballoon') and 'levitate' not in opp_abilities
            if grounded_opp:
                if 'MISTY_TERRAIN' in active_fields:
                    return True
                if 'ELECTRIC_TERRAIN' in active_fields and move.status.name == 'SLP':
                    return True

        # 12. DISRUPTION já ativa / imunidades
        if move.id in ['taunt', 'torment', 'encore', 'disable']:
            if opponent.effects:
                if any(move.id in str(e).lower() for e in opponent.effects):
                    return True
            if move.id == 'taunt' and any(ab in opp_abilities for ab in ['oblivious', 'aromaveil']):
                return True

        return False
