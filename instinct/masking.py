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
- get_available_actions(battle, history) -> lista de nomes de MoveCategory jogáveis.
- is_move_useless(move, opponent, battle, history) -> bool (True = podar).
- is_hazard_already_set(move, battle) -> bool.

NOTA DE AUDITORIA: dois bugs do monólito original foram PRESERVADOS aqui (marcados
com `# BUG:`) para manter equivalência de comportamento durante a refatoração. Devem
ser decididos explicitamente — corrigir muda a estratégia de jogo, por isso não foi
feito automaticamente.
"""

from shared import diagnostico
from shared.definitions import MoveCategory



def nomes_de_enum(colecao):
    """Nomes dos membros de enum de uma coleccao do poke-env, sem o prefixo da classe.

    CORRIGE UM BUG SILENCIOSO PRESENTE EM 8 SITIOS (28/08/2026).

    `str(SideCondition.REFLECT).upper()` devolve "SIDECONDITION.REFLECT", nao
    "REFLECT". O projeto tinha DOIS estilos de teste misturados:

        SUBCADEIA   any(h in cond for cond in lista)   -> funciona
        PERTENCA    "REFLECT" in lista                 -> SEMPRE FALSO

    Todos os testes por pertenca estavam mortos. Consequencias medidas:
      - `_get_hazard_damage` devolvia sempre 0: a escolha de troca ignorava o dano
        de entrada, com hazards em 91% dos times do pool
      - a reorganizacao por barreiras (Reflect / Light Screen / Aurora Veil) nunca
        corria
      - o masking deixava repetir Reflect, Light Screen, Safeguard e Tailwind ja em
        campo, e deixava tentar adormecer sob Electric Terrain

    Um so estilo no projeto inteiro, e o teste passa a ser por pertenca sobre o NOME
    do membro, que e mais estrito que subcadeia (SPIKES deixa de casar por acidente
    dentro de TOXIC_SPIKES).
    """
    # MIGRADO PARA `.name` EM 03/09/2026. O `.split(".")` nao fazia NADA, porque
    # o `__str__` do poke-env nao tem ponto: devolvia
    # 'REFLECT (SIDE CONDITION) OBJECT'. Logo esta funcao, criada em 28/08 para
    # permitir testes por PERTENCA, entregava exactamente o oposto do que
    # prometia e todos os seus consumidores estavam mortos.
    return sorted({str(getattr(k, "name", k)).upper() for k in (colecao or {})})

class ActionMasker:
    """Componente de poda de ações. Recebe a física por injeção."""

    # Habilidades que vale a pena TER. Nao pretende ser exaustiva: serve para
    # decidir se uma troca de habilidade e ganho ou prejuizo, e na duvida (fora da
    # lista) trata-se como neutra, o que faz o filtro NAO podar por engano.
    HABILIDADES_VALIOSAS = {
        # clima e terreno
        'drought', 'drizzle', 'sandstream', 'snowwarning', 'electricsurge',
        'grassysurge', 'mistysurge', 'psychicsurge', 'orichalcumpulse', 'hadronengine',
        # ofensivas
        'hugepower', 'purepower', 'adaptability', 'sheerforce', 'technician',
        'skilllink', 'guts', 'protean', 'libero', 'speedboost', 'moxie', 'beastboost',
        'moldbreaker', 'toughclaws', 'strongjaw', 'ironfist',
        # defensivas e utilitarias
        'regenerator', 'multiscale', 'magicbounce', 'levitate', 'unaware',
        'prankster', 'intimidate', 'thickfat', 'naturalcure', 'poisonheal',
        'waterabsorb', 'voltabsorb', 'flashfire', 'stormdrain', 'lightningrod',
        'sapsipper', 'wonderguard', 'magicguard', 'noguard', 'serenegrace',
    }
    # Habilidades que ATRAPALHAM o dono: passa-las ao adversario e a jogada.
    HABILIDADES_PREJUDICIAIS = {
        'truant', 'slowstart', 'defeatist', 'klutz', 'stall', 'slowstart',
        'normalize', 'weakarmor',
    }
    # Skill Swap FALHA contra estas. Lista de regra de jogo, nao de gosto.
    HABILIDADES_INTROCAVEIS = {
        'wonderguard', 'multitype', 'stancechange', 'schooling', 'comatose',
        'shieldsdown', 'disguise', 'rkssystem', 'battlebond', 'powerconstruct',
        'iceface', 'gulpmissile', 'asoneglastrier', 'asonespectrier', 'zerotohero',
        'commander', 'hungerswitch', 'neutralizinggas', 'protosynthesis', 'quarkdrive',
    }

    # Itens cujo dono PERDE com eles. Sao os unicos que vale a pena passar com
    # Trick / Switcheroo: os Choice trancam o alvo num golpe, os Orbs envenenam ou
    # queimam, o Iron Ball e o Lagging Tail atrasam, o Sticky Barb magoa.
    ITENS_QUE_ATRAPALHAM = {
        'choiceband', 'choicespecs', 'choicescarf',
        'toxicorb', 'flameorb', 'stickybarb',
        'ironball', 'laggingtail', 'ringtarget', 'fullincense', 'machobrace',
    }

    def __init__(self, physics):
        self.physics = physics

    # ======================================================================
    # ORQUESTRADOR: que categorias de ação estão disponíveis neste turno
    # ======================================================================

    def get_available_actions(self, battle, history=None):
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
                if self.is_move_useless(move, battle.opponent_active_pokemon, battle, history):
                    continue

                # CONTEXTO (29/08/2026). Sem `opponent` e `battle`, o classify_move
                # nao sabe se o efeito de um ATTACK_TECH ja esta gasto — e o EXECUTOR
                # sabe, porque recebe contexto desde 28/08. As duas camadas ficavam a
                # discordar sobre o mesmo golpe: a mascara oferecia ATTACK_TECH, o
                # executor via ATTACK_STRONG, nao encontrava candidatos e devolvia
                # None. O ranking avancava e jogava-se outra coisa.
                #
                # O problema NAO e so o turno perdido: a Q-table registava a
                # recompensa na ACAO ERRADA. Corromper a representacao e pior do que
                # qualquer questao de simetria entre agentes.
                cat = self.physics.classify_move(
                    move, battle.opponent_active_pokemon, battle)
                if cat != MoveCategory.UNKNOWN:
                    if cat == MoveCategory.HAZARD and self.is_hazard_already_set(move, battle):
                        continue
                    available.add(cat.name)

                    if cat in [MoveCategory.ATTACK_STRONG, MoveCategory.ATTACK_TECH, MoveCategory.ATTACK_PIVOT]:
                        if move.type:
                            damaging_types.add(move.type)

            if MoveCategory.ATTACK_STRONG.name in available and len(damaging_types) > 1:
                _, benched_opponents = self.physics.equipa_adversaria(battle)
                if benched_opponents:
                    available.add(MoveCategory.ATTACK_PREDICTIVE.name)

        available_list = list(available)

        # Rede de segurança: nunca devolver lista vazia (o agente precisa de opções).
        if not available_list:
            if battle.available_switches:
                return [MoveCategory.SWITCH_DEFENSIVE.name, MoveCategory.SWITCH_OFFENSIVE.name]
            else:
                if battle.available_moves:
                    # Contexto tambem aqui: esta e a rede de seguranca, e devolver
                    # uma categoria que o executor nao reconhece seria devolver uma
                    # rede rota.
                    cats = [self.physics.classify_move(
                        m, battle.opponent_active_pokemon, battle).name
                        for m in battle.available_moves]
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
        # Uniformizado com nomes_de_enum (28/08/2026): este bloco usava SUBCADEIA e
        # ja funcionava, mas ter dois estilos no mesmo ficheiro foi o que escondeu o
        # bug durante meses. Por pertenca exata, SPIKES deixa de casar por acidente
        # dentro de TOXIC_SPIKES, o que tornava a ordem dos elif significativa.
        for condition, layers in battle.opponent_side_conditions.items():
            # NORMALIZACAO ROBUSTA (29/08/2026). Observado em batalha manual: um
            # Clefable do InstinctBot usou Stealth Rock em DOIS turnos seguidos, e
            # falhou nos dois — este filtro nao estava a podar.
            #
            # A causa e a forma da chave. Conforme a versao do poke-env e o caminho
            # de parsing, `opponent_side_conditions` pode vir com:
            #
            #     SideCondition.STEALTH_ROCK   -> "STEALTH_ROCK"   (com underscore)
            #     "stealthrock"                -> "STEALTHROCK"    (SEM underscore)
            #
            # A versao anterior so tratava a primeira. Com a segunda, NENHUM dos elif
            # casava, `current` ficava VAZIO, e `'STEALTH_ROCK' in current` devolvia
            # False — ou seja, o filtro dizia sempre "ainda nao esta posto".
            #
            # Comparar sem underscores resolve as duas formas de uma vez.
            # ==========================================================
            # DUAS CORRECCOES EM 03/09/2026. ESTE E O LOOP DO STEALTH ROCK.
            # ==========================================================
            # 1. O NOME. `str(condition)` devolve
            #    'STEALTH_ROCK (side condition) object' e nao ha ponto nenhum,
            #    logo o `.split(".")` nao fazia nada e NENHUM dos `elif` casava.
            #    `current` ficava vazio e o filtro dizia sempre "ainda nao esta
            #    posto". Provado pela sonda [HAZ] numa batalha manual:
            #      raw={<SideCondition.STEALTH_ROCK: 19>: 14}  normalizado={}
            #    e o instinto repetiu Stealth Rock no turno seguinte ao de o ter
            #    posto. O `.name` e o unico atributo que o poke-env nao redefine.
            #
            # 2. O SIGNIFICADO DO VALOR DEPENDE DO HAZARD (verificado 04/09/2026
            #    no codigo do poke-env instalado, `abstract_battle._side_start`):
            #
            #        if condition in STACKABLE_CONDITIONS:
            #            conditions[condition] = conditions.get(condition, 0) + 1
            #        elif condition not in conditions:
            #            conditions[condition] = self.turn
            #
            #    e `STACKABLE_CONDITIONS = {SPIKES: 3, TOXIC_SPIKES: 2}`. Ou seja:
            #
            #        SPIKES, TOXIC_SPIKES  -> o valor E o NUMERO DE CAMADAS
            #        tudo o resto          -> o valor e o TURNO DE INICIO
            #
            #    O poke-env tambem ja limpa sozinho em Defog e Rapid Spin, via
            #    `side_end`, que faz `conditions.pop(condition)`. NAO E PRECISO
            #    contador proprio: a contagem existe e e fiavel.
            #
            #    NOTA DE HISTORIA, para nao se repetir o erro: em 03/09 assumiu-se
            #    que o valor era SEMPRE o turno (por analogia com `_fields`, que e
            #    mesmo sempre o turno) e passou-se a contar uma camada fixa. Era
            #    errado para os dois unicos hazards que empilham. O `int(layers)`
            #    original estava CERTO; o que estava morto era so o NOME, acima.
            nome = str(getattr(condition, "name", condition)).upper().replace("_", "")
            if nome == 'TOXICSPIKES':
                current['TOXIC_SPIKES'] = int(layers) if isinstance(layers, int) else 1
            elif nome == 'SPIKES':
                current['SPIKES'] = int(layers) if isinstance(layers, int) else 1
            elif nome == 'STEALTHROCK':
                current['STEALTH_ROCK'] = 1
            elif nome == 'STICKYWEB':
                current['STICKY_WEB'] = 1

        # ==============================================================
        # SONDA [HAZ] (03/09/2026) — TEMPORARIA, SAI QUANDO RESPONDER
        # ==============================================================
        # PERGUNTA QUE ESTA SONDA EXISTE PARA RESPONDER: porque e que o Stealth
        # Rock repetido nao e podado. O bug foi dado como corrigido no v16 (6.35
        # Bug 3), a 6.42 marcou-o como "por confirmar", e em 03/09 voltou a ser
        # observado — Garchomp do InstinctBot a usar Stealth Rock em tres turnos
        # seguidos, a falhar nos tres.
        #
        # Este filtro esta CORRECTO POR LEITURA e nao segura em EXECUCAO, que e a
        # mesma assinatura do carrossel de trocas. Ler mais nao resolve esta
        # classe de defeito; o `repr` da coleccao resolve.
        #
        # O `!r` e o essencial: mostra a FORMA exacta das chaves e dos valores.
        # Quatro hipoteses, e o repr distingue-as todas de uma vez:
        #   1. o dicionario vem VAZIO
        #   2. vem com chave de forma inesperada e a normalizacao falha
        #   3. vem correcto -> a poda funciona e o problema esta A MONTANTE
        #   4. `layers` nao e o numero de camadas mas o TURNO DE INICIO
        #
        # RESPONDIDA EM 04/09/2026: era a hipotese 2, o NOME. A hipotese 4
        # confirmou-se APENAS para os hazards que NAO empilham; Spikes e Toxic
        # Spikes guardam mesmo a contagem de camadas. Ver a nota acima e 6.46 E.
        diagnostico.log("HAZ", f"turno={getattr(battle, 'turn', '?')} "
                               f"golpe={move.id} alvo={target_hazard} "
                               f"raw={battle.opponent_side_conditions!r} "
                               f"normalizado={current!r}")

        # Poda apenas se ESTE hazard específico já está no seu limite de camadas.
        if target_hazard == 'STEALTH_ROCK':
            return 'STEALTH_ROCK' in current
        if target_hazard == 'STICKY_WEB':
            return 'STICKY_WEB' in current
        # Limites reais do jogo, e os mesmos que o poke-env usa em
        # STACKABLE_CONDITIONS: Spikes 3 camadas, Toxic Spikes 2.
        if target_hazard == 'SPIKES':
            return current.get('SPIKES', 0) >= 3
        if target_hazard == 'TOXIC_SPIKES':
            return current.get('TOXIC_SPIKES', 0) >= 2
        return False

    # ======================================================================
    # A REGRA PESADA: um golpe é inútil neste contexto?
    # (14 filtros, do mais absoluto ao mais situacional)
    # ======================================================================

    def is_move_useless(self, move, opponent, battle, history=None):
        if not move:
            return True
        active = battle.active_pokemon
        if not active or not opponent:
            return True

        # 0. IMUNIDADE DE TIPO ABSOLUTA (multiplicador 0)
        #
        # MIGRADO PARA `physics.multiplicador_de_tipo` EM 03/09/2026. O
        # `damage_multiplier` do poke-env resolve a tabela de tipos e ignora as
        # excepcoes escritas na descricao do proprio golpe:
        #
        #   Thousand Arrows contra VOADOR    poke-env diz 0x -> este filtro
        #                                    PODAVA um golpe legal e bom
        #   Freeze-Dry contra AGUA           0,5x quando o real e 2x
        #
        # A regra vive no `physics` e num sitio so, pela mesma razao do
        # `mais_rapido`: e aritmetica do MOTOR DO JOGO, nao estrategia, logo vale
        # por igual para os tres agentes.
        if move.category.name != "STATUS":
            if self.physics.multiplicador_de_tipo(move, opponent) == 0:
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

            # ==========================================================
            # 1b. IMUNIDADE POR PROPRIEDADE DO GOLPE (04/09/2026)
            # ==========================================================
            # As imunidades acima sao todas por TIPO do golpe. Faltavam as que
            # dependem de uma PROPRIEDADE dele, e que nenhum tipo nem
            # multiplicador revela:
            #
            #   Bulletproof   anula projecteis e bombas, seja qual for o tipo
            #   Soundproof    anula golpes de som
            #
            # Observado em batalha manual: Tyranitar usou `rockblast` contra um
            # Kommo-o com Bulletproof, tres vezes. O golpe e ROCHA e o
            # multiplicador de tipo e normal — nada no filtro 0 o podia apanhar.
            #
            # POR ID E NAO POR FLAG. O poke-env expoe `move.flags`, mas o conteudo
            # varia entre versoes e ja custou uma campanha de medicoes ao projeto
            # confiar num campo de biblioteca sem o verificar. As listas sao
            # curtas, estaveis e verificaveis; acrescentar um golpe e uma linha.
            _BULLETPROOF = {
                'aurasphere', 'acidspray', 'barrage', 'beakblast', 'bulletseed',
                'eggbomb', 'electroball', 'energyball', 'focusblast', 'gyroball',
                'iceball', 'magnetbomb', 'mistball', 'mudbomb', 'octazooka',
                'pollenpuff', 'poweruppunch', 'pyroball', 'rockblast',
                'rockwrecker', 'searingshot', 'seedbomb', 'shadowball',
                'sludgebomb', 'weatherball', 'zapcannon', 'syrupbomb',
            }
            _SOUNDPROOF = {
                'boomburst', 'bugbuzz', 'chatter', 'clangingscales',
                'clangoroussoul', 'disarmingvoice', 'echoedvoice', 'growl',
                'howl', 'hypervoice', 'metalsound', 'nobleroar', 'overdrive',
                'partingshot', 'perishsong', 'relicsong', 'roar', 'round',
                'screech', 'shadowpanic', 'sing', 'snarl', 'snore', 'sparklingaria',
                'supersonic', 'torchsong', 'uproar', 'alluringvoice', 'psychicnoise',
            }
            _mid = getattr(move, "id", "")
            if 'bulletproof' in opp_abilities and _mid in _BULLETPROOF:
                return True
            if 'soundproof' in opp_abilities and _mid in _SOUNDPROOF:
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
            # No poke-env, Recover/Roost etc. continuam sendo MoveCategory.STATUS
            # porque a enum nativa so tem PHYSICAL/SPECIAL/STATUS; no ALFINETE eles
            # sao classificados funcionalmente como HEAL. A imunidade de Dark a
            # Prankster vale apenas para golpes priorizados que AFETAM o adversario,
            # nunca para Recover ou outro status de alvo proprio.
            _target_name = str(getattr(getattr(move, 'target', None), 'name', '') or '').upper()
            _targets_opponent = _target_name not in {
                'SELF', 'ALLY_SIDE', 'ALLY_TEAM', 'ADJACENT_ALLY',
                'ADJACENT_ALLY_OR_SELF', 'ALLIES'
            }
            if active.ability == 'prankster' and 'DARK' in opp_types and _targets_opponent:
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
            # Good as Gold / Magic Bounce sao tratados acima apenas quando o
            # golpe de status realmente mira o adversario. Magic Guard NAO concede
            # imunidade geral a status; apenas evita dano indireto.
            if move.id in ['confuseray', 'swagger'] and any(ab in opp_abilities for ab in ['owntempo', 'oblivious']):
                return True

            if move.status:
                if opponent.status:
                    return True

                # 4b. STATUS QUE ALIMENTA O ADVERSARIO (adicionado 24/08/2026)
                # Erro crasso que o instinto cometia sistematicamente: envenenar um
                # Poison Heal (cura-o 1/8 por turno) ou queimar/envenenar um Guts
                # (sobe-lhe o ataque 50%). Um jogador medio nao faz isto.
                # Tambem cobre o item: Toxic Orb e Flame Orb existem precisamente
                # para auto-aplicar o status que a habilidade converte em beneficio,
                # logo quem os segura QUER ser afetado.
                _st = move.status.name
                _item = str(opponent.item).lower() if opponent.item else ""

                # Poison Heal: qualquer veneno passa a ser cura.
                if _st in ['TOX', 'PSN'] and 'poisonheal' in opp_abilities:
                    return True
                # Guts / Marvel Scale / Quick Feet: QUALQUER status vira bonus.
                if any(ab in opp_abilities for ab in ['guts', 'marvelscale', 'quickfeet']):
                    return True
                # Flare Boost (queimadura -> +SpA) e Toxic Boost (veneno -> +Atk).
                if _st == 'BRN' and 'flareboost' in opp_abilities:
                    return True
                if _st in ['TOX', 'PSN'] and 'toxicboost' in opp_abilities:
                    return True
                # Itens de auto-status: quem os segura quer o status.
                if _st in ['TOX', 'PSN'] and _item == 'toxicorb':
                    return True
                if _st == 'BRN' and _item == 'flameorb':
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

        # ==============================================================
        # 5b. BUFF SOB RISCO DE MORTE (04/09/2026)
        # ==============================================================
        # Um buff e um INVESTIMENTO: paga-se um turno agora para ganhar dano ou
        # resistencia nos turnos seguintes. So compensa se houver turnos
        # seguintes. No HP em que o proximo golpe mata, o turno gasto a bufar e
        # simplesmente OFERECIDO, e o buff morre com o Pokemon.
        #
        # ISTO ACRESCENTA, NAO SUBSTITUI. O filtro 5 continua a podar buffs ja
        # maximizados; a Regra Global 1 continua a despromover BUFF em situacao
        # instavel; o contador `buffs_consecutivos` continua a travar a corrida de
        # buffs. Esta e a condicao que faltava e que nenhuma das tres cobre: o
        # buff LEGITIMO, o primeiro, com o estagio por encher, feito no turno em
        # que ja nao ha tempo para o aproveitar.
        #
        # `se_desconhecido=True` de proposito: sem golpe revelado NAO se assume o
        # pior, senao a abertura ficaria sem buffs nenhuns e o Dragon Dance de
        # turno 1 (jogada normal e correcta) seria podado.
        #
        # NAO se aplica a buffs DEFENSIVOS puros: subir Defesa ou Defesa Especial
        # pode ser precisamente o que faz sobreviver ao golpe seguinte, e poda-los
        # aqui seria proibir a jogada que resolve o problema. So se poda o buff
        # que troca um turno por dano FUTURO (Ataque, Ataque Especial, Velocidade).
        if move.category.name == "STATUS":
            _boosts = getattr(move, 'boosts', None) or getattr(move, 'self_boost', None)
            if _boosts and 'self' in str(getattr(move, 'target', '')).lower():
                _ofensivo = any(_boosts.get(k, 0) > 0 for k in ('atk', 'spa', 'spe'))
                _defensivo = any(_boosts.get(k, 0) > 0 for k in ('def', 'spd'))
                if _ofensivo and not _defensivo:
                    try:
                        if self.physics.sobrevive_a(active, opponent, battle,
                                                    se_desconhecido=True) is False:
                            return True
                    except Exception:
                        pass

        # ==============================================================
        # 5c. CURSE: TECTO LIDO DO POKEMON, NAO DO GOLPE (04/09/2026)
        # ==============================================================
        # O filtro 5 poda buffs maximizados lendo `move.boosts`. Para o Curse isso
        # pode nao chegar: no Showdown os `boosts` do Curse sao aplicados por
        # `onModifyMove` conforme o TIPO de quem usa, e nao se sabe se o poke-env
        # os expoe estaticamente. Se nao expuser, `move.boosts` vem vazio e o
        # filtro 5 fica cego ao Curse.
        #
        # NAO SE ASSUME NADA: le-se o tecto no PROPRIO Pokemon (`active.boosts`),
        # que e o dado que existe de certeza. Com Ataque e Defesa ambos no maximo,
        # o Curse so entrega o custo (Velocidade a descer) e nenhum ganho.
        #
        # Observado: `curse` em quatro turnos seguidos.
        #
        # SO PARA QUEM NAO E FANTASMA. Para um tipo Fantasma o Curse e OUTRO
        # golpe: corta metade do proprio HP e amaldicoa o adversario. Nao tem
        # tecto de estagios e esta regra nao se lhe aplica.
        if move.id == 'curse':
            try:
                _meus_tipos = [t.name for t in (getattr(active, "types", None) or []) if t]
                if 'GHOST' not in _meus_tipos:
                    if (active.boosts.get('atk', 0) >= 6
                            and active.boosts.get('def', 0) >= 6):
                        return True
            except Exception:
                pass

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
            my_side = nomes_de_enum(battle.side_conditions.keys())
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
            my_side = nomes_de_enum(battle.side_conditions.keys())
            opp_side = nomes_de_enum(battle.opponent_side_conditions.keys())
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
                # CORRIGIDO (30/08/2026): lia `prev_action` E `last_action` com OR, o
                # que banía o Protect por DOIS turnos em vez de um.
                #
                # O contador de Protect ZERA assim que se usa outro golpe com sucesso.
                # Logo Protect -> ataque -> Protect tem 100% de sucesso, e era
                # exatamente essa alternância que a versão antiga proibia. Não é caso
                # de nicho: é a técnica base de desgaste com Toxic, Leftovers e Poison
                # Heal (Gliscor, Toxapex, Blissey). O instinto evitava um erro real
                # (repetir, ~33% de sucesso) ao custo de proibir a jogada certa
                # (alternar, 100%).
                #
                # Só `last_action`, o turno IMEDIATAMENTE anterior, é a leitura certa:
                # é nesse caso, e só nesse, que o uso seguinte cai para 1/3.
                last_act = history.get('last_action')
                str_last = str(last_act[0]) if isinstance(last_act, tuple) else str(last_act)
                protegeu_no_turno_anterior = "PROTECT" in str_last

                # O contador também ZERA ao trocar de Pokémon: quem acaba de entrar
                # nunca usou Protect. Sem esta guarda o filtro erra em dois casos:
                #
                #   1. troca FORÇADA (pós-faint). Nem o TabularAgent nem o InstinctBot
                #      escrevem history nesse turno (comportamento igual nos dois, ver
                #      6.37), logo `last_action` fica com o valor de DOIS turnos antes
                #      e o Pokémon que entra perde o Protect no seu primeiro turno.
                #   2. troca voluntária seguida de Protect, que é legítima e comum.
                #
                # `last_active_id` a None (turno 1 da batalha) conta como "trocou":
                # ninguém usou Protect ainda, e o primeiro uso tem sempre 100%.
                mesmo_pokemon = (history.get('last_active_id') is not None
                                 and history.get('last_active_id') == getattr(active, 'species', None))

                if protegeu_no_turno_anterior and mesmo_pokemon:
                    return True

        # 7b. HAZARD JA POSTO (30/08/2026)
        #
        # `is_hazard_already_set` existia e era consultada em DOIS sitios
        # especiais: `get_available_actions` (l.96) e o bloco de suporte do
        # executor (execution.py l.628). Nao estava no `is_move_useless`, que e o
        # filtro que TODOS os caminhos consultam.
        #
        # Depender de dois chamadores se lembrarem de aplicar uma regra e como
        # esta escrito o resto dos defeitos deste projeto: basta um caminho novo
        # esquecer-se, e a regra fica inerte sem dar erro. Posta aqui, vale para
        # todos os caminhos, presentes e futuros.
        #
        # As duas chamadas antigas ficam: sao redundantes e nao fazem mal.
        if self.is_hazard_already_set(move, battle):
            return True

        # 7c. TRICK / SWITCHEROO (30/08/2026)
        #
        # LOOP OBSERVADO EM BATALHA MANUAL:
        #   turno 5   Latios usa Trick   -> Latios fica com Heavy-Duty Boots
        #   turno 6   Latios usa Trick   -> Latios fica com Choice Scarf outra vez
        # Dois turnos gastos para voltar ao ponto de partida, e continuaria assim
        # enquanto ninguem trocasse.
        #
        # A GUARDA E SOBRE O VALOR DA TROCA, e nao sobre "ja usei". O Trick tem UMA
        # finalidade competitiva: passar um item que ATRAPALHA (Choice, Iron Ball,
        # Toxic Orb) a um alvo que sofre com ele, tipicamente uma parede. Se o nosso
        # item nao atrapalha, dar-lho e presente; se o dele tambem atrapalha, e trocar
        # lixo por lixo. Nos dois casos nao ha nada a ganhar, e e isso que fecha o
        # loop sem precisar de guardar estado por batalha.
        #
        # No caso observado, apos o turno 5 o Latios passou a segurar Heavy-Duty
        # Boots, que nao esta na lista: o segundo Trick fica podado.
        if move.id in ('trick', 'switcheroo'):
            meu = str(getattr(active, "item", "") or "").lower().replace(" ", "").replace("-", "")
            dele = str(getattr(opponent, "item", "") or "").lower().replace(" ", "").replace("-", "")
            if not meu:
                return True                                   # nada para dar
            if not self.physics.item_e_removivel(active):
                return True                                   # o nosso nao sai (mega stone, Sticky Hold)
            if dele and not self.physics.item_e_removivel(opponent):
                return True                                   # o dele nao sai: o golpe falha
            if meu not in self.ITENS_QUE_ATRAPALHAM:
                return True                                   # o nosso item e bom: dar seria presente
            if dele in self.ITENS_QUE_ATRAPALHAM:
                return True                                   # lixo por lixo

        # 7d. SKILL SWAP / ENTRAINMENT / ROLE PLAY (30/08/2026)
        #
        # LOOP OBSERVADO EM BATALHA MANUAL:
        #   turno 1   Ribombee usa Skill Swap  -> fica com Drought, da Shield Dust
        #   turno 2   Ribombee usa Skill Swap  -> devolve Drought, recebe Shield Dust
        # O turno 1 foi jogada CERTA (roubou o clima). O turno 2 desfez a jogada.
        #
        # A guarda distingue os dois pelo VALOR das habilidades, nao por "ja usei".
        # Skill Swap so vale em dois casos:
        #
        #   GANHAR  a habilidade dele e valiosa e a nossa nao -> trocamos e lucramos
        #   DESPEJAR a nossa habilidade e PREJUDICIAL (Truant, Slow Start) -> passa-se
        #
        # Fora disso e turno perdido, ou pior: dar de graca uma habilidade boa. No
        # caso observado, ao turno 2 a nossa era Drought (valiosa) e a dele Shield
        # Dust (nao): cai no ramo de "dar de graca" e e podado.
        if move.id in ('skillswap', 'entrainment', 'roleplay'):
            minha = str(getattr(active, "ability", "") or "").lower().replace(" ", "")
            dela = str(getattr(opponent, "ability", "") or "").lower().replace(" ", "")
            if not dela or minha == dela:
                return True                                   # desconhecida ou iguais: sem efeito
            if minha in self.HABILIDADES_INTROCAVEIS or dela in self.HABILIDADES_INTROCAVEIS:
                return True                                   # o golpe falha
            ganhamos = dela in self.HABILIDADES_VALIOSAS and minha not in self.HABILIDADES_VALIOSAS
            despejamos = minha in self.HABILIDADES_PREJUDICIAIS
            if not (ganhamos or despejamos):
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
        #
        # O WISH É EXCEÇÃO (29/08/2026). Ao contrário de Recover ou Roost, que curam
        # AGORA, o Wish cura no FIM DO TURNO SEGUINTE — e cura quem estiver em campo
        # nessa altura. Logo tem dois usos legítimos com o ativo a full:
        #
        #   1. antecipar dano: lançar a full e receber a cura já magoado no turno
        #      seguinte, sem gastar o turno em que se está sob pressão
        #   2. curar um ALIADO: lançar e trocar, e quem entra recebe a cura
        #
        # Podá-lo por "HP cheio" removia as duas jogadas, que são precisamente as que
        # justificam levar Wish em vez de Recover. Só se poda quando já há um Wish
        # pendente, porque aí o segundo substitui o primeiro sem ganho.
        # ==============================================================
        # A LISTA E OBRIGATORIA PARA O REST (nota de 05/09/2026)
        # ==============================================================
        # A condicao deste bloco tem duas portas: o `id` estar nesta lista, OU
        # `move.heal` ser positivo. Medido no poke-env instalado:
        #
        #     recover  heal=0.5    -> passa pelas DUAS portas
        #     rest     heal=0.0    -> passa SO por esta lista
        #
        # O Showdown resolve o Rest por `onModifyMove` e o poke-env nao expoe
        # valor estatico (o mesmo acontece com o Curse: `boosts=None`). Logo
        # TIRAR `'rest'` desta lista por parecer redundante DESLIGA o filtro de
        # overheal para ele, em silencio, e volta o caso observado de seis Rest
        # seguidos. Nao e redundante: e a unica porta que ele tem.
        healing_moves = ['recover', 'roost', 'slackoff', 'softboiled', 'milkdrink',
                         'shoreup', 'moonlight', 'morningsun', 'synthesis', 'healorder',
                         'rest']

        if move.id == 'wish':
            # ==========================================================
            # WISH POR HISTORICO PROPRIO (04/09/2026)
            # ==========================================================
            # VERIFICADO EM 04/09: a string "WISH" NAO EXISTE no `pokemon.py` do
            # poke-env, e o Wish tambem nao consta das `SideCondition`. E uma
            # SLOT condition, e a biblioteca nao a expoe de todo. As duas
            # consultas abaixo NUNCA poderiam encontrar nada — nem depois da
            # correccao dos enums da 6.46, que arrumou o NOME mas nao inventa um
            # dado que nao existe.
            #
            # Observado: Wish nos turnos 40, 42, 43, 44 e 45 da mesma batalha,
            # com "But it failed!" no 43.
            #
            # O Wish resolve DOIS turnos depois de ser lancado. Enquanto um
            # estiver pendente, outro nao pega. `wish_turno` e escrito pelo
            # `_atualizar_history` do agente e pelo do InstinctBot, com as mesmas
            # chaves — a paridade de historico entre os tres agentes e o que
            # garante que esta poda vale igual para todos.
            if self.physics.tem(battle.side_conditions, "WISH"):
                return True
            if self.physics.tem(getattr(active, "effects", None), "WISH"):
                return True
            try:
                lancado = (history or {}).get('wish_turno')
                if lancado is not None and (battle.turn - int(lancado)) < 2:
                    return True
            except Exception:
                pass
            return False

        # ==============================================================
        # FUTURE SIGHT E DOOM DESIRE: EFEITO DIFERIDO (04/09/2026)
        # ==============================================================
        # Mesma familia do Wish e pela mesma razao: sao SLOT conditions e o
        # poke-env nao as expoe. Enquanto um ataque diferido estiver pendente,
        # outro NAO pega.
        #
        # ARITMETICA, e ela e diferente da do Wish. Lancado no turno N, resolve no
        # FIM do turno N+2. Fica pendente em N+1 e N+2, e so em N+3 volta a poder
        # ser usado. O Wish resolve em N+1, dai o limiar de 2 la em cima e de 3
        # aqui.
        #
        # Observado: `futuresight` nos turnos 19 a 22 da mesma batalha, com
        # "But it failed!" nos turnos 21 e 22 e o ataque a cair no fim do 22.
        if move.id in ('futuresight', 'doomdesire'):
            try:
                lancado = (history or {}).get('diferido_turno')
                if lancado is not None and (battle.turn - int(lancado)) < 3:
                    return True
            except Exception:
                pass

        # CORRIGIDO: mesma questão de enum-vs-string do Filtro 9. Agora a segunda
        # condição (qualquer golpe de status com heal > 0) também dispara, cobrindo
        # golpes de cura que não estejam na lista explícita healing_moves.
        # ==============================================================
        # REST: CURA QUE COBRA DOIS TURNOS (04/09/2026)
        # ==============================================================
        # O Rest cura tudo, mas ADORMECE por dois turnos. E por isso a cura mais
        # cara do jogo, e nao pode ter o mesmo limiar das outras.
        #
        # Observado em batalha manual: `rest` em SEIS turnos seguidos. Escapava a
        # este bloco porque `classify_move` o devolvia como STATUS (corrigido em
        # `physics` na mesma data) e porque nao constava de `healing_moves`.
        #
        # TRES REGRAS, da mais absoluta para a mais situacional:
        if move.id == 'rest':
            # 1. JA A DORMIR: repetir falha. Nao e opiniao, e regra do jogo.
            if str(getattr(getattr(active, "status", None), "name", "")).upper() == "SLP":
                return True

            # 2. COM PLANO PARA ACORDAR, o Rest deixa de custar os dois turnos e
            #    passa a valer como uma cura normal. Sleep Talk age a dormir;
            #    Chesto Berry acorda no proprio turno; Early Bird e Shed Skin
            #    encurtam o sono. Foi o Sleep Talk que motivou esta excepcao.
            try:
                tem_plano = (
                    any(getattr(m, "id", "") in ("sleeptalk", "snore")
                        for m in (getattr(battle, "available_moves", None) or []))
                    or str(getattr(active, "item", "") or "").lower().replace(" ", "") == "chestoberry"
                    or str(getattr(active, "ability", "") or "").lower().replace(" ", "")
                    in ("earlybird", "shedskin")
                )
            except Exception:
                tem_plano = False

            # 3. SEM PLANO, so vale quando o estrago ja e grande: dois turnos
            #    indefeso por uma cura pequena e mau negocio. Com plano, cai no
            #    limiar normal das outras curas, mais abaixo.
            if not tem_plano and active.current_hp_fraction > 0.50:
                return True

        if move.id in healing_moves or (getattr(move, 'heal', 0) and move.category.name == "STATUS"):
            if active.current_hp_fraction >= 0.95:
                # ======================================================
                # A UNICA EXCEPCAO AO OVERHEAL (04/09/2026)
                # ======================================================
                # Curar com a vida cheia e, por definicao, um turno deitado fora.
                # Ha UM caso em que nao e, e e o relogio do desgaste:
                #
                #   o adversario esta a PERDER vida por turno (veneno, queimadura,
                #   Leech Seed, Salt Cure), somos MAIS LENTOS, e ele mata-nos em
                #   DOIS golpes mas nao em um.
                #
                # Nesse cenario a batalha e uma corrida entre o relogio dele e o
                # nosso sustento. Sendo lentos, levamos o golpe ANTES de agir: se
                # atacarmos agora, chegamos ao turno seguinte a meio e o segundo
                # golpe mata. Curar mantem a margem de dois golpes indefinidamente
                # enquanto o residual o consome. E a jogada do Toxapex e do
                # Gliscor, e sem esta excepcao o instinto nunca a poderia fazer.
                #
                # AS TRES CONDICOES SAO CUMULATIVAS de proposito. Sem residual, a
                # corrida nao tem fim. Sendo rapidos, ataca-se e cura-se depois.
                # Se um golpe ja mata, curar nao salva nada.
                try:
                    residual = (
                        str(getattr(getattr(opponent, "status", None), "name", "")).upper()
                        in ("PSN", "TOX", "BRN")
                        or self.physics.tem(getattr(opponent, "effects", None),
                                            "LEECH_SEED", "LEECHSEED", "CURSE", "SALT_CURE")
                    )
                    mais_lentos = not self.physics.mais_rapido(active, opponent, battle)
                    dano = self.physics.dano_maximo_conhecido(opponent, active, battle)
                    hp = active.current_hp_fraction
                    dois_matam = 0.0 < dano < hp <= dano * 2.0
                    if residual and mais_lentos and dois_matam:
                        return False
                except Exception:
                    pass
                return True
        # 10b. CURA QUE PERDE A CORRIDA (30/08/2026)
        #
        # O filtro 10 chama-se "overheal" e e so isso: bloqueia curar com vida cheia.
        # Nada bloqueava curar quando a CURA E MENOR QUE O DANO QUE SE VAI LEVAR.
        #
        # Observado em batalha manual, com o ULTIMO Pokemon e desvantagem de tipo:
        #
        #   turno 27   Roost cura 50%   Moonblast tira 60%
        #   turno 28   Roost cura 50%   Moonblast tira 58%
        #
        # Curou duas vezes e ficou PIOR do que comecou (32%). Cada ciclo perdia 8 a
        # 10 pontos liquidos e gastava um turno que podia ter sido dano.
        #
        # QUANDO CURAR NUMA CORRIDA PERDIDA AINDA E CERTO: quando a vida do
        # ADVERSARIO tambem esta a descer. Ai a corrida nao e perdida, e uma corrida —
        # e ganha-se com quem tiver mais sustento. Fora disso, curar contra dano maior
        # que a cura so alonga a batalha para o mesmo desfecho.
        #
        # Duas fontes entram na conta do NOSSO lado:
        #   - a fracao curada pelo golpe
        #   - o residual do item (Leftovers e Black Sludge ~1/16; Toxic Orb com
        #     Poison Heal ~1/8)
        # E uma condicao ANULA o filtro:
        #   - o adversario a perder vida por status, Leech Seed ou Curse
        #
        # O dano usa os golpes JA REVELADOS. Sem golpe conhecido nao se poda: seria
        # decidir sobre um numero que nao existe.
        if move.id in healing_moves or (getattr(move, 'heal', 0) and move.category.name == "STATUS"):
            try:
                # `fracao_de_cura` e nao `move.heal`: Synthesis num granizo cura
                # 1/4 e nao 1/2, e Shore Up numa areia cura 2/3. Usar o valor
                # fixo faria a corrida parecer ganha quando esta perdida.
                cura = self.physics.fracao_de_cura(move, battle)

                item_act = str(getattr(active, "item", "") or "").lower()
                abil_act = str(getattr(active, "ability", "") or "").lower()
                residual = 0.0
                if item_act in ('leftovers', 'blacksludge'):
                    residual = 1.0 / 16.0
                elif item_act in ('toxicorb', 'flameorb') and abil_act == 'poisonheal':
                    residual = 1.0 / 8.0

                # MIGRADO PARA `physics.tem` EM 03/09/2026: Leech Seed e Curse
                # nunca eram lidos, logo "o adversario ja esta a perder vida"
                # dependia so do status.
                opp_a_perder = bool(getattr(opponent, "status", None)) or \
                    self.physics.tem(getattr(opponent, "effects", None),
                                     "LEECH_SEED", "LEECHSEED", "CURSE")

                golpes_conhecidos = [m for m in opponent.moves.values() if m.base_power > 0]
                if golpes_conhecidos and not opp_a_perder:
                    pior = max(self.physics.estimate_damage_percent(m, opponent, active, battle)
                               for m in golpes_conhecidos)
                    if pior >= (cura + residual):
                        return True
            except Exception:
                pass

        if move.id in ['aromatherapy', 'healbell', 'junglehealing']:
            team_needs_heal = any(m.status is not None and not m.fainted for m in battle.team.values())
            if not team_needs_heal:
                return True

            # 10c. NAO DESLIGAR O PROPRIO BONUS (adicionado 24/08/2026)
            # Se o ativo tem Guts/Quick Feet/Marvel Scale/Poison Heal e esta afetado,
            # o status E o bonus dele. Curar so compensa se HOUVER OUTROS aliados
            # afetados; se o ativo e o unico, curar e auto-sabotagem.
            _abi = str(active.ability).lower() if active.ability else ""
            if active.status and _abi in ['guts', 'quickfeet', 'marvelscale', 'poisonheal']:
                outros_afetados = [m for m in battle.team.values()
                                   if m.status is not None and not m.fainted and m is not active]
                if not outros_afetados:
                    return True

        # 11. TERRENOS que bloqueiam status (Misty impede todos, Electric impede sleep)
        # NOTA: a verificação redundante `if opponent.status: return True` foi REMOVIDA
        # daqui — já é feita no Filtro 4. Mantém-se apenas a lógica única de terrenos.
        if move.status:
            active_fields = nomes_de_enum(battle.fields.keys())
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

        # 13. CAMPOS JÁ ATIVOS (Trick Room, terrenos, Gravity, Magic/Wonder Room)
        #
        # ADICIONADO 30/08/2026. Buraco com a MESMA FORMA do bug do Stealth Rock
        # (6.35, bug 3): a regra de "já está posto" existia para side conditions
        # (filtro 6.5) e para clima (filtro 6), mas os CAMPOS não são nem uma coisa
        # nem outra. Vivem em `battle.fields` e não passavam por filtro nenhum.
        #
        # Passou a importar agora por causa do item 17 da 6.31, que deu ao Trick Room
        # caminho próprio no `_mod_lead`, Árvore 0 na escolha de lead e +400 na
        # pontuação de troca. O instinto foi ensinado a QUERER Trick Room sem nada
        # que o impedisse de o repor com ele já em campo.
        #
        # A MECÂNICA NÃO É A MESMA PARA TODOS, e a diferença importa:
        #
        #   TRICK ROOM / MAGIC ROOM / WONDER ROOM  ->  ALTERNAM. Usar com o campo
        #       ativo não falha: DESLIGA-O. É pior do que perder o turno, porque
        #       desfaz a própria vantagem. Este é o caso grave.
        #   TERRENOS / GRAVITY                     ->  FALHAM se o MESMO já está
        #       ativo. Perde-se o turno e mais nada. Substituir um terreno DIFERENTE
        #       é legítimo, por isso a comparação é por campo específico e não por
        #       "existe algum campo ativo".
        #
        # LIMITE DECLARADO: o poke-env não diz de QUEM é o campo. Assume-se que um
        # Trick Room ativo nos serve, porque só se chega aqui se o Pokémon em campo
        # souber Trick Room, e uma equipa que o carrega é uma equipa lenta, que ganha
        # com ele esteja ativo por quem estiver. O caso contrário (equipa rápida a
        # usar Trick Room para CANCELAR o do adversário) fica de fora de propósito:
        # é jogada de jogador experiente, e o instinto representa um jogador médio.
        CAMPOS_POR_GOLPE = {
            'trickroom': 'TRICKROOM',
            'magicroom': 'MAGICROOM',
            'wonderroom': 'WONDERROOM',
            'electricterrain': 'ELECTRICTERRAIN',
            'grassyterrain': 'GRASSYTERRAIN',
            'mistyterrain': 'MISTYTERRAIN',
            'psychicterrain': 'PSYCHICTERRAIN',
            'gravity': 'GRAVITY',
        }
        campo_alvo = CAMPOS_POR_GOLPE.get(move.id)
        if campo_alvo:
            # Mesma normalização defensiva do bug do Stealth Rock: conforme a versão
            # do poke-env e o caminho de parsing, a chave pode vir como
            # `Field.TRICK_ROOM` ("TRICK_ROOM") ou como "trickroom" ("TRICKROOM").
            # Comparar sem underscores cobre as duas formas. Foi tratar só uma delas
            # que deixou o filtro de hazards inerte durante semanas.
            # MIGRADO PARA `.name` EM 03/09/2026: um campo JA activo nunca era
            # detectado, logo o instinto repunha Trick Room e terrenos por cima
            # de si proprios.
            campos_ativos = {n.replace("_", "")
                             for n in self.physics.nomes_de(battle.fields)}
            if campo_alvo in campos_ativos:
                return True

        return False
