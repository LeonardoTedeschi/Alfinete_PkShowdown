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

    def get_role(self, pokemon, battle=None) -> Role:
        """Classifica o papel do Pokémon a partir dos base stats.

        `battle` e opcional. Quando dado E houver TRICK ROOM ativo, a contribuicao da
        velocidade para a pontuacao ofensiva e INVERTIDA: sob Trick Room quem e lento
        age primeiro, logo um Pokemon lento e MAIS sweeper e um rapido e menos.
        Sem `battle` o resultado e o de antes, e nenhum chamador antigo muda.
        """
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
        # INVERSAO SOB TRICK ROOM (30/08/2026). A velocidade entra na pontuacao
        # ofensiva porque agir primeiro e o que transforma poder em sweep. Sob Trick
        # Room a relacao inverte-se, e usar `b_spe` cru diria que o Pokemon lento que
        # o Trick Room existe para habilitar e o MENOS ofensivo dos dois.
        #
        # 255 e o teto de velocidade base do jogo; a inversao e por espelho, para a
        # escala e o peso de 0.35 continuarem os mesmos.
        spe_efetiva = b_spe
        try:
            if battle is not None and self.turnos_de_campo(battle, "TRICK_ROOM") is not None:
                spe_efetiva = 255 - b_spe
        except Exception:
            spe_efetiva = b_spe

        ofensiva = max(b_atk, b_spa) + 0.35 * spe_efetiva
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

    # ==================================================================
    # VELOCIDADE E O CAMPO (30/08/2026)
    # ==================================================================
    # ATE AQUI `_get_speed_mod` SO CONHECIA PARALISIA E BOOSTS DE ESTAGIO, e nem
    # sequer recebia `battle`: era ESTRUTURALMENTE INCAPAZ de saber que clima ou que
    # campo estava ativo. Faltavam nove fontes, todas comuns no pool.
    #
    # O sintoma que levou ao diagnostico: em batalha manual foi possivel ultrapassar
    # Pokemon do InstinctBot que tinham Swift Swim com chuva em campo, ou seja
    # velocidade dobrada, e que na conta do instinto continuavam com a velocidade
    # base.
    #
    # A INCOERENCIA INTERNA QUE ISTO CRIAVA. O `shared/state.py` JA DETETA tudo isto
    # e devolve `FIELD_SPEED` na dimensao 5 do estado. A dimensao 6 (`speed_tier`)
    # sai desta funcao e dizia o contrario. As duas dimensoes do MESMO estado
    # contradiziam-se, e os agentes aprendiam sobre uma percepcao inconsistente.
    #
    # E a `policy.py` ja tinha a tabela `ABUSADORES_POR_CLIMA` com `swiftswim` la
    # dentro, usada para decidir se vale a pena POR chuva. Ou seja: o instinto sabia
    # que Swift Swim existe para decidir o clima, e nao sabia que dobra a velocidade
    # na hora de decidir quem bate primeiro. As duas metades da mesma informacao
    # viviam em ficheiros diferentes e nao se falavam.
    #
    # TRICK ROOM NAO ENTRA AQUI, DE PROPOSITO. Ele nao multiplica velocidade nenhuma:
    # INVERTE A ORDEM DE ACAO. Um multiplicador nao consegue exprimir isso (nao ha
    # numero que transforme "maior ganha" em "menor ganha"). Fica no comparador
    # `mais_rapido` abaixo, que e o unico sitio onde a inversao faz sentido.
    #
    # `battle=None` por omissao mantem o comportamento antigo em qualquer chamador
    # que ainda nao passe contexto, tal como se fez no `classify_move` (6.31 item 19).

    # x2 por habilidade, condicionadas ao clima. Sao as quatro do mesmo desenho.
    VELOCIDADE_POR_CLIMA = {
        'swiftswim':   ("RAINDANCE", "PRIMORDIALSEA"),
        'chlorophyll': ("SUNNYDAY", "DESOLATELAND"),
        'sandrush':    ("SANDSTORM",),
        'slushrush':   ("HAIL", "SNOW", "SNOWSCAPE"),
    }
    # x2 por habilidade, condicionadas ao terreno.
    VELOCIDADE_POR_TERRENO = {'surgesurfer': "ELECTRIC_TERRAIN"}

    # x1.5, e SO se a Velocidade for o atributo base mais alto (fora HP). As duas
    # habilidades de paradoxo impulsionam o MAIOR atributo, e para os outros o
    # multiplicador seria 1.3 e nao 1.5 — por isso a condicao do atributo importa.
    # Ativam por campo/clima OU por Booster Energy.
    IMPULSO_PARADOXO = {
        'quarkdrive':     ("ELECTRIC_TERRAIN",),
        'protosynthesis': ("SUNNYDAY", "DESOLATELAND"),
    }

    # Itens que CORTAM velocidade a metade. O Choice Scarf (x1.5) vive no bloco de
    # itens do `estimate_stat`, junto com os outros itens de atributo.
    ITENS_QUE_ATRASAM = {
        'ironball', 'machobrace', 'poweranklet', 'powerband', 'powerbelt',
        'powerbracer', 'powerlens', 'powerweight',
    }

    # ======================================================================
    # LEITURA DE ENUMS DO POKE-ENV — A CORRECCAO DE RAIZ (03/09/2026)
    # ======================================================================
    # ISTO ANULA E SUBSTITUI A CORRECCAO DE 28/08 (6.16). A premissa dela estava
    # ERRADA, e por isso ela nunca funcionou.
    #
    # O comentario antigo dizia que `str(SideCondition.REFLECT)` devolve
    # "SIDECONDITION.REFLECT" e que bastava `.split(".")[-1]`. NAO E VERDADE. O
    # poke-env define um `__str__` PROPRIO em TODOS os seus enums:
    #
    #     str(SideCondition.STEALTH_ROCK) -> 'STEALTH_ROCK (side condition) object'
    #     str(Field.TRICK_ROOM)           -> 'TRICK_ROOM (field) object'
    #     str(Weather.RAINDANCE)          -> 'RAINDANCE (weather) object'
    #     str(Effect.LEECH_SEED)          -> 'LEECH_SEED (effect) object'
    #     str(Status.BRN)                 -> 'BRN (status) object'
    #
    # NAO HA PONTO NENHUM na string, logo o `.split(".")[-1]` nao faz NADA e o
    # resultado da normalizacao antiga era:
    #
    #     'STEALTH_ROCK (SIDE CONDITION) OBJECT'
    #
    # Consequencia: TODO teste por PERTENCA (`'REFLECT' in lista`) e SEMPRE
    # FALSO. Foram 22 sitios em cinco ficheiros, e o comportamento que cada um
    # devia produzir NUNCA correu desde o inicio do projeto. Nao e regressao: e
    # um defeito que a correccao de 28/08 julgou ter fechado e nao fechou.
    #
    # Confirmado em 03/09 por tres vias independentes: o comando de introspeccao
    # ao poke-env instalado, uma reproducao em sandbox da normalizacao antiga, e
    # o log da sonda [HAZ] numa batalha manual, que mostrou
    # `raw={<SideCondition.STEALTH_ROCK: 19>: 14}` com `normalizado={}`.
    #
    # A FORMA CORRECTA E `.name`, que e o unico atributo que o poke-env nao
    # redefine. Vive aqui, num sitio so, pela mesma razao do `mais_rapido` e do
    # `sobrevive_a`: e leitura do MOTOR DO JOGO, nao estrategia.
    #
    # ARMADILHA A EVITAR NA MIGRACAO: a alternativa "obvia" de testar por
    # SUBCADEIA (`any('SPIKES' in x for x in lista)`) funciona por acidente com a
    # string antiga, MAS da FALSO POSITIVO — 'SPIKES' esta contido em
    # 'TOXIC_SPIKES'. Por isso a migracao e para pertenca EXACTA sobre `.name`, e
    # nao para subcadeia.

    # ======================================================================
    # A EQUIPA ADVERSARIA: PREVIEW vs REVELADOS (04/09/2026)
    # ======================================================================
    # PORQUE EXISTE. `_vale_a_pena_hazard` contava `vivos=1` no TURNO 1, com o
    # adversario a ter seis. A contagem lia `battle.opponent_team` e assumia que
    # ele continha a equipa toda desde o team preview.
    #
    # AS DUAS EVIDENCIAS DISCORDAM, e por isso isto e escrito para funcionar nos
    # dois casos em vez de escolher um:
    #   - o codigo do poke-env associa `opponent_team` ao
    #     `_teampreview_opponent_team`, o que daria SEIS desde o turno 1
    #   - o log mede `vivos` a oscilar 1, 2, 2, 1, que e comportamento de
    #     REVELADOS MENOS DESMAIADOS, nao de uma equipa fixa de seis
    #
    # A resolucao usa cada fonte para o que ela sabe de certeza:
    #   TOTAL   do team preview (ou dos revelados, se o preview vier vazio)
    #   MORTOS  dos revelados, que sao os unicos objectos que o `fainted` actualiza
    #
    # Os objectos do preview sao INSTANCIAS DIFERENTES das da batalha, logo o
    # `fainted` deles pode nunca mudar. Contar mortos por ai daria sempre zero.

    @staticmethod
    def equipa_adversaria(battle):
        """(total_vivos, lista_de_candidatos_no_banco) do lado adversario.

        `total_vivos` inclui o activo. Os candidatos excluem o activo e os que ja
        desmaiaram, e vem preferencialmente do PREVIEW, para o preditivo poder
        pontuar contra a equipa toda desde o turno 1 e nao so contra o que ja foi
        revelado.
        """
        try:
            preview = list(getattr(battle, "teampreview_opponent_team", None) or [])
            revelados = dict(getattr(battle, "opponent_team", None) or {})
            activo = getattr(battle, "opponent_active_pokemon", None)
            especie_activa = str(getattr(activo, "species", "") or "").lower()

            mortos = {str(getattr(m, "species", "") or "").lower()
                      for m in revelados.values() if getattr(m, "fainted", False)}

            fonte = preview if preview else list(revelados.values())
            vistos, banco = set(), []
            for m in fonte:
                esp = str(getattr(m, "species", "") or "").lower()
                if not esp or esp in vistos:
                    continue
                vistos.add(esp)
                if esp in mortos:
                    continue
                if esp == especie_activa:
                    continue
                banco.append(m)

            vivos = len(vistos - mortos)
            return max(vivos, 1), banco
        except Exception:
            # Na duvida NAO se assume equipa vazia: assumir vazio desliga o
            # preditivo e a regra do hazard, que foi exactamente o defeito.
            try:
                banco = [m for m in (battle.opponent_team or {}).values()
                         if not m.fainted and not m.active]
                return max(len(banco) + 1, 1), banco
            except Exception:
                return 6, []

    @staticmethod
    def nomes_de(colecao):
        """Nomes dos membros de enum de uma coleccao do poke-env, em MAIUSCULAS.

        Aceita dict, lista ou set. Devolve um `set`, para o teste de pertenca ser
        O(1) e para nao haver duplicados. Membros sem `.name` caem no `str()`, que
        cobre o caso de a coleccao ja vir com strings.
        """
        try:
            return {str(getattr(k, "name", k)).upper() for k in (colecao or {})}
        except Exception:
            return set()

    @staticmethod
    def tem(colecao, *nomes):
        """Algum destes nomes esta na coleccao? Pertenca EXACTA, nunca subcadeia."""
        try:
            presentes = GamePhysics.nomes_de(colecao)
            return any(str(n).upper() in presentes for n in nomes)
        except Exception:
            return False

    @staticmethod
    def _nomes_de_campo(colecao):
        """Nomes de membros de enum sem o prefixo da classe.

        `str(Field.ELECTRIC_TERRAIN).upper()` da "FIELD.ELECTRIC_TERRAIN". Sem o
        `.split(".")[-1]`, qualquer teste por PERTENCA da sempre False. E o bug que
        ja apareceu em policy, masking e state; aqui usa-se a forma corrigida de raiz.
        """
        try:
            # MIGRADO PARA `.name` EM 03/09/2026: ver `nomes_de` acima. A forma
            # antiga devolvia 'TRICK_ROOM (FIELD) OBJECT' e matava todos os
            # testes por pertenca que consomem esta lista.
            return sorted(GamePhysics.nomes_de(colecao))
        except Exception:
            return []

    def _tem_habilidade(self, pokemon, nome, condicao_ativa):
        """O Pokemon tem (ou pode ter) a habilidade `nome`?

        DUAS VIAS, e a segunda e CONDICIONADA (30/08/2026):

          CONFIRMADA      `pokemon.ability` esta preenchida -> basta comparar.
          NAO REVELADA    cai para `possible_abilities`, mas SO quando a condicao
                          que ativaria a habilidade JA ESTA EM CAMPO.

        A condicao existe para nao presumir o pior de graca. Swift Swim, Chlorophyll,
        Sand Rush e Slush Rush sao SILENCIOSAS: nao emitem mensagem ao ativar, logo a
        habilidade do adversario pode ficar por revelar a batalha inteira. Enquanto
        nao ha chuva, atribuir Swift Swim a um Pokemon que tem tres habilidades
        possiveis seria inventar uma ameaca que nao existe naquele turno. Com chuva em
        campo, e a leitura prudente e alinhada com a convencao do projeto de presumir
        investimento maximo no adversario.

        Consequencia deliberada: a mesma especie e lida de forma diferente conforme o
        campo. E o que se quer — a habilidade so importa quando pode agir.
        """
        hab = str(getattr(pokemon, "ability", "") or "").lower()
        if hab:
            return hab == nome
        if not condicao_ativa:
            return False
        possiveis = {str(a).lower() for a in (getattr(pokemon, "possible_abilities", None) or [])}
        return nome in possiveis

    def _spe_e_o_maior(self, pokemon):
        """A Velocidade e o atributo base mais alto (excluindo HP)?

        Criterio das habilidades de paradoxo: elas impulsionam o MAIOR atributo, com
        x1.5 para Velocidade e x1.3 para os restantes. Sem esta verificacao
        aplicar-se-ia x1.5 a Velocidade de um Pokemon cujo impulso foi para o Ataque.
        """
        base = getattr(pokemon, "base_stats", None) or {}
        spe = base.get('spe', 0)
        outros = [v for k, v in base.items() if k not in ('hp', 'spe')]
        return bool(outros) and spe > max(outros)

    def _get_speed_mod(self, pokemon, battle=None):
        """Modificador de velocidade: paralisia, boosts, clima, terreno e Tailwind."""
        mod = 1.0
        # MIGRADO PARA `.name` EM 03/09/2026. Este funcionava por ACIDENTE: o
        # teste era por SUBCADEIA sobre 'PAR (STATUS) OBJECT'. Passa a pertenca
        # exacta, como todos os outros, para nao restar um estilo diferente no
        # ficheiro — foi ter dois estilos misturados que escondeu o bug original.
        if pokemon.status and getattr(pokemon.status, "name", "").upper() == 'PAR':
            mod *= 0.5
        stage = pokemon.boosts.get('spe', 0)
        if stage > 0:
            mod *= (1 + 0.5 * stage)
        elif stage < 0:
            mod *= (2 / (2 + abs(stage)))

        if battle is None:
            return mod

        try:
            clima = next(iter(battle.weather)).name.upper() if battle.weather else "CLEAR"
            campos = self._nomes_de_campo(getattr(battle, "fields", None))
            item = str(getattr(pokemon, "item", "") or "").lower()
            meu = pokemon in (getattr(battle, "team", {}) or {}).values()

            # --- x2 por habilidade de clima, ou de terreno ---
            # A condicao entra ANTES da habilidade: e ela que autoriza a leitura de
            # `possible_abilities` (ver `_tem_habilidade`). Nenhum Pokemon tem duas
            # destas, mas o `break` torna a exclusividade explicita.
            dobrou = False
            for hab, climas in self.VELOCIDADE_POR_CLIMA.items():
                cond = clima in climas
                if cond and self._tem_habilidade(pokemon, hab, cond):
                    mod *= 2.0
                    dobrou = True
                    break
            if not dobrou:
                for hab, campo in self.VELOCIDADE_POR_TERRENO.items():
                    cond = campo in campos
                    if cond and self._tem_habilidade(pokemon, hab, cond):
                        mod *= 2.0
                        dobrou = True
                        break

            # --- x1.5 por habilidade de paradoxo, so se a Velocidade for o maior ---
            if not dobrou and self._spe_e_o_maior(pokemon):
                for hab, gatilhos in self.IMPULSO_PARADOXO.items():
                    cond = (clima in gatilhos) or any(g in campos for g in gatilhos) \
                           or item == "boosterenergy"
                    if cond and self._tem_habilidade(pokemon, hab, cond):
                        mod *= 1.5
                        break

            # --- SLOW START: x0.5 nos primeiros 5 turnos (so o Regigigas a tem) ---
            # Lido por efeito volatil, que e como o servidor o anuncia. Se a versao do
            # poke-env nao expuser `effects`, o `except` no fim devolve o modificador
            # sem esta parcela, que e o comportamento antigo.
            # MIGRADO PARA `.name` EM 03/09/2026: devolvia
            # 'SLOW_START (EFFECT) OBJECT' e o Slow Start NUNCA foi lido.
            efeitos = self.nomes_de(getattr(pokemon, "effects", None))
            if "SLOW_START" in efeitos or "SLOWSTART" in efeitos:
                mod *= 0.5

            # --- UNBURDEN: x2 depois de perder o item ---
            # SO PARA O NOSSO LADO. Para o adversario, `item is None` significa quase
            # sempre "ainda nao foi revelado" e nao "foi consumido"; aplicar la
            # dobraria a velocidade de meia equipa adversaria sem evidencia nenhuma.
            if meu and not item and str(getattr(pokemon, "ability", "") or "").lower() == "unburden":
                mod *= 2.0

            # --- TAILWIND: x2 no LADO, nao no Pokemon ---
            # Escolhe-se o lado certo pela pertenca a equipa, como ja se faz nas
            # barreiras do calculo de dano.
            lado = battle.side_conditions if meu else battle.opponent_side_conditions
            if "TAILWIND" in self._nomes_de_campo(lado):
                mod *= 2.0
        except Exception:
            # Nunca rebentar a decisao por causa de um modificador: na duvida,
            # devolve-se a velocidade sem contexto, que e o comportamento antigo.
            pass
        return mod

    def mais_rapido(self, a, b, battle=None):
        """`a` age antes de `b` neste turno? UNICO sitio onde o Trick Room entra.

        Trick Room inverte a ordem de acao: com ele ativo, quem tem MENOS velocidade
        age primeiro. Isso nao e um multiplicador, e uma inversao do comparador, e e
        por isso que nao pode viver no `_get_speed_mod`.

        A ironia que motivou isto: o item 17 da 6.31 ensinou o instinto a construir a
        partida a volta do Trick Room, e o `trick_room_ativo` da policy so era usado
        para decidir se valia a pena PO-LO. O instinto gastava o turno a ganhar a
        vantagem e depois continuava a jogar como se fosse lento.

        Empates vao para False (nao somos mais rapidos), que e a leitura conservadora
        e a que a `get_speed_tier` ja fazia.
        """
        va = self.estimate_stat(a, 'spe', battle)
        vb = self.estimate_stat(b, 'spe', battle)
        if battle is not None and "TRICK_ROOM" in self._nomes_de_campo(getattr(battle, "fields", None)):
            return va < vb
        return va > vb

    # -- Estimativa de atributo --------------------------------------------

    def estimate_stat(self, pokemon, stat_name, battle=None):
        """Estima o valor efetivo de um atributo, incluindo boosts, status e itens.
        Usa stats reais quando disponíveis; caso contrário infere a partir do base
        stat e do papel presumido (investimento típico competitivo).

        `battle` e opcional e so e usado para a velocidade (clima, terreno, Tailwind).
        Sem ele o resultado e o de antes de 30/08/2026."""
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
            val *= self._get_speed_mod(pokemon, battle)
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
        elif stat_name == 'spe' and item_str in self.ITENS_QUE_ATRASAM:
            # Iron Ball e a familia Power/Macho Brace cortam a Velocidade a metade.
            # O corte vive aqui, junto do Choice Scarf, porque e modificador de ITEM;
            # o `_get_speed_mod` trata do que depende do CAMPO.
            item_mod = 0.5

        return int(val * item_mod)

    # Acima desta razao atk/spa (ou spa/atk) o Pokemon conta como puro.
    # 1.15 = 15% de vantagem. Abaixo disso e MISTO e o desempate e outro.
    LIMIAR_PUREZA = 1.15

    def _is_physical(self, pokemon):
        """True se o Pokemon e predominantemente FISICO.

        REVISTO EM 28/08/2026. A versao anterior era `atk > spa`, um desempate
        binario: um Pokemon com 110 de Atk e 108 de SpA era classificado como puro
        fisico por 2 pontos. Isso propagava-se para decisoes que dependem de acertar
        no eixo certo:

          - `_mod_tank_vs_*` (policy l.142 e l.432): escolher se a defesa RELEVANTE e
            Def ou SpD, e portanto se vale a pena ficar em campo
          - conversao de buff defensivo (policy l.745): a condicao "a defesa subida e
            a certa para este atacante"
          - leitura de barreiras (policy l.652-655): quem no banco consegue passar por
            Reflect ou Light Screen
          - `get_boost_state` (state.py): a queimadura so conta como DEBUFF em fisicos

        Agora usa-se uma RAZAO. Em caso de duvida (misto) devolve-se True, mantendo o
        comportamento anterior no ramo de decisao — a mudanca so afeta os casos em que
        o SpA domina claramente, que antes podiam ser lidos ao contrario por margem
        minima.

        Ver tambem `perfil_ofensivo`, que devolve os tres casos e deve ser preferido
        em codigo novo.
        """
        if not pokemon:
            return True
        return self.perfil_ofensivo(pokemon) != "SPECIAL"

    def perfil_ofensivo(self, pokemon):
        """'PHYSICAL', 'SPECIAL' ou 'MIXED'.

        MIXED quando nenhum dos dois ataques supera o outro por mais de
        LIMIAR_PUREZA. Um Pokemon misto ataca pelos dois lados, logo NENHUMA das duas
        defesas do adversario e a "certa": quem decidir com base nisto deve saber que
        esta em terreno ambiguo, em vez de receber um True ou False com falsa certeza.
        """
        if not pokemon:
            return "PHYSICAL"
        atk = self.estimate_stat(pokemon, 'atk')
        spa = self.estimate_stat(pokemon, 'spa')
        if spa <= 0:
            return "PHYSICAL"
        if atk >= spa * self.LIMIAR_PUREZA:
            return "PHYSICAL"
        if spa >= atk * self.LIMIAR_PUREZA:
            return "SPECIAL"
        return "MIXED"

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

        # 1b. FACADE (30/08/2026). Dobra a potencia se o utilizador estiver
        # queimado, envenenado ou paralisado. E o par natural do Guts, e faltava:
        # sem ele, o golpe que EXISTE para ser usado com status era estimado como um
        # Body Slam qualquer, e o instinto nunca via razao para manter o Pokemon
        # queimado em campo.
        #
        # Congelamento e sono NAO contam (o Pokemon nem sequer ataca), por isso a
        # lista e explicita em vez de `if attacker.status`.
        # MIGRADO PARA `.name` EM 03/09/2026: devolvia 'BRN (STATUS) OBJECT' e o
        # `in ('BRN', ...)` era SEMPRE falso. O Facade nunca dobrou a potencia.
        status_nome = str(getattr(getattr(attacker, 'status', None), 'name', '') or '').upper()
        if move.id == 'facade' and status_nome in ('BRN', 'PSN', 'TOX', 'PAR'):
            bp *= 2.0

        # 1c. POTENCIA DEPENDENTE DO CLIMA (30/08/2026)
        #
        # Regras de jogo que faltavam. Auditadas contra a referencia do Bulbapedia.
        #
        #   SOLAR BEAM / SOLAR BLADE   metade da potencia em CHUVA, AREIA e GRANIZO.
        #       O codigo ja tratava o SOL, mas so para saltar o turno de carga; a
        #       penalizacao nos outros climas nao existia em lado nenhum. Um Solar
        #       Beam numa areia estava a ser estimado ao DOBRO do real.
        #
        #   WEATHER BALL   muda de TIPO e dobra a potencia conforme o clima. Sem
        #       isto era estimado como um golpe Normal de 50, que e o que ele so e
        #       com o ceu limpo.
        clima_bp = next(iter(battle.weather)).name.upper() if battle and battle.weather else "CLEAR"
        if move.id in ('solarbeam', 'solarblade'):
            if clima_bp in ('RAINDANCE', 'PRIMORDIALSEA', 'SANDSTORM', 'HAIL', 'SNOW', 'SNOWSCAPE'):
                bp *= 0.5
        elif move.id == 'weatherball' and clima_bp not in ('CLEAR', 'NONE'):
            bp *= 2.0

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
        # CONTAGEM DUPLA DO CHOICE, CORRIGIDA 30/08/2026.
        #
        # Havia aqui `if item_str == 'choiceband': atk *= 1.5` (e o par para o
        # Specs). Mas o `estimate_stat` JA aplica esse 1.5 no seu bloco de itens.
        # Resultado: 1.5 x 1.5 = 2.25, ou seja o dano de qualquer atacante com Choice
        # saia 50% inflacionado. Os itens Choice sao dos mais comuns do pool.
        #
        # O 1.5 fica no `estimate_stat`, que e onde vivem os modificadores de
        # ATRIBUTO. Aqui so entram os que dependem do GOLPE ou da situacao.
        if move.category.name == "PHYSICAL":
            atk = self.estimate_stat(attacker, 'atk')
            if attacker_ability in ['hugepower', 'purepower']:
                atk *= 2.0
            if attacker_ability == 'hustle':
                atk *= 1.5

            # QUEIMADURA (30/08/2026). Corta o Ataque FISICO a metade, e nao existia
            # em lado nenhum do ficheiro: um atacante fisico queimado tinha o dano
            # sobrestimado ao DOBRO. Will-O-Wisp e jogada de rotina no pool.
            #
            # GUTS E A EXCECAO, e e dupla: ignora o corte da queimadura E multiplica o
            # Ataque por 1.5 por estar com status. Um Conkeldurr queimado bate MAIS
            # forte, nao menos. Tratar Guts com o corte generico invertia o sinal da
            # leitura no exato Pokemon em que ela mais importa.
            # FACADE tambem IGNORA o corte da queimadura desde a Gen 6. Sem esta
            # excecao o golpe levaria x2 na potencia e x0.5 no ataque, anulando-se, e
            # a estimativa daria exatamente o valor errado: igual ao golpe sem status,
            # quando na verdade e o dobro.
            # MIGRADO PARA `.name` EM 03/09/2026 (funcionava por subcadeia).
            queimado = getattr(getattr(attacker, "status", None), "name", "").upper() == 'BRN'
            if attacker_ability == 'guts' and attacker.status:
                atk *= 1.5
            elif queimado and move.id != 'facade':
                atk *= 0.5

            # Body Press usa a DEFESA e por isso ignora tudo o que mexe no Ataque,
            # incluindo a queimadura. A atribuicao vem depois de proposito.
            if move.id == 'bodypress':
                atk = self.estimate_stat(attacker, 'def')
            defense = self.estimate_stat(defender, 'def')
            stat_defensiva = 'def'
        else:
            atk = self.estimate_stat(attacker, 'spa')
            defense = self.estimate_stat(defender, 'spd')
            stat_defensiva = 'spd'
            if move.id in ['psyshock', 'psystrike', 'secretsword']:
                defense = self.estimate_stat(defender, 'def')
                stat_defensiva = 'def'

        # ==============================================================
        # DEFESA PELO CLIMA (30/08/2026)
        # ==============================================================
        # NAO EXISTIA. Zero ocorrencias de bonus defensivo por clima no ficheiro.
        #
        #   AREIA  -> tipo PEDRA ganha 1.5x de Defesa Especial
        #   NEVE   -> tipo GELO  ganha 1.5x de Defesa fisica (Gen 9)
        #
        # Caso observado em batalha manual: Latias-Mega (SpA 140) contra Regirock com
        # areia em campo. A estimativa usava SpD 100 quando a real era 150, ou seja
        # dava 50% de dano a mais. Isso fez o Ice Beam parecer 1,50x mais forte que o
        # Psyshock quando na verdade EMPATAM — e o desempate entre dois golpes
        # equivalentes caiu num deles por acaso.
        #
        # E o quarto defeito da mesma familia neste calculo (enums das barreiras,
        # Choice contado a dobrar, queimadura ausente), e TODOS empurravam a
        # estimativa PARA CIMA. O padrao importa: o instinto vinha a sobrestimar
        # sistematicamente o proprio dano, o que o torna mais agressivo do que a
        # aritmetica do jogo justifica.
        # O clima e lido AQUI e nao reaproveitado da variavel `weather` mais abaixo:
        # ela so e atribuida na l.566, DEPOIS de `base_dmg` ja ter usado a defesa.
        # Referencia-la aqui daria NameError, que o `except` engoliria — e o bonus
        # nunca se aplicaria, sem erro nenhum. Exatamente o modo de falha que este
        # projeto ja pagou varias vezes.
        try:
            clima_def = next(iter(battle.weather)).name.upper() if battle and battle.weather else "CLEAR"
            tipos_def = [t.name for t in defender.types if t]
            # A condicao e sobre o ATRIBUTO usado, nao sobre a categoria do golpe.
            # A areia sobe a Defesa ESPECIAL do tipo Pedra; o Psyshock e um golpe
            # ESPECIAL que bate na Defesa FISICA, logo nao deve receber o bonus.
            # A primeira versao desta correccao testava `move.category` e dava-lho;
            # o teste apanhou, e e por isso que se testa antes de entregar.
            if clima_def == "SANDSTORM" and "ROCK" in tipos_def and stat_defensiva == "spd":
                defense *= 1.5
            elif clima_def in ("SNOW", "SNOWSCAPE") and "ICE" in tipos_def and stat_defensiva == "def":
                defense *= 1.5
        except Exception:
            pass

        if defense <= 0:
            defense = 1

        base_dmg = ((((2 * level / 5) + 2) * atk * bp / defense) / 50) + 2

        # ==============================================================
        # GOLPES QUE O CLIMA ALTERA (30/08/2026)
        # ==============================================================
        # Regras de jogo que faltavam. Ambas mexem no dano e nenhuma existia.
        #
        # SOLAR BEAM / SOLAR BLADE: potencia a METADE em chuva, areia e granizo. O
        # codigo so tratava o SOL, e apenas para dispensar o turno de carga; em clima
        # adverso o golpe era estimado com potencia cheia, ou seja ao DOBRO do real.
        #
        # WEATHER BALL: muda de TIPO e DOBRA a potencia conforme o clima. Sem isto e
        # estimado como um golpe Normal de 50, que e o unico caso em que ele nao esta.
        tipo_efetivo = move.type
        try:
            if move.id in ('solarbeam', 'solarblade') and clima_def in (
                    "RAINDANCE", "PRIMORDIALSEA", "SANDSTORM", "HAIL", "SNOW", "SNOWSCAPE"):
                bp *= 0.5
            elif move.id == 'weatherball':
                novo = self.TIPO_DO_WEATHER_BALL.get(clima_def)
                if novo:
                    bp *= 2.0
                    # O tipo tem de mudar TAMBEM para o multiplicador e para o STAB.
                    # Procura-se o enum do tipo entre os tipos ja conhecidos da
                    # batalha; se nao aparecer, mantem-se o tipo original e so a
                    # potencia dobra — degradacao suave em vez de excecao.
                    for cand in list(defender.types) + list(attacker.types):
                        if cand is not None and cand.name == novo:
                            tipo_efetivo = cand
                            break
                    else:
                        tipo_efetivo = getattr(type(move.type), novo, move.type)
        except Exception:
            tipo_efetivo = move.type

        # 4. STAB, tipo, item de dano final
        stab_multiplier = 2.0 if attacker_ability == 'adaptability' else 1.5
        stab = stab_multiplier if tipo_efetivo in attacker.types else 1.0
        # MIGRADO PARA `multiplicador_de_tipo` EM 03/09/2026: o
        # `damage_multiplier` resolve a tabela de tipos e ignora as excepcoes que
        # o proprio golpe declara. Freeze-Dry contra Agua era estimado a 0,5x
        # quando o real e 2x — um erro de 4x, na direccao que faz o instinto
        # descartar o golpe que resolve o confronto.
        try:
            type_mod = self.multiplicador_de_tipo(move, defender, tipo_efetivo)
        except Exception:
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
            # CORRIGIDO 30/08/2026 — NONA OCORRENCIA DO BUG DOS ENUMS.
            #
            # `str(SideCondition.REFLECT).upper()` da "SIDECONDITION.REFLECT". Como o
            # teste abaixo e por PERTENCA A LISTA ('REFLECT' in active_screens) e nao
            # por subcadeia, dava SEMPRE False: Reflect, Light Screen e Aurora Veil
            # NUNCA reduziram o dano estimado, desde sempre.
            #
            # A varredura de 6.31 item 1 corrigiu oito ocorrencias em policy.py e
            # masking.py, e state.py corrigiu as suas em 26/08. O physics.py nunca foi
            # varrido, apesar de ser o ficheiro onde o erro custa mais: esta
            # estimativa alimenta letalidade, sacrificio, escolha de troca e o
            # `has_lethal` que poda o overkill do Blue.
            # MIGRADO PARA `.name` EM 03/09/2026. Este era dos piores: REFLECT,
            # LIGHT_SCREEN e AURORA_VEIL NUNCA reduziram a estimativa de dano, em
            # nenhuma versao do projeto. Contra uma equipa de ecrans o instinto
            # jogava como se eles nao existissem.
            active_screens = self.nomes_de(side_to_check)
            if move.category.name == "PHYSICAL" and ('REFLECT' in active_screens or 'AURORA_VEIL' in active_screens):
                margin *= 0.5
            elif move.category.name == "SPECIAL" and ('LIGHT_SCREEN' in active_screens or 'AURORA_VEIL' in active_screens):
                margin *= 0.5

        # 7. CLIMA E TERRENO
        weather_mod = 1.0
        terrain_mod = 1.0
        if battle:
            move_type = move.type.name if move.type else ""
            # WEATHER BALL muda de TIPO com o clima (30/08/2026): numa areia deixa de
            # ser Normal e passa a Pedra, com fraquezas e resistencias diferentes.
            if move.id == 'weatherball':
                move_type = self.TIPO_DO_WEATHER_BALL.get(weather, move_type)

            # ESTRUTURA REPARADA EM 30/08/2026. Uma colagem anterior deixou TODO o
            # bloco de clima e terreno aninhado dentro do `if weatherball`. O efeito
            # foi enorme e silencioso: a chuva deixou de potenciar Agua, o sol deixou
            # de potenciar Fogo, os quatro terrenos e o Sand Force deixaram de existir
            # — para todos os golpes menos o Weather Ball. Compilava, nao dava erro, e
            # so aparecia como dano subestimado em times de clima.
            #
            # CORRIGIDO 30/08/2026 — DECIMA OCORRENCIA DO BUG DOS ENUMS.
            # Os quatro terrenos nunca modificaram o dano estimado pela mesma razao.
            # MIGRADO PARA `.name` EM 03/09/2026: o TERRENO nunca modificou dano.
            # Electric, Grassy, Psychic e Misty Terrain eram todos invisiveis.
            current_fields = self.nomes_de(battle.fields)

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

    # ==================================================================
    # GOLPES DE PRISAO (28/08/2026)
    # ==================================================================
    # Prendem o alvo e causam dano residual durante ~5 turnos. Passam a ATTACK_TECH
    # pelo mesmo criterio que rege a lista: o EFEITO vale mais que a potencia.
    #
    #   Bind 15, Wrap 15, Infestation 20, Fire Spin 35, Whirlpool 35, Sand Tomb 35
    #
    # MAGMA STORM FICA DE FORA, de proposito. Com 100 de potencia e STAB no Heatran e
    # dos golpes mais fortes do pool: classifica-lo como TECH faria o instinto perder
    # o acesso a ele quando quer DANO, porque o _select_best_move_in_category escolhe
    # dentro da categoria por utilidade e nao por potencia.
    GOLPES_DE_PRISAO = {
        'bind', 'wrap', 'infestation', 'firespin', 'whirlpool', 'sandtomb',
    }

    # ==================================================================
    # EFEITO TECH JA GASTO (28/08/2026)
    # ==================================================================
    # Um ATTACK_TECH vale pelo efeito. Depois de o efeito estar aplicado, o golpe
    # passa a ser so o seu dano — e normalmente e dano fraco.
    #
    # O problema concreto: Knock Off com o item ja removido continua a ser escolhido
    # como TECH, porque a categoria nao muda. Um Pokemon com DOIS golpes tech fica a
    # alterna-los indefinidamente, sem nunca usar o ataque forte que tem no moveset.
    #
    # Solucao: quando o efeito ja nao pode aplicar-se, o golpe passa a ATTACK_STRONG
    # e compete pelo DANO, que e o unico valor que ainda tem.
    #
    # So se classificam como gastos os casos VERIFICAVEIS COM CERTEZA. Os secundarios
    # probabilisticos (Scald a queimar, Nuzzle a paralisar) contam como gastos apenas
    # quando o alvo JA TEM status, porque ai o secundario nao pode mesmo aplicar-se.

    # ==================================================================
    # GOLPES DE DRENO (29/08/2026)
    # ==================================================================
    # Curam uma fracao do dano causado. Entram em `tech_moves` pelo mesmo criterio do
    # resto da lista: o EFEITO vale mais que a potencia — o Giga Drain e dos melhores
    # golpes de Grama justamente pela cura, nao pelos 75 de potencia.
    #
    # A fracao real vem de `move.drain` no poke-env. Como nao foi possivel confirmar
    # que esse atributo existe na versao instalada, ha tabela de recurso: quase todos
    # curam 50%, mas Draining Kiss e Oblivion Wing curam 75% (ha 2 Draining Kiss no
    # pool de eval).
    GOLPES_DE_DRENO = {
        'gigadrain', 'drainpunch', 'hornleech', 'leechlife', 'absorb', 'megadrain',
        'drainingkiss', 'paraboliccharge', 'oblivionwing', 'dreameater',
        'bitterblade', 'matchagotcha', 'bouncybubble',
    }
    _DRENO_75 = {'drainingkiss', 'oblivionwing'}

    # ======================================================================
    # EFECTIVIDADE POR TEXTO, NAO POR TIPO (03/09/2026)
    # ======================================================================
    # O `damage_multiplier` do poke-env resolve a tabela de tipos e mais nada. Um
    # punhado de golpes tem, ESCRITO NA PROPRIA DESCRICAO, uma excepcao a essa
    # tabela, e para esses o multiplicador do poke-env esta simplesmente errado:
    #
    #   Freeze-Dry      e GELO, mas e SUPER EFECTIVO contra AGUA.
    #                   poke-env diz 0,5x; o real e 2x. ERRO DE 4x, na direccao
    #                   que faz o instinto descartar exactamente o golpe que
    #                   resolve o confronto.
    #   Flying Press    e LUTA, mas conta TAMBEM como VOADOR: os dois
    #                   multiplicadores multiplicam-se.
    #   Thousand Arrows e TERRA, mas ACERTA em VOADOR (e em Levitate) a 1x.
    #                   poke-env diz 0x -> o filtro 0 do masking PODA um golpe
    #                   legal e bom, e a guarda de imunidade descarta-o.
    #   Freeze Shock /  variantes de Freeze-Dry no mesmo padrao ficam registadas
    #   Flying Press    aqui a medida que aparecerem no pool.
    #
    # PORQUE VIVE NO PHYSICS E NUM SITIO SO. Esta e a mesma classe de erro que o
    # `mais_rapido` corrigiu em 30/08: uma regra do MOTOR DO JOGO espalhada por
    # onze chamadores em cru. Aqui a regra passa a ter um dono, e quem precisa do
    # multiplicador ofensivo chama isto em vez de `damage_multiplier`.
    #
    # NAO E ESTRATEGIA, e aritmetica do jogo: aplica-se por igual aos tres
    # agentes e nao alarga a diferenca entre o Blue e o Green.

    # Golpes cujo tipo efectivo contra CERTOS defensores nao e o tipo do golpe.
    # {id do golpe: {NOME DO TIPO DEFENSOR: multiplicador FIXO a impor}}
    EFECTIVIDADE_POR_TEXTO = {
        'freezedry': {'WATER': 2.0},
    }
    # Golpes que somam um SEGUNDO tipo a conta de efectividade (multiplicam).
    TIPO_EXTRA_DO_GOLPE = {
        'flyingpress': 'FLYING',
    }
    # Golpes que ignoram a imunidade do defensor e batem a 1x nesses tipos.
    IGNORA_IMUNIDADE = {
        'thousandarrows': {'FLYING'},
    }

    def multiplicador_de_tipo(self, move, defender, tipo_efetivo=None):
        """Multiplicador OFENSIVO deste golpe contra este defensor.

        Substitui `defender.damage_multiplier(move)` em todo o lado onde o que se
        quer e o multiplicador de ATAQUE. Sem excepcao aplicavel o resultado e
        identico ao do poke-env, logo nenhum chamador antigo muda de valor.

        `tipo_efetivo` permite passar o tipo ja resolvido (Weather Ball, Judgment,
        Terapagos e afins mudam de tipo antes de chegar aqui).
        """
        try:
            mid = str(getattr(move, "id", "")).lower()
            tipos_def = [t.name for t in getattr(defender, "types", []) or [] if t]

            # 1. Multiplicador FIXO por texto (Freeze-Dry contra Agua). Vence a
            #    tabela inteira: e a excepcao que o proprio golpe declara.
            fixos = self.EFECTIVIDADE_POR_TEXTO.get(mid)
            if fixos:
                for t in tipos_def:
                    if t in fixos:
                        return fixos[t]

            # 2. Multiplicador base, com o tipo resolvido se veio um.
            if tipo_efetivo is not None and tipo_efetivo is not getattr(move, "type", None):
                mult = defender.damage_multiplier(tipo_efetivo)
            else:
                mult = defender.damage_multiplier(move)

            # 3. Segundo tipo que MULTIPLICA (Flying Press).
            extra = self.TIPO_EXTRA_DO_GOLPE.get(mid)
            if extra:
                alvo = None
                for t in list(getattr(defender, "types", []) or []):
                    if t is not None:
                        alvo = getattr(type(t), extra, None)
                        break
                if alvo is not None:
                    mult *= defender.damage_multiplier(alvo)

            # 4. Ignora imunidade (Thousand Arrows contra Voador). So sobe de
            #    ZERO para 1x: nao inventa efectividade que o golpe nao tem.
            if mult == 0:
                ignora = self.IGNORA_IMUNIDADE.get(mid)
                if ignora and any(t in ignora for t in tipos_def):
                    return 1.0

            return mult
        except Exception:
            try:
                return defender.damage_multiplier(move)
            except Exception:
                return 1.0

    # ======================================================================
    # AMEACA: quanto o adversario nos faz, e sobrevivemos a isso? (03/09/2026)
    # ======================================================================
    # PORQUE VIVE NO PHYSICS. A pergunta "aguento o proximo golpe?" e ARITMETICA
    # DO JOGO, nao estrategia: e dano contra HP. Estava escrita em duplicado (uma
    # copia no `execution.py`, outra por escrever na `policy.py`), e duas copias
    # da mesma regra e como o projecto ja produziu tres bugs de divergencia
    # silenciosa. Fica num sitio so, e `policy` e `execution` consomem-na.
    #
    # NAO ENTRA NA TUPLA DE ESTADO, de proposito. O `state.get_matchup_state`
    # continua a ler EFECTIVIDADE DE TIPO, porque e essa a representacao que o
    # Blue e o Green estao a aprender e mexer nela seria a quarta mudanca de
    # percepcao do mesmo ciclo (ver 6.42). Isto serve a decisao do INSTINTO.
    #
    # LIMITE DECLARADO: so conta golpes JA REVELADOS. Um adversario que ainda nao
    # atacou devolve 0.0, e quem chama tem de decidir o que fazer com essa
    # ausencia em vez de a tratar como "nao faz dano". Nao se inventa um valor.

    def dano_maximo_conhecido(self, atacante, defensor, battle=None):
        """Maior dano (fraccao do HP do defensor) entre os golpes JA REVELADOS.

        Devolve 0.0 quando o atacante ainda nao revelou nenhum golpe de dano.
        Esse zero significa DESCONHECIDO, nao INOFENSIVO.
        """
        try:
            maior = 0.0
            for mv in (getattr(atacante, "moves", None) or {}).values():
                if getattr(mv, "base_power", 0) > 0:
                    maior = max(maior, self.estimate_damage_percent(
                        mv, atacante, defensor, battle))
            return maior
        except Exception:
            return 0.0

    # A margem existe porque `estimate_damage_percent` nao conhece EVs, IVs,
    # nature nem item do adversario, e nao modela critico. Sobreviver por 20% de
    # folga e o minimo para a decisao nao depender de a estimativa estar certa.
    MARGEM_SOBREVIVENCIA = 1.20

    def sobrevive_a(self, defensor, atacante, battle=None, margem=None,
                    golpes=1, se_desconhecido=None):
        """O defensor aguenta `golpes` do melhor ataque conhecido do atacante?

        `se_desconhecido` e o valor devolvido quando o atacante ainda nao revelou
        golpe de dano. Quem chama decide: `None` (o default) devolve `None` para
        obrigar a uma decisao explicita a montante; passar `True` ou `False`
        escolhe o optimismo ou a prudencia nesse caso.
        """
        try:
            dano = self.dano_maximo_conhecido(atacante, defensor, battle)
            if dano <= 0.0:
                return se_desconhecido
            margem = self.MARGEM_SOBREVIVENCIA if margem is None else margem
            hp = getattr(defensor, "current_hp_fraction", 1.0) or 0.0
            return hp > dano * margem * max(1, int(golpes))
        except Exception:
            return se_desconhecido

    def morre_num_golpe(self, defensor, atacante, battle=None):
        """Atalho legivel: o melhor golpe conhecido do atacante mata-nos agora?

        Sem golpe revelado devolve False (nao se assume o pior sem dado), e quem
        precisa de prudencia usa `sobrevive_a(..., se_desconhecido=False)`.
        """
        return self.sobrevive_a(defensor, atacante, battle,
                                se_desconhecido=True) is False

    def fracao_de_dreno(self, move):
        """Fracao do dano que este golpe recupera. 0.0 se nao for de dreno."""
        mid = getattr(move, "id", "")
        if mid not in self.GOLPES_DE_DRENO:
            return 0.0
        bruto = getattr(move, "drain", None)
        if bruto:
            try:
                v = float(bruto)
                # poke-env pode expor como fracao (0.5) ou percentagem (50).
                return v / 100.0 if v > 1.0 else v
            except (TypeError, ValueError):
                pass
        return 0.75 if mid in self._DRENO_75 else 0.5

    _TECH_REMOVE_ITEM = {'knockoff', 'thief'}

    # ==================================================================
    # ITENS QUE O KNOCK OFF NAO CONSEGUE DERRUBAR (30/08/2026)
    # ==================================================================
    # REGRA DE JOGO que o codigo nao conhecia. O bonus de Knock Off no executor
    # exigia apenas "o alvo tem item", e ha uma familia inteira de itens que NAO
    # saem: Mega Stones, Z-Crystals, Orbes Primais e as Rusted do Zacian/Zamazenta.
    # Quando o item nao sai, o golpe tambem NAO recebe o x1.5 de potencia.
    #
    # Observado em batalha manual: Rillaboom usou Knock Off contra um Mega Swampert.
    # Contra o Pelipper anterior o log dizia "knocked off Pelipper's Damp Rock";
    # contra o Swampert nao houve mensagem de remocao nenhuma. O bonus foi pago por
    # um efeito que nao aconteceu.
    #
    # NAO E CASO RARO NESTE POOL: a varredura do teams_eval encontrou nove tipos de
    # Z-crystal e varias mega stones.
    REMOVEDORES_DE_ITEM = {'knockoff', 'thief', 'covet'}

    # WEATHER BALL: o clima muda o TIPO do golpe e dobra a potencia. Constante de
    # CLASSE e usada nos DOIS sitios que precisam dela (o STAB/efetividade e o
    # modificador de clima), para nao voltarem a divergir.
    TIPO_DO_WEATHER_BALL = {
        'SUNNYDAY': 'FIRE', 'DESOLATELAND': 'FIRE',
        'RAINDANCE': 'WATER', 'PRIMORDIALSEA': 'WATER',
        'SANDSTORM': 'ROCK',
        'HAIL': 'ICE', 'SNOW': 'ICE', 'SNOWSCAPE': 'ICE',
    }

    _MEGA_STONES = {
        'abomasite', 'absolite', 'aerodactylite', 'aggronite', 'alakazite',
        'altarianite', 'ampharosite', 'audinite', 'banettite', 'beedrillite',
        'blastoisinite', 'blazikenite', 'cameruptite', 'charizarditex',
        'charizarditey', 'diancite', 'galladite', 'garchompite', 'gardevoirite',
        'gengarite', 'glalitite', 'gyaradosite', 'heracronite', 'houndoominite',
        'kangaskhanite', 'latiasite', 'latiosite', 'lopunnite', 'lucarionite',
        'manectite', 'mawilite', 'medichamite', 'metagrossite', 'mewtwonitex',
        'mewtwonitey', 'pidgeotite', 'pinsirite', 'sablenite', 'salamencite',
        'sceptilite', 'scizorite', 'sharpedonite', 'slowbronite', 'steelixite',
        'swampertite', 'tyranitarite', 'venusaurite',
    }
    # Lista EXPLICITA em vez de sufixo "ite": o Eviolite acabaria protegido por
    # engano, e a Charizardite X/Y e a Mewtwonite X/Y nem sequer terminam em "ite".
    _ORBES_PRESOS = {'redorb', 'blueorb', 'griseousorb', 'rustedsword', 'rustedshield'}

    # ==================================================================
    # CLIMA: DANO RESIDUAL E CURA (30/08/2026)
    # ==================================================================
    # Listas de REGRA DE JOGO, verificadas contra a referencia do granizo e da areia.
    IMUNES_AO_GRANIZO = {'icebody', 'snowcloak', 'magicguard', 'overcoat'}
    IMUNES_A_AREIA = {'sandveil', 'sandrush', 'sandforce', 'magicguard', 'overcoat'}
    CURAS_PELO_CLIMA = {'moonlight', 'synthesis', 'morningsun'}

    # ==================================================================
    # DURACAO DO CLIMA E DOS CAMPOS (30/08/2026)
    # ==================================================================
    # NAO EXISTIA CONTADOR NENHUM. Clima, terreno e Trick Room duram 5 turnos (8 com
    # a pedra correspondente), e o instinto via a mesma coisa no turno 1 e no turno 5.
    # Sao situacoes OPOSTAS: montar um sweep sob um Trick Room que acaba no proximo
    # turno e entregar o Pokemon ao adversario com a velocidade de volta ao normal.
    #
    # O poke-env guarda o TURNO DE INICIO como valor de `battle.weather` e
    # `battle.fields`, logo o contador sai de uma subtracao — nao e preciso estado
    # extra por batalha.
    DURACAO_PADRAO = 5
    DURACAO_ESTENDIDA = 8          # com Heat/Damp/Smooth/Icy Rock ou Terrain Extender

    _PEDRAS_DE_CLIMA = {
        'SUNNYDAY': 'heatrock', 'DESOLATELAND': 'heatrock',
        'RAINDANCE': 'damprock', 'PRIMORDIALSEA': 'damprock',
        'SANDSTORM': 'smoothrock',
        'HAIL': 'icyrock', 'SNOW': 'icyrock', 'SNOWSCAPE': 'icyrock',
    }

    def _turnos_restantes(self, turno_inicio, turno_atual, duracao):
        if turno_inicio is None or turno_atual is None:
            return None
        restam = duracao - (int(turno_atual) - int(turno_inicio))
        return max(0, restam)

    def turnos_de_clima(self, battle, dono=None):
        """Turnos que faltam ao clima atual. None se nao ha clima.

        `dono` e opcional: se for o Pokemon que instalou o clima e ele carregar a
        pedra correspondente, a duracao passa de 5 para 8. Sem ele assume-se 5, que
        e o caso comum e o erro cai para o lado de subestimar o tempo restante.
        """
        try:
            if not battle or not battle.weather:
                return None
            clima, inicio = next(iter(battle.weather.items()))
            nome = clima.name.upper() if hasattr(clima, "name") else str(clima).upper()
            duracao = self.DURACAO_PADRAO
            if dono is not None:
                item = str(getattr(dono, "item", "") or "").lower().replace(" ", "")
                if item and item == self._PEDRAS_DE_CLIMA.get(nome):
                    duracao = self.DURACAO_ESTENDIDA
            return self._turnos_restantes(inicio, getattr(battle, "turn", None), duracao)
        except Exception:
            return None

    def turnos_de_campo(self, battle, nome_do_campo):
        """Turnos que faltam a um campo (TRICK_ROOM, ELECTRIC_TERRAIN...). None se inativo."""
        try:
            for campo, inicio in (getattr(battle, "fields", None) or {}).items():
                # MIGRADO PARA `.name` EM 03/09/2026: `turnos_de_campo` devolvia
                # sempre None, logo `campo_a_expirar` nunca disparava.
                nome = str(getattr(campo, "name", campo)).upper().replace("_", "")
                if nome == nome_do_campo.upper().replace("_", ""):
                    return self._turnos_restantes(inicio, getattr(battle, "turn", None),
                                                  self.DURACAO_PADRAO)
            return None
        except Exception:
            return None

    def campo_a_expirar(self, battle, nome_do_campo, limiar=1):
        """O campo acaba dentro de `limiar` turnos? False se nem sequer esta ativo.

        E a pergunta que decide: sob Trick Room com 1 turno restante, montar um
        sweep lento e pior do que nao fazer nada, porque no turno seguinte a
        velocidade volta ao normal com o nosso Pokemon exposto.
        """
        restam = self.turnos_de_campo(battle, nome_do_campo)
        return restam is not None and restam <= limiar

    def sofre_dano_de_clima(self, mon, battle):
        """Este Pokemon leva o residual de 1/16 do clima atual?

        A lista de imunidades estava INCOMPLETA no `get_weather_state`: so olhava
        para o tipo, Magic Guard e Overcoat. Faltavam Ice Body e Snow Cloak no
        granizo, Sand Veil, Sand Rush e Sand Force na areia, e as Safety Goggles nos
        dois. Um Excadrill com Sand Rush era lido como se estivesse a sofrer com a
        areia que a propria equipa instalou.

        NEVE NAO CAUSA DANO na Gen 9 — so o granizo. Por isso SNOW e SNOWSCAPE nao
        entram na lista de climas que magoam.
        """
        try:
            clima = next(iter(battle.weather)).name.upper() if battle and battle.weather else "CLEAR"
            tipos = [t.name for t in mon.types if t]
            hab = str(getattr(mon, "ability", "") or "").lower()
            item = str(getattr(mon, "item", "") or "").lower().replace(" ", "")
            if item == 'safetygoggles':
                return False
            if clima == "SANDSTORM":
                if any(t in tipos for t in ("ROCK", "GROUND", "STEEL")):
                    return False
                return hab not in self.IMUNES_A_AREIA
            if clima == "HAIL":
                if "ICE" in tipos:
                    return False
                return hab not in self.IMUNES_AO_GRANIZO
            return False
        except Exception:
            return False

    def fracao_de_cura(self, move, battle):
        """Quanto este golpe cura DE FACTO, contando o clima.

        Moonlight, Synthesis e Morning Sun curam 1/2 sem clima, 2/3 em sol forte e
        apenas 1/4 em qualquer outro clima. Shore Up cura 1/2 e sobe para 2/3 na
        areia. O `move.heal` do poke-env nao conhece nada disto.

        Importa para o filtro 10b do masking, que compara a cura com o dano que se
        vai levar: com Synthesis num granizo a cura real e METADE do que o filtro
        assumia, e a corrida que ele julgava ganha esta perdida.
        """
        try:
            base = float(getattr(move, "heal", 0) or 0.0)
            clima = next(iter(battle.weather)).name.upper() if battle and battle.weather else "CLEAR"
            if move.id in self.CURAS_PELO_CLIMA:
                if clima in ("SUNNYDAY", "DESOLATELAND"):
                    return 2.0 / 3.0
                if clima in ("RAINDANCE", "PRIMORDIALSEA", "SANDSTORM", "HAIL", "SNOW", "SNOWSCAPE"):
                    return 0.25
                return 0.5
            if move.id == 'shoreup':
                return 2.0 / 3.0 if clima == "SANDSTORM" else 0.5
            return base
        except Exception:
            return float(getattr(move, "heal", 0) or 0.0)

    def precisao_efetiva(self, move, battle):
        """Precisao real, contando o clima. Blizzard NUNCA erra no granizo."""
        try:
            acc = move.accuracy
            acc = float(acc) if isinstance(acc, (int, float)) else 100.0
            if acc <= 1.0:
                acc *= 100.0
            clima = next(iter(battle.weather)).name.upper() if battle and battle.weather else "CLEAR"
            if move.id == 'blizzard' and clima in ("HAIL", "SNOW", "SNOWSCAPE"):
                return 100.0
            return acc
        except Exception:
            return 100.0

    def item_e_removivel(self, alvo):
        """O item do alvo pode ser derrubado por Knock Off / Thief / Covet?

        Z-Crystals detetam-se pelo sufixo `iumz`, que cobre as genericas
        (`rockiumz`) e as de especie (`pikaniumz`, `ultranecroziumz`) sem precisar
        de as listar uma a uma.

        LIMITE ASSUMIDO: uma Mega Stone so e inamovivel no Pokemon que a USA — um
        Pikachu com Charizardite perde-a. Nao se modela isso, e o erro cai para o
        lado seguro: na duvida NAO se paga o bonus, em vez de o pagar por um efeito
        que nao vai acontecer.

        Sticky Hold entra aqui por ser a mesma pergunta: o item nao sai.
        """
        try:
            item = str(getattr(alvo, "item", "") or "").lower().replace(" ", "").replace("-", "")
            if not item:
                return False
            if str(getattr(alvo, "ability", "") or "").lower() == 'stickyhold':
                return False
            if item.endswith('iumz'):
                return False
            return item not in self._MEGA_STONES and item not in self._ORBES_PRESOS
        except Exception:
            return False
    _TECH_QUEBRA_BARREIRA = {'brickbreak', 'psychicfangs'}
    _TECH_APLICA_STATUS = {'scald', 'nuzzle', 'discharge', 'lavaplume'}

    def efeito_tech_gasto(self, move, defender, battle):
        """O efeito deste ATTACK_TECH ja nao pode produzir nada?

        Devolve False em caso de duvida: e melhor manter o golpe como TECH do que
        rebaixa-lo por engano.
        """
        if defender is None or battle is None:
            return False
        mid = getattr(move, "id", "")

        try:
            # DRENO (04/09/2026). O "efeito" de um golpe de dreno e a CURA. Com o
            # HP quase cheio ela nao devolve nada, logo o golpe vale so pelo dano e
            # deve competir como ataque normal em vez de continuar a ser preferido
            # como tech. Ferido, drenar continua a valer mais que bater, e ai
            # mantem-se TECH. O caso LETAL ja foi resolvido antes, no topo do
            # `classify_move`.
            if mid in self.GOLPES_DE_DRENO:
                atacante = getattr(battle, "active_pokemon", None)
                hp = getattr(atacante, "current_hp_fraction", None)
                return hp is not None and hp >= 0.90

            if mid in self._TECH_REMOVE_ITEM:
                # Sem item para roubar ou derrubar, sobra o dano.
                return not getattr(defender, "item", None)

            if mid in self._TECH_QUEBRA_BARREIRA:
                # MIGRADO PARA `.name` EM 03/09/2026: Brick Break e afins eram
                # dados como uteis mesmo sem barreira nenhuma para partir.
                return not self.tem(getattr(battle, "opponent_side_conditions", None),
                                    "REFLECT", "LIGHT_SCREEN", "AURORA_VEIL")

            if mid in self._TECH_APLICA_STATUS:
                # So um status por Pokemon: com um ja aplicado, o secundario nao pega.
                return getattr(defender, "status", None) is not None

            if mid == 'saltcure':
                # MIGRADO PARA `.name` EM 03/09/2026.
                return self.tem(getattr(defender, "effects", None), "SALT_CURE")

            if mid in self.GOLPES_DE_PRISAO:
                # Ja esta preso: repetir nao acrescenta turnos uteis.
                # MIGRADO PARA `.name` EM 03/09/2026: o instinto repetia golpes de
                # prisao sobre um alvo JA preso, porque nunca via o efeito.
                return self.tem(getattr(defender, "effects", None),
                                "BIND", "WRAP", "INFESTATION", "FIRE_SPIN",
                                "WHIRLPOOL", "SAND_TOMB", "MAGMA_STORM")

            if mid == 'fakeout':
                # So funciona no primeiro turno do Pokemon em campo.
                return not getattr(battle, "active_pokemon", None) or \
                    getattr(battle, "turn", 0) > 1
        except Exception:
            return False
        return False

    def classify_move(self, move, defender=None, battle=None) -> MoveCategory:
        """Mapeia um golpe para a sua categoria funcional (usada pelo masking e
        pela política)."""
        move_id = move.id

        # ==================================================================
        # O GOLPE LETAL E SEMPRE ATTACK_STRONG (04/09/2026)
        # ==================================================================
        # PORQUE ESTA REGRA VEM PRIMEIRO. A classificacao por BLOCO diz para que
        # serve um golpe; quando ele MATA, essa pergunta deixa de ter interesse.
        # Um Giga Drain 4x que mata nao e "um golpe de dreno": e o fim do
        # confronto. Nenhum efeito secundario compete com um KO.
        #
        # DEFEITO QUE ISTO CORRIGE. Os treze golpes de dreno estavam na lista
        # `tech_moves` e so saiam de la por `efeito_tech_gasto`, que nao tratava
        # dreno de todo — logo eram ATTACK_TECH PARA SEMPRE. Observado: Volcarona
        # a 59% contra Gastrodon (WATER/GROUND) usou `psychic` (1x) tendo
        # `gigadrain` (4x) no moveset. Baixa a taxa de kill da regua, e para o
        # CEREBRO e pior: o Blue e o Green nao tem `comparar_ataques`, escolhem a
        # intencao pela Q-table, e o melhor golpe estar permanentemente atras do
        # rotulo ATTACK_TECH obriga-os a aprender que ATTACK_TECH as vezes quer
        # dizer "o meu melhor ataque". E ruido na representacao, da mesma familia
        # do colapso de accoes do atalho de pivo.
        #
        # SEM CONTEXTO NAO SE MEXE. Com `defender` ou `battle` a None o
        # comportamento e o de sempre, o que preserva os chamadores sem contexto
        # (em particular o `pure_agent`, e com ele o grupo de controlo).
        #
        # EFEITO LATERAL DECLARADO: um golpe de pivo LETAL passa a ATTACK_STRONG,
        # logo deixa de ser candidato quando a intencao e ATTACK_PIVOT. Nao se
        # perde a jogada: o executor, sem candidatos na categoria pedida, cai em
        # ATTACK_STRONG, e o U-turn letal mata E troca na mesma.
        if defender is not None and battle is not None and getattr(move, "base_power", 0) > 0:
            try:
                atacante = getattr(battle, "active_pokemon", None)
                alvo_hp = getattr(defender, "current_hp_fraction", None)
                if atacante is not None and alvo_hp:
                    if self.estimate_damage_percent(move, atacante, defender, battle) >= alvo_hp:
                        return MoveCategory.ATTACK_STRONG
            except Exception:
                pass

        if move_id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']:
            return MoveCategory.ATTACK_PIVOT

        tech_moves = [
            'knockoff', 'foulplay', 'thief', 'nuzzle', 'scald', 'discharge', 'lavaplume', 'saltcure',
            'superfang', 'naturesmadness', 'ruination', 'seismictoss', 'nightshade', 'icywind', 'electroweb',
            'rocktomb', 'bulldoze', 'snarl', 'mysticalfire', 'strugglebug', 'fakeout', 'brickbreak',
            'psychicfangs', 'bodypress'
        ] + sorted(self.GOLPES_DE_PRISAO) + sorted(self.GOLPES_DE_DRENO)
        if move_id in tech_moves:
            # Se o efeito ja esta gasto, o golpe vale so pelo dano: compete como
            # ataque normal em vez de continuar a ser preferido como tech.
            # Sem contexto (defender/battle a None) o comportamento e o de sempre,
            # o que mantem intactos os chamadores que nao passam contexto — em
            # particular o pure_agent, preservando o grupo de controlo.
            if self.efeito_tech_gasto(move, defender, battle):
                return MoveCategory.ATTACK_STRONG
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
        # `rest` ACRESCENTADO EM 04/09/2026, e a CAUSA CORRIGIDA EM 05/09.
        #
        # A primeira versao desta nota dizia que o Rest saia como STATUS porque o
        # poke-env lhe expoe o `status` de sono. E FALSO, e fica registado para
        # ninguem repetir o raciocinio. Medido no poke-env instalado:
        #
        #     rest    boosts=None   heal=0.0   status=None
        #     curse   boosts=None   heal=0.0   status=None
        #     recover boosts=None   heal=0.5   status=None
        #
        # O Rest nao tem NADA preenchido, tal como o Curse: o Showdown resolve os
        # dois por `onModifyMove` e o poke-env nao expoe valor estatico. Caia no
        # `return MoveCategory.STATUS` do FIM do ramo generico, por exclusao de
        # partes, e nao por o status de sono ser detectado.
        #
        # A consequencia era a mesma: escolhido sob a intencao STATUS, escapava
        # aos filtros 10 e 10b de cura. Observado `STATUS obj=rest` em SEIS turnos
        # seguidos. E cura, e e como cura que tem de ser filtrado — dai a
        # classificacao por ID, que nao depende de campos que a biblioteca nao
        # preenche.
        #
        # LIGACAO A MANTER: como `rest.heal` e 0.0, o filtro 10 do `masking` SO o
        # apanha pela lista explicita `healing_moves`, nunca pela deteccao
        # generica. Ver a nota la.
        if move_id in ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun', 'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap', 'rest']:
            return MoveCategory.HEAL
        if move_id in ['reflect', 'lightscreen', 'auroraveil']:
            return MoveCategory.BARRIER
        if move.id in ['taunt', 'torment', 'encore', 'disable']:
            return MoveCategory.DISRUPTION

        # ==================================================================
        # CURSE: DOIS GOLPES DIFERENTES NO MESMO ID (04/09/2026)
        # ==================================================================
        # Para quem NAO e Fantasma, o Curse sobe Ataque e Defesa e baixa
        # Velocidade: e um BUFF de bruiser, e como tal tem de ser tratado, para
        # lhe valerem a Regra Global 1 (nao bufar em situacao instavel) e o
        # contador `buffs_consecutivos`.
        #
        # Chegava aqui como STATUS e escapava as duas. Observado: `curse` em
        # quatro turnos seguidos.
        #
        # PARA UM TIPO FANTASMA E OUTRO GOLPE: corta METADE DO PROPRIO HP e
        # amaldicoa o adversario. Chamar-lhe BUFF seria mandar um Gengar
        # suicidar-se para "melhorar as suas estatisticas". Fica STATUS, que e o
        # comportamento actual — nao ha caso observado que justifique mexer-lhe,
        # e sem prova nao se substitui.
        #
        # SEM CONTEXTO NAO SE MEXE: com `battle` a None nao ha como ler os tipos
        # de quem usa, e mantem-se o comportamento antigo. Preserva o
        # `pure_agent`, e com ele o grupo de controlo.
        if move_id == 'curse':
            try:
                _atacante = getattr(battle, "active_pokemon", None) if battle else None
                _tipos = [t.name for t in (getattr(_atacante, "types", None) or []) if t]
                if _tipos and 'GHOST' not in _tipos:
                    return MoveCategory.BUFF
            except Exception:
                pass
            return MoveCategory.STATUS

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
