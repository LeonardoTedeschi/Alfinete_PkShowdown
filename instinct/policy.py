"""
Camada 4 — Política de Instinto (InstinctPolicy).

O instinto tático propriamente dito: dado o estado da battle, produz um RANKING de
intenções de ação (categorias como ATTACK_STRONG, HEAL, HAZARD...), da mais à menos
recomendada. É o "conhecimento de domínio" que o agente híbrido (Blue) usa como prior
de exploração e que o agente Instinto-puro usa diretamente.

ESTE FICHEIRO É A ÚNICA DIFERENÇA ENTRE O BLUE E O GREEN
--------------------------------------------------------
Registado aqui em 28/08/2026 porque a informação vivia no `hybrid_agent.py`, onde já
não é visível: com `USAR_MASKING = True` e `USAR_RANKING = True` (a configuração de
todos os ciclos recentes), aquele ficheiro limita-se a documentar uma ablação que não
está a ser usada. A distinção real acontece aqui.

Os dois agentes partilham TUDO o resto — estado, cérebro, recompensa, execução,
replay, espaço de 36 ações, e até o `classify_move` da física para NOMEAR as ações
legais. O que os separa são as duas saídas de `get_instinct_profile`:

  candidate_mask  ->  BLUE recebe a lista PODADA pelos 13 filtros do ActionMasker.
                      GREEN constrói a sua a partir de todos os golpes legais, sem
                      poda tática.

  ranking_list    ->  BLUE recebe as intenções ORDENADAS pelos modos e modificadores
                      abaixo, e usa-as como prior de exploração.
                      GREEN recebe lista vazia.

O que os dados dizem sobre cada uma (ciclo v8/v5, 400k batalhas, secções 6.26/6.27):

  a PODA parece PROTEGER    — no holdout o Blue ganha ao Green 57,20% vs 54,56%
                              (2,7 sigma). A poda é conhecimento por PAPÉIS e
                              MATCHUPS, que transfere para material nunca visto.
  o RANKING parece LIMITAR  — no treino o Green ganha ao Blue 71,04% vs 68,72%
                              (13,2 sigma), e a distribuição de ações do Blue é
                              muito mais estreita (77,0% ataque vs 55,4%).

Se a leitura estiver certa, `USAR_MASKING=True` com `USAR_RANKING=False` daria o
Win Rate de treino do Green com a robustez do Blue. É a ablação que o
`hybrid_agent.py` foi desenhado para permitir e que ainda não foi corrida.

Arquitetura em 3 sub-camadas internas:
  1. _get_tactical_mode: matchup + papéis -> MODO tático (PRESS, CONTEST, GRIND,
     LEAD, WALLBREAK). O ESCAPE deixou de ser modo em 6.16: passou a ser um SINAL de
     urgência, e o seu template e `_mod_escape` foram removidos em 28/08/2026.
  2. mode_templates: cada modo tem uma LISTA base de prioridades de categoria.
  3. _mod_*: funções que REORDENAM a lista base conforme o contexto fino
     (velocidade, HP, ameaça, papéis role-vs-role).
Depois, get_instinct_profile filtra o ranking pelo action mask e aplica ajustes
finais (letalidade, barreiras, conversão de buff).

DEPENDÊNCIAS (todas por injeção): GamePhysics, StateParser, ActionMasker.
A policy é o topo da hierarquia — depende de todas as camadas abaixo.

CONTRATO (get_instinct_profile) — CORRIGIDO:
  retorna SEMPRE 5 valores: (primary, confidence, ranking_list, candidate_mask, has_lethal)
  A versão monolítica tinha um early-return com apenas 4 valores, o que provocava
  ValueError no desempacotamento do agente e queda em choose_random_move. Aqui todos
  os caminhos devolvem 5 valores.
"""

from shared import diagnostico
from shared.definitions import Role, MatchupState, TacticalMode



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
    # MIGRADO PARA `.name` EM 03/09/2026: ver a nota extensa em
    # `GamePhysics.nomes_de`. A forma antiga devolvia
    # 'REFLECT (SIDE CONDITION) OBJECT' e matava os testes por pertenca.
    return sorted({str(getattr(k, "name", k)).upper() for k in (colecao or {})})

class InstinctPolicy:
    """Decisão de intenção tática. Recebe física, parser e masker por injeção."""

    def __init__(self, physics, parser, masker):
        self.physics = physics
        self.parser = parser
        self.masker = masker
        self.mode_templates = self._build_mode_templates()
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

    # ======================================================================
    # SUB-CAMADA 2: templates base de prioridade por modo tático
    # ======================================================================

    def _build_mode_templates(self):
        return {
            TacticalMode.PRESS: [
                "ATTACK_PREDICTIVE", "ATTACK_STRONG", "BUFF", "ATTACK_TECH", "DISRUPTION",
                "HAZARD", "FIELD_CONTROL", "ATTACK_PIVOT", "CLEAN_HAZARD",
                "STATUS", "DEBUFF", "HEAL", "HEAL_STATUS", "STAT_CLEAN",
                "PHAZE", "PROTECT", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE",
            ],
            TacticalMode.CONTEST: [
                "ATTACK_STRONG", "ATTACK_TECH", "PROTECT", "ATTACK_PIVOT",
                "STATUS", "BUFF", "HEAL", "HAZARD", "CLEAN_HAZARD",
                "DEBUFF", "FIELD_CONTROL", "ATTACK_PREDICTIVE", "DISRUPTION", "STAT_CLEAN",
                "PHAZE", "HEAL_STATUS", "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE",
            ],
            TacticalMode.GRIND: [
                "HAZARD", "STATUS", "HEAL", "PROTECT", "DISRUPTION", "DEBUFF",
                "CLEAN_HAZARD", "PHAZE", "STAT_CLEAN", "HEAL_STATUS",
                "BUFF", "FIELD_CONTROL", "ATTACK_TECH", "ATTACK_PIVOT",
                "SWITCH_OFFENSIVE", "SWITCH_DEFENSIVE", "ATTACK_STRONG", "ATTACK_PREDICTIVE",
            ],
            # ESCAPE reformulado: "SAIR COM VALOR", nao "fugir a todo o custo".
            #
            # O template anterior punha SWITCH_DEFENSIVE, ATTACK_PIVOT e
            # SWITCH_OFFENSIVE nas quatro primeiras posicoes e ATTACK_STRONG em 17o
            # de 18. Combinado com o ESCAPE ser o modo por omissao, produzia ciclos
            # de troca que levavam batalhas ao turno 1000.
            #
            # Agora o modo pressupoe que sair e a intencao certa, mas ordena as
            # formas de o fazer por VALOR:
            #   1. ATTACK_PIVOT   sai de campo E causa dano (estritamente melhor
            #                     que uma troca crua)
            #   2. SWITCH_DEFENSIVE  sai limpo para quem aguenta
            #   3. PROTECT        ganha um turno e revela a jogada do adversario
            #   4. ATTACK_STRONG  se vamos cair de qualquer forma, causar dano vale
            #                     mais que trocar outra vez (era 17o, agora 4o)
            # SWITCH_OFFENSIVE desce: entrar com um frágil num matchup mau raramente
            # e fuga, e mais frequentemente perder duas pecas em vez de uma.
            # TacticalMode.ESCAPE: template REMOVIDO em 28/08/2026 (ver acima).
            TacticalMode.LEAD: [
                "HAZARD", "FIELD_CONTROL", "ATTACK_PIVOT", "DISRUPTION",
                "ATTACK_STRONG", "STATUS", "DEBUFF", "BUFF", "PROTECT",
                "CLEAN_HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
                "ATTACK_PREDICTIVE", "ATTACK_TECH", "STAT_CLEAN", "HEAL_STATUS", "PHAZE",
            ],
            TacticalMode.WALLBREAK: [
                "ATTACK_TECH", "DISRUPTION", "STATUS", "BUFF", "ATTACK_PIVOT",
                "DEBUFF", "HAZARD", "ATTACK_STRONG", "ATTACK_PREDICTIVE",
                "HEAL", "CLEAN_HAZARD", "PROTECT", "SWITCH_OFFENSIVE",
                "STAT_CLEAN", "HEAL_STATUS", "PHAZE", "FIELD_CONTROL", "SWITCH_DEFENSIVE",
            ],
        }

    # ======================================================================
    # SUB-CAMADA 1: matchup + papéis -> modo tático
    # ======================================================================

    def _get_tactical_mode(self, matchup, my_role, opp_role, is_faster,
                           my_hp_frac, opp_hp_frac, is_threat, active, opponent,
                           battle=None):
        # Tank com a defesa "errada" contra o sweeper inimigo -> foge
        if my_role == Role.TANK and opp_role == Role.SWEEPER:
            opp_is_physical = self.physics._is_physical(opponent)
            my_def = active.base_stats.get('def', 0)
            my_spd = active.base_stats.get('spd', 0)
            is_right_def = (opp_is_physical and my_def >= my_spd) or (not opp_is_physical and my_spd > my_def)
            if not is_right_def:
                # Antes devolvia o modo ESCAPE. Agora e apenas um sinal de que a
                # urgencia de saida se aplicaria (mecanismo removido em 24/08): o tank com a
                # defesa errada quer sair, mas a ordenacao do resto continua a ser a
                # do modo apropriado ao matchup.
                return TacticalMode.GRIND if not getattr(battle, "available_switches", None) \
                    else TacticalMode.CONTEST

        if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
            return TacticalMode.PRESS
        if matchup in [MatchupState.VOLATILE, MatchupState.NEUTRAL]:
            return TacticalMode.CONTEST
        if matchup in [MatchupState.STALEMATE, MatchupState.DEFENSIVE_ADV]:
            return TacticalMode.GRIND

        # ==============================================================
        # EMERGENCIA: nao e um MODO, e um MODIFICADOR
        # ==============================================================
        # O ESCAPE era um modo completo com o seu proprio template de 18 intencoes,
        # em que ATTACK_STRONG ficava em 17o lugar. Isso tinha dois defeitos:
        #
        #  1. Substituia TODA a ordenacao. Se a fuga falhasse (sem banco, ou preso),
        #     a lista de recurso era pessima: o agente ficava com PROTECT e
        #     DISRUPTION no topo em vez de atacar.
        #  2. Sendo o `return` por omissao, disparava em qualquer matchup
        #     desfavoravel — e como o Pokemon seguinte encontrava frequentemente o
        #     mesmo matchup, o ciclo de trocas fechava-se.
        #
        # Agora a emergencia e tratada como o que realmente e: uma URGENCIA DE SAIR
        # que se sobrepoe a ordenacao normal, promovendo as intencoes de saida ao
        # topo do template que ja se aplicava. O resto da ordenacao (que ataque, que
        # status) mantem-se coerente com a situacao.
        #
        # Ganho de desenho: menos um modo para manter, e a lista de recurso deixa de
        # ser absurda quando a fuga nao e possivel.
        return TacticalMode.WALLBREAK if opp_role == Role.TANK else TacticalMode.CONTEST

    # ======================================================================
    # HELPERS de apoio à decisão
    # ======================================================================

    def is_threatening(self, my_mon, opp_mon, battle=None):
        if not opp_mon or not my_mon:
            return False
        if opp_mon.boosts.get('atk', 0) >= 2 or opp_mon.boosts.get('spa', 0) >= 2:
            return True
        # MIGRADO PARA `physics.mais_rapido` EM 30/08/2026. A comparacao crua
        # `estimate_stat(a,'spe') > estimate_stat(b,'spe')` ignorava clima, terreno,
        # Tailwind e — pior — o Trick Room, que INVERTE a ordem de accao. O
        # comparador aplica os modificadores aos dois lados e inverte sob Trick Room.
        # `mais_rapido(opp, my)` e o equivalente exacto de `opp_speed > my_speed`:
        # estrito, logo o empate continua a NAO contar como ameaca.
        if my_mon.current_hp_fraction < 0.45 and self.physics.mais_rapido(opp_mon, my_mon, battle):
            opp_atk = max(self.physics.estimate_stat(opp_mon, 'atk'), self.physics.estimate_stat(opp_mon, 'spa'))
            if opp_atk > 250:
                return True
        return False

    def _has_recovery(self, pokemon):
        if not pokemon:
            return False
        recovery_moves = ['recover', 'roost', 'moonlight', 'slackoff', 'morningsun',
                          'synthesis', 'softboiled', 'milkdrink', 'shoreup', 'strengthsap']
        return any(m.id in recovery_moves for m in pokemon.moves.values())

    def _has_move(self, pokemon, move_ids):
        if not pokemon:
            return False
        return any(m.id in move_ids for m in pokemon.moves.values())

    def _opponent_can_setup(self, opponent):
        if not opponent:
            return False
        setup_moves = ['swordsdance', 'dragondance', 'nastyplot', 'quiverdance',
                       'shellsmash', 'shiftgear', 'calmmind', 'bulkup', 'workup', 'coil']
        return any(m.id in setup_moves for m in opponent.moves.values())

    def _get_hazard_damage(self, candidate, battle):
        """Fracao de HP que `candidate` perde AO ENTRAR em campo, do NOSSO lado.

        Usada por `_get_survival_score` para escolher para onde trocar. Ate 28/08/2026
        devolvia SEMPRE ZERO por causa do bug dos enums (ver `nomes_de_enum`), logo a
        escolha de troca ignorava por completo o dano de entrada — com hazards
        presentes em 91% dos times do pool.

        Completada na mesma data. O que faltava:

          HEAVY-DUTY BOOTS  anula TODOS os hazards de entrada. Sem isto, o instinto
                            evitava trocar para o Pokemon que justamente pode entrar
                            de graca. E o erro mais caro dos quatro, porque inverte a
                            decisao em vez de a enviesar.
          TOXIC SPIKES      1 camada = veneno, 2 = veneno grave. Nao e dano imediato,
                            e por isso contabiliza-se o custo do PRIMEIRO tique
                            (6,25% / 12,5%), que e o que se paga no turno de entrada.
                            Absorvido por Pokemon de tipo Veneno, que limpam as camadas.
          STICKY WEB        nao tira HP: baixa a velocidade em 1. Custo tratado como
                            equivalente pequeno de HP, para o score de sobrevivencia o
                            penalizar sem o tratar como dano real.
          SPIKES            a leitura de camadas usava `battle.side_conditions.get('spikes')`
                            com uma chave STRING, quando as chaves sao enums. Devolvia
                            sempre o default 1, mesmo com 3 camadas.

        Imunidades a hazards de contacto com o solo (Spikes, Toxic Spikes, Sticky Web):
        tipo Voador, Levitate, Air Balloon e Magic Guard.
        """
        dmg = 0.0

        # Heavy-Duty Boots anula tudo. Verificar primeiro evita as restantes contas.
        item = str(getattr(candidate, "item", "") or "").lower()
        if item == "heavydutyboots":
            return 0.0

        habilidade = str(getattr(candidate, "ability", "") or "").lower()
        if habilidade == "magicguard":
            return 0.0

        cond = nomes_de_enum(battle.side_conditions.keys())
        # CAMADAS REAIS (reposto em 04/09/2026). Verificado no codigo do poke-env
        # instalado (`abstract_battle._side_start`): para SPIKES e TOXIC_SPIKES o
        # valor E o numero de camadas (`conditions.get(condition, 0) + 1`); para
        # todos os outros e o TURNO de inicio. Em 03/09 assumiu-se turno para
        # todos e passou-se a contar uma camada fixa — errado precisamente para os
        # dois que empilham. Ver a mesma nota em `masking.is_hazard_already_set`.
        camadas = {nomes_de_enum([k])[0]: v for k, v in battle.side_conditions.items()}
        tipos = [t.name for t in candidate.types if t]

        # --- Stealth Rock: afeta todos, com multiplicador de tipo Rocha ---
        if 'STEALTH_ROCK' in cond:
            for t in candidate.types:
                if t:
                    rock_enum = getattr(type(t), 'ROCK', None)
                    if rock_enum:
                        dmg += 0.125 * candidate.damage_multiplier(rock_enum)
                        break

        # --- Hazards de solo: so afetam quem toca no chao ---
        no_chao = ('FLYING' not in tipos
                   and habilidade != 'levitate'
                   and item != 'airballoon')

        if no_chao:
            if 'SPIKES' in cond:
                n = camadas.get('SPIKES', 1)
                n = int(n) if isinstance(n, int) else 1
                dmg += {1: 0.125, 2: 0.1667, 3: 0.25}.get(n, 0.125)

            if 'TOXIC_SPIKES' in cond and 'POISON' not in tipos and 'STEEL' not in tipos:
                n = camadas.get('TOXIC_SPIKES', 1)
                n = int(n) if isinstance(n, int) else 1
                # Custo do primeiro tique. O envenenamento continua a pesar depois,
                # mas isso e HP que o resto do sistema ja contabiliza.
                dmg += 0.125 if n >= 2 else 0.0625

            if 'STICKY_WEB' in cond:
                # Nao e dano. Equivalente pequeno para o score de sobrevivencia
                # preferir, em igualdade de circunstancias, quem nao perde velocidade.
                dmg += 0.05

        return dmg

    def _get_survival_score(self, candidate, opponent, battle, is_active=False):
        if not candidate:
            return -9999
        hp_frac = candidate.current_hp_fraction
        if not is_active:
            hazard_dmg = self._get_hazard_damage(candidate, battle)
            if hp_frac <= hazard_dmg + 0.05:
                return -9999
        score = 0.0
        if hp_frac >= 0.7:
            score += 150
        elif hp_frac >= 0.4:
            score += 50
        else:
            score -= 100
        if not opponent:
            return score

        opp_types_obj = [t for t in opponent.types if t]
        known_opp_moves = [m for m in opponent.moves.values() if m.base_power > 0]
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

        # MIGRADO PARA `physics.mais_rapido` EM 30/08/2026. A comparacao crua
        # `estimate_stat(a,'spe') > estimate_stat(b,'spe')` ignorava clima, terreno,
        # Tailwind e — pior — o Trick Room, que INVERTE a ordem de accao. O
        # comparador aplica os modificadores aos dois lados e inverte sob Trick Room.
        if self.physics.mais_rapido(candidate, opponent, battle):
            score += 100
            has_se_move = any(m.base_power > 0 and opponent.damage_multiplier(m) > 1.5 for m in candidate.moves.values())
            if has_se_move:
                score += 150
        else:
            if has_weakness:
                score -= 200

        matchup = self.parser.get_matchup_state(candidate, opponent)
        if matchup == MatchupState.DOMINANT:
            score += 200
        elif matchup == MatchupState.DEFENSIVE_ADV:
            score += 100
        elif matchup == MatchupState.CRITICAL_DIS:
            score -= 300
        return score

    def _is_active_best_remaining(self, active, opponent, battle):
        # NOTA: no monólito original este método estava DUPLICADO (definido duas
        # vezes, linhas 325 e 1061). As duas versões eram funcionalmente idênticas
        # (mesma margem de +50), então o comportamento nunca variou — em Python a
        # segunda definição apenas sobrepunha a primeira. Aqui fica uma só cópia.
        if not battle.available_switches:
            return True
        active_score = self._get_survival_score(active, opponent, battle, is_active=True)
        best_bench_score = -9999
        for bench_mon in battle.available_switches:
            bench_score = self._get_survival_score(bench_mon, opponent, battle, is_active=False)
            if bench_score > best_bench_score:
                best_bench_score = bench_score
        # Só troca se o banco for consideravelmente mais seguro (+50), evitando
        # trocas infinitas entre dois Pokémon ruins.
        if best_bench_score > active_score + 50:
            return False
        return True

    # ======================================================================
    # SUB-CAMADA 3: os 12 modificadores de role (reordenam a lista base)
    # ======================================================================

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
        opp_is_physical = self.physics._is_physical(opponent)
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
        if self._has_recovery(opponent) and "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        return modified

    def _mod_tank_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        has_delay = self._has_move(active, ['futuresight', 'doomdesire'])
        has_pivot = self._has_move(active, ['uturn', 'voltswitch', 'flipturn', 'teleport'])
        has_protect = self._has_move(active, ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker'])
        if has_delay and has_pivot:
            if "ATTACK_TECH" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
            if "ATTACK_PIVOT" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
            return modified
        if "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "STATUS" in modified:
            modified.insert(0, modified.pop(modified.index("STATUS")))
        if "HEAL" in modified:
            modified.insert(0, modified.pop(modified.index("HEAL")))
        if has_protect and "PROTECT" in modified:
            modified.insert(0, modified.pop(modified.index("PROTECT")))
        if "SWITCH_OFFENSIVE" in modified:
            modified.insert(0, modified.pop(modified.index("SWITCH_OFFENSIVE")))
        if "ATTACK_TECH" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        return modified

    def _mod_tank_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        if "STATUS" in modified:
            modified.insert(0, modified.pop(modified.index("STATUS")))
        if self._has_move(opponent, ['uturn', 'voltswitch', 'flipturn', 'teleport']) and "ATTACK_TECH" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        return modified

    def _mod_utility_vs_sweeper(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if is_threat and not is_faster:
            if "SWITCH_DEFENSIVE" in modified:
                modified.insert(0, "SWITCH_DEFENSIVE")
            if "ATTACK_TECH" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_TECH")))
        if self._opponent_can_setup(opponent) and "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        elif "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        if is_faster and opp_hp_frac <= 0.4 and "ATTACK_STRONG" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        return modified

    def _mod_utility_vs_tank(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        if "STATUS" in modified:
            modified.insert(0, modified.pop(modified.index("STATUS")))
        if "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        if "DEBUFF" in modified:
            modified.insert(0, modified.pop(modified.index("DEBUFF")))
        if "HAZARD" in modified:
            modified.insert(0, modified.pop(modified.index("HAZARD")))
        # NOTA: bloco original removia ATTACK_STRONG quando havia SWITCH_DEFENSIVE.
        if "ATTACK_STRONG" in modified and "SWITCH_DEFENSIVE" in modified:
            modified.remove("ATTACK_STRONG")
        return modified

    def _mod_utility_vs_utility(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac, is_threat):
        modified = base.copy()
        for cat in ["HAZARD", "STATUS", "FIELD_CONTROL", "ATTACK_PIVOT", "DEBUFF"]:
            if cat in modified:
                modified.insert(0, modified.pop(modified.index(cat)))
        return modified

    # REMOVIDO EM 28/08/2026: `_mod_escape`.
    # Residuo da reformulacao do ESCAPE (ver 6.16). O `_get_tactical_mode` deixou de
    # devolver TacticalMode.ESCAPE — passou a ser um SINAL de urgencia, nao um modo —
    # logo o despacho de modificadores nunca chegava aqui. Era codigo morto que dava
    # a impressao de existir uma politica de fuga ativa, tal como o anti_loop deu a
    # impressao de existir uma guarda anti-ciclo.

    # ==================================================================
    # CLIMA: DE QUEM E, E A QUEM SERVE (29/08/2026)
    # ==================================================================
    # O `_mod_lead` verificava apenas se EXISTIA clima (`battle.weather` nao vazio).
    # Com isso, o clima do ADVERSARIO bloqueava a nossa promocao de FIELD_CONTROL: se
    # ele punha areia e a nossa equipa era de sol, o instinto aceitava passivamente a
    # areia dele mesmo com Sunny Day na mao e abusadores de sol vivos.
    #
    # Passa a distinguir tres coisas:
    #   1. que climas a NOSSA equipa aproveita (por habilidade E por golpe)
    #   2. se o clima ATUAL e um deles
    #   3. se por o nosso clima PREJUDICA o adversario, o que justifica poe-lo
    #      mesmo sem abusadores vivos

    ABUSADORES_POR_CLIMA = {
        'sol':     {'chlorophyll', 'solarpower', 'leafguard', 'flowergift', 'harvest'},
        'chuva':   {'swiftswim', 'raindish', 'dryskin', 'hydration'},
        'areia':   {'sandrush', 'sandforce', 'sandveil'},
        'granizo': {'slushrush', 'icebody', 'snowcloak', 'iceface'},
    }
    GOLPES_POR_CLIMA = {
        'sol':     {'solarbeam', 'solarblade', 'growth', 'sunnyday', 'morningsun'},
        'chuva':   {'thunder', 'hurricane', 'raindance'},
        'areia':   {'sandstorm', 'shoreup'},
        'granizo': {'auroraveil', 'blizzard', 'snowscape', 'hail'},
    }
    HABILIDADES_POR_CLIMA = {
        'drought': 'sol', 'orichalcumpulse': 'sol', 'desolateland': 'sol',
        'drizzle': 'chuva', 'primordialsea': 'chuva',
        'sandstream': 'areia',
        'snowwarning': 'granizo',
    }
    _NOME_DE_CLIMA = {
        'SUNNYDAY': 'sol', 'DESOLATELAND': 'sol',
        'RAINDANCE': 'chuva', 'PRIMORDIALSEA': 'chuva',
        'SANDSTORM': 'areia',
        'HAIL': 'granizo', 'SNOW': 'granizo', 'SNOWSCAPE': 'granizo',
    }

    @staticmethod
    def equipa_de_trick_room(equipa):
        """A equipa depende de Trick Room?

        TRICK ROOM E UM CASO A PARTE do resto do FIELD_CONTROL, e por isso tem
        caminho proprio (29/08/2026):

          - NAO EXISTE HABILIDADE que o ative. Ao contrario do sol ou da chuva, nao
            ha forma de o obter de graca: ou se gasta o turno com o golpe, ou nao ha
            Trick Room. Logo a guarda do `clima_de_graca` NAO se lhe aplica.
          - Nao esta em `battle.weather`, esta em `battle.fields`.
          - Times construidos a volta dele sao LENTOS DE PROPOSITO: sem Trick Room
            ativo, sao simplesmente times lentos. A dependencia e total, nao marginal.

        Deteta-se por ter o golpe na equipa, nao so pela velocidade media: uma equipa
        pode ter velocidade media enganadora e na mesma girar a volta de um setter.
        """
        for m in equipa:
            if getattr(m, "fainted", False):
                continue
            for mv in getattr(m, "moves", {}).values():
                if getattr(mv, "id", "") == "trickroom":
                    return True
        return False

    def trick_room_ativo(self, battle):
        try:
            # MIGRADO PARA `physics.tem` EM 03/09/2026: o Trick Room NUNCA foi
            # detectado pela policy. Uma equipa de Trick Room jogava como equipa
            # lenta mesmo com ele activo.
            return self.physics.tem(battle.fields, "TRICK_ROOM")
        except Exception:
            return False

    def _clima_atual(self, battle):
        try:
            if not battle.weather:
                return None
            bruto = next(iter(battle.weather)).name.upper()
        except Exception:
            return None
        return self._NOME_DE_CLIMA.get(bruto)

    def _climas_uteis(self, equipa):
        """Climas que a equipa VIVA aproveita, por habilidade ou por golpe."""
        uteis = set()
        for m in equipa:
            if getattr(m, "fainted", False):
                continue
            hab = str(getattr(m, "ability", "")).lower()
            for clima, habs in self.ABUSADORES_POR_CLIMA.items():
                if hab in habs:
                    uteis.add(clima)
            if hab in self.HABILIDADES_POR_CLIMA:
                uteis.add(self.HABILIDADES_POR_CLIMA[hab])
            for mv in getattr(m, "moves", {}).values():
                mid = getattr(mv, "id", "")
                for clima, golpes in self.GOLPES_POR_CLIMA.items():
                    if mid in golpes:
                        uteis.add(clima)
        return uteis

    def _clima_prejudica(self, clima, alvo):
        """Por este clima causa dano ou fraqueza ao alvo?

        Justifica poe-lo mesmo sem abusadores vivos do nosso lado.

        AREIA e a unica que causa dano residual (na Gen 9 a neve ja nao o faz: passou
        a subir a Defesa dos tipo Gelo). Sol e chuva nao ferem, mas enfraquecem
        metade do dano dos tipos opostos.
        """
        if not alvo:
            return False
        tipos = {t.name for t in getattr(alvo, "types", []) if t}
        hab = str(getattr(alvo, "ability", "")).lower()
        item = str(getattr(alvo, "item", "") or "").lower()

        if clima == 'areia':
            imune_tipo = bool(tipos & {"ROCK", "GROUND", "STEEL"})
            imune_hab = hab in ("magicguard", "overcoat", "sandveil", "sandrush",
                                "sandforce", "sandspit")
            return not (imune_tipo or imune_hab or item == "safetygoggles")
        if clima == 'sol':
            return "WATER" in tipos or hab == "dryskin"
        if clima == 'chuva':
            return "FIRE" in tipos
        return False

    # ==================================================================
    # PRESERVACAO DO SETTER DE CLIMA (30/08/2026)
    # ==================================================================
    # SO OS GOLPES QUE INSTALAM clima. O `GOLPES_POR_CLIMA` mistura instaladores
    # (Sunny Day) com aproveitadores (Solar Beam, Thunder), logo nao serve para
    # distinguir quem POE de quem USA. As habilidades vem do `HABILIDADES_POR_CLIMA`
    # que ja existe: uma tabela nova para a mesma pergunta acabaria por divergir da
    # antiga, e este projeto ja pagou isso vezes suficientes.
    GOLPES_QUE_INSTALAM = {
        'sunnyday': 'sol', 'raindance': 'chuva',
        'sandstorm': 'areia', 'hail': 'granizo', 'snowscape': 'granizo',
    }

    def _clima_que_instala(self, mon):
        """Que clima este Pokemon consegue instalar? None se nenhum."""
        hab = str(getattr(mon, "ability", "") or "").lower()
        if hab in self.HABILIDADES_POR_CLIMA:
            return self.HABILIDADES_POR_CLIMA[hab]
        for mv in (getattr(mon, "moves", None) or {}):
            clima = self.GOLPES_QUE_INSTALAM.get(str(mv).lower())
            if clima:
                return clima
        return None

    # ======================================================================
    # MATCHUP PERDIDO: TIPO **E** AMEACA (03/09/2026)
    # ======================================================================
    # O `get_matchup_state` le EFECTIVIDADE DE TIPO e mais nada. Um Pokemon com
    # matchup NEUTRAL contra quem o mata num golpe nao era "perdido" para regra
    # nenhuma, e ficava em campo a HP cheio. Foi assim que setters de clima
    # unicos morreram no turno 1.
    #
    # A leitura por AMEACA vive aqui e no `physics`, NUNCA na tupla de estado: a
    # dimensao 2 continua a ser efectividade de tipo, porque e essa a
    # representacao que o Blue e o Green estao a aprender, e mexer nela seria a
    # quarta mudanca de percepcao do mesmo ciclo (6.42).
    #
    # ---------------------------------------------------------------------
    # O CASO VOLATILE, e porque e CONDICIONAL e nao absoluto
    # ---------------------------------------------------------------------
    # VOLATILE e super-efectivo dos DOIS lados: quem mata primeiro ganha, quem
    # nao mata morre. Isso pede jogadas OPOSTAS conforme quem esta em campo:
    #
    #   Infernape (rapido, sobrevive ao troco) contra Corviknight  -> BATER.
    #       Correccao de 29/08: pivotar num VOLATILE sendo mais rapido e
    #       oferecer o primeiro golpe de graca. Ver `matchup_favoravel` no
    #       `_mod_lead`, que deixaria de correr se VOLATILE fosse perdido
    #       INCONDICIONALMENTE — a correccao do Infernape seria revertida sem
    #       ninguem dar por isso.
    #
    #   Tyranitar (setter unico de areia) contra qualquer coisa      -> SAIR.
    #       Para quem tem de sobreviver a partida inteira, uma moeda ao ar nao
    #       e jogada: perder o setter perde o plano, nao uma peca.
    #
    # A condicao une as duas leituras sem inventar regra nova: VOLATILE conta
    # como perdido quando somos o setter unico OU quando nao sobrevivemos ao
    # melhor golpe conhecido do adversario.

    _MATCHUPS_PERDIDOS = (MatchupState.DEFENSIVE_DIS,
                          MatchupState.CRITICAL_DIS,
                          MatchupState.OFFENSIVE_DIS)

    def _matchup_perdido(self, matchup, active, opponent, battle):
        """O confronto esta perdido, por TIPO ou por AMEACA?"""
        try:
            if matchup in self._MATCHUPS_PERDIDOS:
                return True

            # Nao sobrevivemos ao que ele ja mostrou: o tipo diz o que quiser.
            # `se_desconhecido=True` = sem golpe revelado NAO se assume o pior,
            # senao toda a abertura seria lida como perdida.
            if self.physics.sobrevive_a(active, opponent, battle,
                                        se_desconhecido=True) is False:
                return True

            if matchup == MatchupState.VOLATILE:
                return self._e_setter_unico(active, battle)

            return False
        except Exception:
            # Na duvida, o comportamento antigo: so o tipo decide.
            return matchup in self._MATCHUPS_PERDIDOS

    def _e_setter_unico(self, active, battle):
        """Vale a pena preservar este Pokemon POR CAUSA do clima que ele instala?

        TRES PERGUNTAS, e as tres tem de dar sim:

          1. ELE INSTALA algum clima? (habilidade ou golpe instalador)
          2. HA ALGUEM VIVO que aproveite esse clima? Sem isto o clima nao e
             estrategia de equipa, e uma habilidade que calhou de existir. Um
             Tyranitar com Sand Stream numa equipa sem Excadrill nem Sandslash e um
             atacante normal: perde-lo custa uma peca, nao custa o plano. Trocar
             para o preservar seria pagar um turno por nada.
          3. E O UNICO VIVO que instala ESSE clima? Com dois, perder um nao perde a
             condicao.

        O ABUSADOR PODE SER O PROPRIO SETTER (corrigido 30/08/2026). A versao
        anterior exigia o abusador NO BANCO, excluindo o ativo, com o argumento de
        que troca-lo tiraria de campo quem beneficia. O argumento nao se sustenta, e
        o caso que o derruba e comum: equipas de sol com Ninetales ou Charizard-Y,
        em que quem POE o sol e tambem quem o USA.

        Duas razoes para contar o proprio:
          - preservar e para MAIS TARDE, nao para este turno. Uma habilidade
            instaladora volta a disparar a cada entrada, logo o clima nao se perde
            por sair de campo: perde-se por MORRER.
          - quando o setter e o unico abusador, ele NAO e menos importante, e mais:
            a estrategia inteira esta numa peca so.

        Conta-se abusador por HABILIDADE (Swift Swim, Sand Rush, Solar Power) e por
        GOLPE (Solar Beam, Thunder, Aurora Veil), reaproveitando as tabelas que ja
        servem o `_climas_uteis`.

        LIMITE DECLARADO: habilidades de MEGA so aparecem depois da mega-evolucao.
        Um Charizard por mega-evoluir tem `ability` = Blaze ou Solar Power, nao
        Drought, logo conta como ABUSADOR e nao como instalador. Na pratica isso
        favorece a leitura certa (o Ninetales fica como instalador unico), mas fica
        registado por nao ser deliberado.
        """
        try:
            clima = self._clima_que_instala(active)
            if not clima:
                return False

            vivos = [m for m in battle.team.values() if not m.fainted]
            habs_abusadoras = self.ABUSADORES_POR_CLIMA.get(clima, set())
            golpes_abusadores = self.GOLPES_POR_CLIMA.get(clima, set()) - set(self.GOLPES_QUE_INSTALAM)

            # O ATIVO ENTRA NA CONTA (ver docstring): o setter e frequentemente um
            # dos abusadores, e nesse caso preserva-lo importa mais, nao menos.
            tem_abusador = False
            for m in vivos:
                if str(getattr(m, "ability", "") or "").lower() in habs_abusadoras:
                    tem_abusador = True
                    break
                if any(str(mv).lower() in golpes_abusadores for mv in (getattr(m, "moves", None) or {})):
                    tem_abusador = True
                    break
            if not tem_abusador:
                return False

            instaladores = [m for m in vivos if self._clima_que_instala(m) == clima]
            return len(instaladores) == 1
        except Exception:
            return False

    # ======================================================================
    # VALE A PENA GASTAR ESTE TURNO A POR HAZARD? (03/09/2026)
    # ======================================================================
    # A promocao do hazard era binaria e estava presa a abertura. Duas coisas
    # mudam:
    #
    # 1. O VALOR E PROPORCIONAL AS ENTRADAS QUE FALTAM, nao ao turno em que
    #    estamos. Stealth Rock com cinco adversarios vivos vale cinco entradas;
    #    com dois vale duas. Prender isto ao `macro_context` era usar o proxy
    #    errado. O filtro 8 do masking ja poda hazard com o adversario no ultimo
    #    Pokemon, logo aqui so se decide a PRIORIDADE.
    #
    # 2. AS DUAS GUARDAS QUE FALTAVAM, e a segunda foi pedida explicitamente:
    #    - SOBREVIVEMOS ao golpe? Por uma rocha e ficar em campo a receber um
    #      golpe que mata custa o turno, o Pokemon E o hazard, porque nem
    #      chegamos a agir se formos mais lentos.
    #    - SOMOS O SETTER UNICO DE CLIMA? Entao nao se troca o clima por uma
    #      rocha. Um Tyranitar com Stealth Rock e Sand Stream e o caso concreto,
    #      e esta no pool: as duas regras colidiam e a colisao era resolvida pela
    #      ORDEM DAS LINHAS no ficheiro, nao por decisao. Agora e decisao: o
    #      hazard so passa a frente se sobrevivermos a DOIS golpes, ou seja se
    #      houver turno para por a rocha E sair depois.
    LIMIAR_HAZARD_POUCAS_ENTRADAS = 2

    # O ADVERSARIO A MONTAR VENCE O HAZARD (03/09/2026).
    #
    # Observado em batalha manual: Volcarona usou Quiver Dance em turnos
    # seguidos, chegando a +2 ou +3 em Ataque Especial, Defesa Especial e
    # Velocidade, e o instinto respondeu com Stealth Rock nos dois turnos. Uma
    # rocha cobra 12,5% por entrada; um Volcarona a +3 ganha a partida sozinho.
    #
    # ISTO E TAMBEM UMA GUARDA CONTRA A PROPRIA REGRA GLOBAL 10. A regra promove
    # HAZARD em TODOS os modos, e nao so na abertura como antes. Sem esta
    # condicao, ela transformaria um erro de dois turnos num erro de todos os
    # turnos: quanto mais o adversario monta, mais entradas futuras existem para
    # a rocha cobrar, e mais a regra insistiria. Uma regra nova nunca pode
    # amplificar o defeito que se estava a corrigir.
    #
    # O limiar e a SOMA dos boosts ofensivos e de velocidade: dois estagios em
    # atributos diferentes ja mudam o confronto tanto como dois no mesmo.
    LIMIAR_BOOST_ADVERSARIO = 2

    def _adversario_a_montar(self, opponent):
        """O adversario acumulou boosts que mudam o confronto?"""
        try:
            boosts = getattr(opponent, "boosts", None) or {}
            return sum(max(0, int(boosts.get(k, 0) or 0))
                       for k in ("atk", "spa", "spe")) >= self.LIMIAR_BOOST_ADVERSARIO
        except Exception:
            return False

    def _vale_a_pena_hazard(self, active, opponent, battle):
        """Poe-se o hazard agora, ou ha coisa melhor a fazer com este turno?

        SONDA [HAZVALE] (04/09/2026) — TEMPORARIA, SAI QUANDO RESPONDER.
        PERGUNTA: porque e que o instinto deixou de por hazards. Nas quatro
        batalhas manuais de 04/09 o `HAZARD` aparece no ranking mas SEMPRE EM
        ULTIMO (turnos 23, 25 a 29 da batalha 1), e nunca e escolhido. A REGRA
        GLOBAL 10 devia fazer o CONTRARIO: promove-lo. E regressao introduzida em
        03/09 com a propria regra, e ha tres hipoteses indistinguiveis sem log:
          1. esta funcao devolve False, e o log diz em QUAL guarda
          2. a Regra 10 nao e alcancada (um `return` anterior no perfil)
          3. algo DEPOIS volta a despromover o HAZARD
        Se o log nao mostrar linha nenhuma num turno com HAZARD no ranking, e a
        hipotese 2; se mostrar `VALE=True` e o HAZARD ficar em ultimo, e a 3.
        """
        motivo = "?"
        try:
            # MIGRADO PARA `physics.equipa_adversaria` EM 04/09/2026. A leitura
            # antiga contava `vivos=1` no TURNO 1, com o adversario a ter seis, e
            # por isso a Regra Global 10 devolvia False quase sempre: o instinto
            # deixou de por hazards. Medido pela sonda [HAZVALE].
            vivos, _ = self.physics.equipa_adversaria(battle)
            if vivos <= self.LIMIAR_HAZARD_POUCAS_ENTRADAS:
                motivo = f"poucas entradas (vivos={vivos})"
                return False

            # Com o adversario a montar, o turno vale mais em qualquer outra
            # coisa: expulsar, bater ou sair. Ver o comentario acima.
            if self._adversario_a_montar(opponent):
                motivo = "adversario a montar"
                return False

            # Sem golpe revelado NAO se assume o pior: a abertura ficaria toda
            # sem hazard, que e a jogada que esta regra existe para promover.
            aguenta_um = self.physics.sobrevive_a(active, opponent, battle,
                                                  se_desconhecido=True)
            if aguenta_um is False:
                motivo = "nao sobrevive a um golpe"
                return False

            if self._e_setter_unico(active, battle):
                dois = self.physics.sobrevive_a(active, opponent, battle,
                                                golpes=2,
                                                se_desconhecido=True)
                motivo = f"setter unico, sobrevive a dois={dois}"
                return dois is not False
            motivo = "sem impedimento"
            return True
        except Exception as e:
            motivo = f"EXCEPCAO {type(e).__name__}: {e}"
            return True
        finally:
            diagnostico.log("HAZVALE", f"turno={getattr(battle, 'turn', '?')} "
                                       f"motivo={motivo}")

    def _mod_lead(self, base, active, opponent, battle, is_faster):
        modified = base.copy()
        my_team = list(battle.team.values())
        weather_abusers = ['swiftswim', 'chlorophyll', 'sandrush', 'slushrush', 'sandforce',
                           'solarpower', 'hydration', 'drought', 'drizzle', 'sandstream', 'snowwarning']
        team_needs_weather = any(str(m.ability).lower() in weather_abusers for m in my_team)
        avg_speed = sum(m.base_stats.get('spe', 50) for m in my_team) / len(my_team)
        team_needs_tr = avg_speed < 70
        needs_field_control = team_needs_weather or team_needs_tr
        matchup = self.parser.get_matchup_state(active, opponent)
        matchup_lost = self._matchup_perdido(matchup, active, opponent, battle)
        matchup_won = matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]

        if matchup_lost:
            # ==========================================================
            # REDE DE SEGURANCA DO PIVO (30/08/2026)
            # ==========================================================
            # O ramo antigo promovia ATTACK_PIVOT quando somos mais rapidos, e
            # SWITCH_DEFENSIVE apenas quando somos mais LENTOS. Faltava o caso
            # do meio, e ele e comum: mais rapido, matchup perdido, e SEM golpe
            # de pivo na equipa.
            #
            # `ATTACK_PIVOT` esta SEMPRE no template do LEAD (3a posicao), logo
            # a condicao `"ATTACK_PIVOT" in modified` da sempre True e o ramo
            # dispara mesmo sem nenhum U-turn no moveset. O executor devolve
            # None, o ranking avanca, e como SWITCH_DEFENSIVE esta em 11o lugar
            # ele nunca e alcancado: cai-se em ATTACK_STRONG e ataca-se de dentro
            # da desvantagem.
            #
            # Observado em batalha manual: Mega Tyranitar (Rock/Dark), que era o
            # setter de areia da equipa, contra Melmetal (Aco). Aco bate 2x em
            # Pedra, e o Superpower que o Melmetal revelou depois bate 4x. O
            # instinto usou Crunch e perdeu o setter no turno 1. O Tyranitar do
            # pool nao tem um unico golpe de pivo.
            #
            # A correccao mantem a intencao original — pivotar e melhor que
            # trocar, porque ainda faz dano — e so poe a troca LOGO ATRAS, em vez
            # de a deixar em 11o. E o `elif` deixa de exigir `not is_faster`:
            # ser mais rapido nao ajuda nada quando se morre num golpe.
            if is_faster and "ATTACK_PIVOT" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
                if "SWITCH_DEFENSIVE" in modified:
                    modified.insert(1, modified.pop(modified.index("SWITCH_DEFENSIVE")))
            elif "SWITCH_DEFENSIVE" in modified:
                modified.insert(0, modified.pop(modified.index("SWITCH_DEFENSIVE")))
            return modified
        # A PRESERVACAO DO SETTER SAIU DAQUI EM 03/09/2026.
        # Era uma regra de `_mod_lead`, logo so existia enquanto
        # `macro_context == "OPENING"` (total_alive >= 10). Bastavam tres Pokemon
        # desmaiados no total para o setter unico ficar sem guarda nenhuma, e o
        # caso comum e precisamente esse: o setter e guardado para por o clima
        # MAIS TARDE. Passou a REGRA GLOBAL 9, que corre em todos os modos. Ver
        # `get_instinct_profile`.

        # ==============================================================
        # MAIS RAPIDO E COM VANTAGEM: BATER, NAO PIVOTAR (29/08/2026)
        # ==============================================================
        # O template do LEAD poe `ATTACK_PIVOT` em 3o e `ATTACK_STRONG` em 5o, e o
        # modo LEAD dispara em TODA a abertura (`macro_context == "OPENING"`, ou seja
        # total_alive >= 10) SEM consultar o matchup: o `_get_tactical_mode` nem
        # chega a ser chamado.
        #
        # Caso observado: Infernape (108 de velocidade base) contra Corviknight (67),
        # matchup VOLATILE — Fogo 2x contra Aco, Brave Bird 2x contra Lutador. O
        # instinto usou U-turn no turno 1 em vez de bater com vantagem de velocidade
        # e de tipo.
        #
        # Porque escapou as regras que ja existiam: nao havia clima ativo (a regra de
        # baixo exige `weather_active`) e `matchup_won` NAO inclui VOLATILE.
        #
        # VOLATILE entra aqui de proposito: super-efetivo dos DOIS lados e exatamente
        # quando NAO se deve dar o turno ao adversario. Pivotar num VOLATILE sendo
        # mais rapido e oferecer o primeiro golpe de graca.
        #
        # Corre ANTES do bloco de FIELD_CONTROL para que este possa sobrepor-se: com
        # a equipa a precisar de clima, por o clima continua a valer mais que um
        # ataque no turno 1.
        matchup_favoravel = matchup_won or matchup == MatchupState.VOLATILE
        if is_faster and matchup_favoravel:
            if "ATTACK_STRONG" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
            if "ATTACK_PIVOT" in modified:
                # Despromover, nao remover: continua a ser jogavel se o ataque falhar.
                modified.append(modified.pop(modified.index("ATTACK_PIVOT")))

        # O clima por GOLPE (Sunny Day, Rain Dance...) so vale a pena se nao houver
        # forma de o obter de GRACA (29/08/2026). Se algum membro VIVO da equipa tem
        # habilidade de clima, o clima chega sozinho quando esse Pokemon entrar em
        # campo, e gastar um turno com o golpe e desperdicio.
        #
        # O caso que escapava: o setter esta NO BANCO. O clima ainda nao esta ativo,
        # logo `not weather_active` era verdadeiro, e o instinto gastava o turno com
        # o golpe quando bastava trocar mais tarde.
        #
        # A habilidade tem de estar num Pokemon VIVO: se o setter ja desmaiou, o
        # golpe volta a ser a unica via e deve subir na mesma.
        clima_de_graca = any(
            str(getattr(m, "ability", "")).lower() in self.HABILIDADES_POR_CLIMA
            for m in my_team if not getattr(m, "fainted", False)
        )

        # De QUEM e o clima que esta em campo, e a quem serve.
        clima_agora = self._clima_atual(battle)
        climas_uteis = self._climas_uteis(my_team)
        # RENOMEADA EM 30/08/2026, de `clima_ja_e_nosso` para `clima_favoravel`.
        #
        # O calculo sempre foi de UTILIDADE, nunca de POSSE: `climas_uteis` varre a
        # NOSSA equipa viva e nao olha para quem pos o clima. O nome antigo dizia
        # outra coisa e induzia em erro quem lesse.
        #
        # O caso que torna a distincao concreta e o ESPELHO: duas equipas de chuva,
        # com abusadores dos dois lados. A chuva posta pelo adversario serve-nos
        # exatamente na mesma. Tratar isso como "clima dele" e recusar o beneficio
        # seria absurdo, e e o que o nome antigo sugeria que o codigo fazia.
        clima_favoravel = clima_agora is not None and clima_agora in climas_uteis

        # Poe-se clima por GOLPE em dois casos, e so:
        #   1. temos abusadores VIVOS e o clima em campo ainda nao e o nosso
        #      (cobre o caso do adversario ter posto o dele: antes bloqueava)
        #   2. nao temos abusadores, mas o clima PREJUDICA o adversario
        #      (areia contra quem nao resiste, sol contra Agua, chuva contra Fogo)
        vale_a_pena = (
            (bool(climas_uteis) and not clima_favoravel)
            or any(self._clima_prejudica(c, opponent)
                   for c in (climas_uteis or self.ABUSADORES_POR_CLIMA))
        )

        # ==============================================================
        # HAZARD NO TURNO 1 (30/08/2026)
        # ==============================================================
        # O `_mod_lead` promovia EXATAMENTE DUAS coisas: FIELD_CONTROL (clima e
        # Trick Room) e ATTACK_STRONG. Nao havia uma unica linha sobre HAZARD.
        #
        # Observado em batalha manual: o lead e escolhido corretamente pela heuristica
        # de `get_best_lead`, entra em campo e NAO poe hazard nenhum. So os setters de
        # clima agiam como lead, porque eram os unicos a passar por FIELD_CONTROL.
        #
        # PORQUE O TURNO 1 E O MELHOR TURNO PARA HAZARD. O valor de Stealth Rock e
        # proporcional ao numero de entradas que o adversario ainda vai fazer, e esse
        # numero e MAXIMO no turno 1. Poe-lo no turno 20 vale uma fracao. Com hazards
        # em 91% dos times do pool, perder essa jogada em toda a abertura e das
        # maiores fatias de valor que o instinto deixava na mesa.
        #
        # ORDEM: fica DEPOIS do bloco de matchup perdido (que ja devolveu) e ANTES do
        # clima, mas so promove se o clima nao for necessario ou vier de graca. Com a
        # equipa a depender de clima ou de Trick Room, esses continuam a valer mais:
        # sao condicoes que mudam a partida inteira, e o hazard pode esperar um turno.
        #
        # A viabilidade nao se decide aqui: o filtro 8 do masking ja poda hazard com
        # o adversario no ultimo Pokemon, e o `is_hazard_already_set` ja impede a
        # reposicao. Aqui so se decide a PRIORIDADE.
        # ESCALADO PELOS ADVERSARIOS VIVOS, E COM DUAS GUARDAS (03/09/2026).
        # Ver `_vale_a_pena_hazard`: o valor do hazard e proporcional as entradas
        # que o adversario ainda vai fazer, e nao a estarmos ou nao na abertura.
        # E nao se troca o setter de clima por uma rocha.
        clima_pode_esperar = (not needs_field_control) or clima_de_graca or clima_favoravel
        if (clima_pode_esperar
                and self._vale_a_pena_hazard(active, opponent, battle)
                and "HAZARD" in modified):
            modified.insert(0, modified.pop(modified.index("HAZARD")))

        # TRICK ROOM tem caminho PROPRIO e passa a frente do clima.
        # Nao ha habilidade que o ative, logo o `clima_de_graca` nao se aplica: ou se
        # gasta o turno com o golpe, ou a equipa lenta joga sem ele. E a dependencia
        # e total — sem Trick Room, um time de Trick Room e so um time lento.
        equipa_tr = self.equipa_de_trick_room(my_team)
        tr_ativo = self.trick_room_ativo(battle)

        if equipa_tr and not tr_ativo:
            if "FIELD_CONTROL" in modified:
                modified.insert(0, modified.pop(modified.index("FIELD_CONTROL")))
        elif needs_field_control and vale_a_pena and not clima_de_graca:
            if "FIELD_CONTROL" in modified:
                modified.insert(0, modified.pop(modified.index("FIELD_CONTROL")))
        elif clima_favoravel and matchup_won:
            # Idem: por hazards em vez de tratar do clima so faz sentido se o clima
            # que esta em campo ja e o NOSSO. Com o do adversario, ha coisa melhor
            # a fazer.
            if "HAZARD" in modified:
                modified.insert(0, modified.pop(modified.index("HAZARD")))
        # Bater com clima a favor: exige que o clima seja NOSSO, nao apenas que
        # exista. Com a areia do adversario em campo, atacar nao ganha nada por isso.
        if clima_favoravel and is_faster:
            if "ATTACK_STRONG" in modified:
                modified.insert(0, modified.pop(modified.index("ATTACK_STRONG")))
        return modified

    def _mod_wallbreak(self, base, active, opponent, is_faster, my_hp_frac, opp_hp_frac):
        modified = base.copy()
        opponent_has_recovery = self._has_recovery(opponent)
        if opponent_has_recovery:
            if "STATUS" in modified:
                modified.insert(0, modified.pop(modified.index("STATUS")))
            if "ATTACK_TECH" in modified:
                modified.insert(1, modified.pop(modified.index("ATTACK_TECH")))
        if "BUFF" in modified and my_hp_frac >= 0.60:
            insert_idx = 2 if opponent_has_recovery else 0
            modified.insert(insert_idx, modified.pop(modified.index("BUFF")))
        if my_hp_frac < 0.40 and "ATTACK_PIVOT" in modified:
            modified.insert(0, modified.pop(modified.index("ATTACK_PIVOT")))
        return modified

    # ======================================================================
    # ORQUESTRADOR: produz o ranking final de intenções
    # ======================================================================

    # ==================================================================
    # TRAPPING POR HABILIDADE (28/08/2026)
    # ==================================================================
    # Cobre APENAS as tres habilidades, que sao o caso limpo: dependem so do tipo
    # do alvo e da habilidade de quem prende, ambos conhecidos com certeza para o
    # NOSSO ativo. Ficam de fora, por decisao explicita:
    #
    #   - golpes de bind (bind, wrap, firespin, magmastorm, whirlpool, infestation,
    #     sandtomb). Duracao variavel e deteccao menos fiavel. NOTA IMPORTANTE:
    #     nenhum deles esta em `tech_moves` no physics.classify_move, logo o instinto
    #     trata Magma Storm como um golpe de fogo qualquer e NAO SABE QUE PRENDE.
    #   - Mean Look, Block, Spider Web e Ingrain.
    #
    # Sao trabalho futuro e completam o mecanismo pelo lado dos GOLPES; estas tres
    # cobrem-no pelo lado das HABILIDADES.

    IMUNES_SHADOW_TAG = {"GHOST"}

    @staticmethod
    def _ability(mon):
        return str(getattr(mon, "ability", "") or "").lower()

    # Efeitos de campo que impedem a saida. Cobrem os golpes de prisao (bind e
    # afins), o Mean Look e companhia, e o Octolock. Complementam as habilidades:
    # juntos, os dois lados do trapping ficam cobertos.
    EFEITOS_QUE_PRENDEM = {
        "BIND", "WRAP", "INFESTATION", "FIRE_SPIN", "WHIRLPOOL", "SAND_TOMB",
        "MAGMA_STORM", "CLAMP", "SNAP_TRAP", "THUNDER_CAGE",
        "MEAN_LOOK", "BLOCK", "SPIDER_WEB", "OCTOLOCK", "NO_RETREAT", "INGRAIN",
    }

    def _preso_por_efeito(self, mon):
        """`mon` esta preso por um EFEITO em campo (golpe de prisao, Mean Look...)?

        ACRESCENTADO 28/08/2026. Ate aqui a Regra 6 so reconhecia trapping por
        HABILIDADE, o que criava uma assimetria: se o Heatran prendesse um Ferrothorn
        com Magma Storm, o instinto nao sabia que o tinha prendido e nao aplicava o
        "monta e depois bate". Agora os dois lados do trapping ficam cobertos.
        """
        if mon is None:
            return False
        try:
            # MIGRADO PARA `physics.nomes_de` EM 03/09/2026: trapping por EFEITO
            # nunca era detectado, logo o instinto tentava trocar preso.
            efeitos = self.physics.nomes_de(getattr(mon, "effects", None))
        except Exception:
            return False
        return bool(efeitos & self.EFEITOS_QUE_PRENDEM)

    def _preso_por(self, preso, prendedor):
        """`prendedor` impede `preso` de sair, por HABILIDADE ou por EFEITO."""
        if self._preso_por_efeito(preso):
            return True
        if not preso or not prendedor:
            return False
        hab = self._ability(prendedor)
        tipos = {t.name for t in getattr(preso, "types", []) if t}
        item = str(getattr(preso, "item", "") or "").lower()

        if hab == "shadowtag":
            # Shadow Tag nao prende Fantasmas nem outro Shadow Tag.
            return not (tipos & self.IMUNES_SHADOW_TAG) and self._ability(preso) != "shadowtag"
        if hab == "arenatrap":
            # So prende quem toca no chao.
            return ("FLYING" not in tipos
                    and self._ability(preso) != "levitate"
                    and item != "airballoon")
        if hab == "magnetpull":
            return "STEEL" in tipos
        return False

    def _morre_depressa(self, alvo, atacante, battle, has_lethal):
        """O alvo cai em ~2 turnos? Se sim, montar e desperdicar a janela."""
        if has_lethal:
            return True
        try:
            hp = float(getattr(alvo, "current_hp_fraction", 1.0))
            melhor = 0.0
            for mv in battle.available_moves:
                if mv.base_power > 0:
                    melhor = max(melhor,
                                 self.physics.estimate_damage_percent(mv, atacante, alvo, battle))
            return melhor * 2 >= hp
        except Exception:
            return False

    def _assinatura_posicao(self, battle, active, opp, matchup):
        """Assinatura ABSTRATA da posicao, para detetar impasses.

        Nao inclui o turno nem identidades: e a POSICAO que interessa, nao as pecas.
        Foi isso que fez a quarentena por par de especies falhar — o ciclo tinha 5 ou
        6 membros e cada troca parecia produtiva porque o par era sempre novo.

        Os buckets de HP substituem a condicao "o HP nao mexeu": um tique de veneno ou
        uma cura de Leftovers NAO mudam de bucket, mas 30% de dano real muda. Assim a
        guarda solta-se sozinha quando alguma coisa relevante acontece, sem ser
        enganada por dano residual.

        O macro_context entra de proposito: e o indice 14 do estado que o cerebro ve,
        logo o Blue e o Green podem APRENDER sobre esta situacao em vez de apenas
        sofrerem a consequencia dela.
        """
        try:
            return (
                self.physics.get_role(active, battle).name,
                self.physics.get_role(opp, battle).name,
                matchup.name,
                self.parser.get_hp_bucket(active),
                self.parser.get_hp_bucket(opp),
                self.parser.get_macro_context(battle),
            )
        except Exception:
            return None

    def _registar_posicao(self, battle, assinatura):
        """Guarda as ultimas posicoes desta batalha e diz se JA se repetiu."""
        if assinatura is None:
            return False
        if not hasattr(self, "_posicoes"):
            self._posicoes = {}
        tag = getattr(battle, "battle_tag", None)
        hist = self._posicoes.setdefault(tag, [])
        repetida = assinatura in hist
        hist.append(assinatura)
        if len(hist) > self.JANELA_IMPASSE:
            del hist[0]
        return repetida

    def limpar_posicoes(self, battle_tag):
        if hasattr(self, "_posicoes"):
            self._posicoes.pop(battle_tag, None)

    JANELA_IMPASSE = 4

    @staticmethod
    def _promover(ranking, intents):
        for intent in reversed(intents):
            if intent in ranking:
                ranking.remove(intent)
                ranking.insert(0, intent)

    @staticmethod
    def _despromover(ranking, intents):
        for intent in intents:
            if intent in ranking:
                ranking.remove(intent)
                ranking.append(intent)

    def get_instinct_profile(self, battle, history=None):
        # `history` PASSADO EM 04/09/2026. O parametro existia com `None` por
        # omissao e NENHUM dos sete chamadores o passava, logo o filtro do Wish
        # (que depende de memoria de turnos) NUNCA podava: observado Wish em
        # turnos seguidos, com "But it failed!". Wish e Future Sight sao SLOT
        # conditions e o poke-env nao as expoe em `side_conditions` nem em
        # `effects`, logo a memoria tem de ser nossa e tem de CHEGAR ao masking.
        candidate_mask = self.masker.get_available_actions(battle, history)

        # CORRIGIDO: early-return agora devolve 5 valores (era 4 no monólito).
        if not battle.active_pokemon or not battle.opponent_active_pokemon:
            primary = "SWITCH_DEFENSIVE" if battle.available_switches else "ATTACK_STRONG"
            return (primary, 1.0, [primary], candidate_mask, False)

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        # Com `battle`: sob Trick Room o papel inverte-se (ver `get_role`).
        my_role = self.physics.get_role(active, battle)
        opp_role = self.physics.get_role(opp, battle)
        matchup = self.parser.get_matchup_state(active, opp)
        # MIGRADO PARA `physics.mais_rapido` EM 30/08/2026. A comparacao crua
        # `estimate_stat(a,'spe') > estimate_stat(b,'spe')` ignorava clima, terreno,
        # Tailwind e — pior — o Trick Room, que INVERTE a ordem de accao. O
        # comparador aplica os modificadores aos dois lados e inverte sob Trick Room.
        # Esta e a leitura central: alimenta o `_get_tactical_mode` e os quinze
        # `_mod_*`. Era aqui que o Trick Room falhava de forma mais cara — o item 17
        # ensinou o instinto a construir a partida a volta dele, e depois o instinto
        # continuava a jogar como se fosse lento.
        is_faster = self.physics.mais_rapido(active, opp, battle)
        my_hp_frac = active.current_hp_fraction
        opp_hp_frac = opp.current_hp_fraction
        is_threat = self.is_threatening(active, opp, battle)
        macro_context = self.parser.get_macro_context(battle)

        # Sub-camada 1: escolhe o modo
        if macro_context == "OPENING":
            mode = TacticalMode.LEAD
        else:
            mode = self._get_tactical_mode(matchup, my_role, opp_role, is_faster,
                                           my_hp_frac, opp_hp_frac, is_threat, active, opp, battle)

        # Sub-camada 2: template base
        base_priorities = self.mode_templates[mode].copy()

        # Sub-camada 3: modificador de role
        if mode == TacticalMode.LEAD:
            priorities = self._mod_lead(base_priorities, active, opp, battle, is_faster)
        elif mode == TacticalMode.WALLBREAK:
            priorities = self._mod_wallbreak(base_priorities, active, opp, is_faster, my_hp_frac, opp_hp_frac)
        else:
            modifier_fn = self.role_modifiers.get((my_role, opp_role))
            if modifier_fn:
                priorities = modifier_fn(base_priorities, active, opp, is_faster, my_hp_frac, opp_hp_frac, is_threat)
            else:
                priorities = base_priorities

        # Filtragem pelo action mask + regras de bom-senso
        my_hp_crit = my_hp_frac <= 0.35
        my_hp_full = my_hp_frac >= 0.85
        ranking_list = []
        for intent in priorities:
            if intent in candidate_mask:
                if intent == "HEAL" and my_hp_full:
                    # EXCEÇÃO WISH (29/08/2026). Esta poda é ao nível da INTENÇÃO,
                    # logo apaga a categoria HEAL inteira e não distingue golpes. O
                    # Wish cura no fim do turno SEGUINTE e pode curar um aliado, por
                    # isso é legítimo com HP cheio — e é a razão de se levar Wish em
                    # vez de Recover. Se houver um Wish jogável, a intenção sobrevive;
                    # a escolha de QUAL golpe usar dentro dela é do executor, e o
                    # masking já poda os curativos imediatos a full.
                    tem_wish = any(
                        getattr(m, "id", "") == "wish"
                        and not self.masker.is_move_useless(m, opp, battle, history)
                        for m in getattr(battle, "available_moves", [])
                    )
                    if not tem_wish:
                        continue
                if intent == "BUFF" and my_hp_crit:
                    continue
                if intent == "STATUS" and opp.status is not None:
                    continue
                if intent not in ranking_list:
                    ranking_list.append(intent)

        # Hazards desvalorizam quando restam poucos oponentes
        opp_alive, _ = self.physics.equipa_adversaria(battle)
        if opp_alive <= 2 and "HAZARD" in ranking_list:
            ranking_list.remove("HAZARD")
            ranking_list.append("HAZARD")

        # Barreiras (screens) reorganizam a macro-estratégia
        opp_side_conds = nomes_de_enum(battle.opponent_side_conditions.keys())
        physical_blocked = 'REFLECT' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds
        special_blocked = 'LIGHT_SCREEN' in opp_side_conds or 'AURORA_VEIL' in opp_side_conds
        if physical_blocked or special_blocked:
            benched_mons = [m for m in battle.team.values() if not m.fainted and not m.active]
            can_bypass = False
            if physical_blocked and not special_blocked:
                if any(self.physics.get_role(m) == Role.SWEEPER and not self.physics._is_physical(m) for m in benched_mons):
                    can_bypass = True
            elif special_blocked and not physical_blocked:
                if any(self.physics.get_role(m) == Role.SWEEPER and self.physics._is_physical(m) for m in benched_mons):
                    can_bypass = True
            base = ["CLEAN_HAZARD"]
            if can_bypass:
                boost_intents = base + ["SWITCH_OFFENSIVE", "ATTACK_PIVOT", "BUFF"]
            else:
                boost_intents = base + ["SWITCH_DEFENSIVE", "STATUS", "HEAL", "PROTECT", "DEBUFF"]
            for b_intent in reversed(boost_intents):
                if b_intent in ranking_list:
                    ranking_list.remove(b_intent)
                    ranking_list.insert(0, b_intent)

        # ------------------------------------------------------------------
        # URGENCIA DE SAIDA — REMOVIDA EM 24/08/2026 (era codigo morto)
        # ------------------------------------------------------------------
        # O bloco que aqui existia fazia `priorities = self._promover_saida(...)`.
        # Mas `ranking_list` ja foi construida a partir de `priorities` acima, e
        # `priorities` NUNCA MAIS e lida: o que a funcao devolve e `ranking_list`.
        # Logo a promocao nao tinha efeito nenhum e a reformulacao do ESCAPE
        # registada em 6.16 nunca chegou a correr. Quem resolveu o ciclo de trocas
        # foi a quarentena por matchup no execution.py, essa sim ativa.
        #
        # DECISAO (opcao C): apagado em vez de reanimado. Reanimar poria
        # SWITCH_DEFENSIVE no topo absoluto, que e exatamente o padrao que gerava a
        # alternancia infinita. Os metodos _urgencia_de_saida e _promover_saida
        # foram REMOVIDOS do ficheiro por ja nao terem chamador.

        # ------------------------------------------------------------------
        # REGRA GLOBAL 1: NAO BUFAR EM SITUACAO INSTAVEL
        # ------------------------------------------------------------------
        # Sintoma observado: o agente bufava em matchups VOLATILE ou piores, onde a
        # probabilidade de perder o Pokemon (e o buff) no turno seguinte e alta.
        # Bufar so compensa quando ha margem para aproveitar o buff.
        matchup_atual = self.parser.get_matchup_state(active, opp)
        matchups_inseguros = {
            MatchupState.VOLATILE, MatchupState.OFFENSIVE_DIS,
            MatchupState.DEFENSIVE_DIS, MatchupState.CRITICAL_DIS,
        }
        hp_frac = getattr(active, "current_hp_fraction", 1.0)
        if (matchup_atual in matchups_inseguros or hp_frac < 0.5) and "BUFF" in ranking_list:
            ranking_list.remove("BUFF")
            ranking_list.append("BUFF")

        # ------------------------------------------------------------------
        # REGRA GLOBAL 2: NAO DEITAR FORA UM BUFF JA CONQUISTADO
        # ------------------------------------------------------------------
        # Sintoma observado: trocas em excesso, perdendo boosts conquistados em turnos
        # anteriores. Um boost ofensivo/velocidade so existe enquanto o Pokemon estiver
        # em campo: trocar apaga-o. Com buff ativo e situacao sustentavel, atacar deve
        # subir e trocar deve descer.
        # REESCRITA EM 24/08/2026. A versao anterior preservava o buff sempre que o
        # matchup nao fosse CRITICAL_DIS e o HP >= 35%, o que na pratica produzia
        # "manter o buff ate morrer". O erro de fundo era tratar o buff como coisa a
        # PRESERVAR. Um buff nao se preserva, CONVERTE-SE: so vale enquanto houver
        # como o transformar em dano (ofensivo) ou em turnos (defensivo). Quando o
        # matchup vira ou o HP cai, o buff ja esta perdido de qualquer forma; a unica
        # questao que resta e se se perde o buff E o Pokemon, ou so o buff.
        #
        # Nota importante sobre o ramo "nao convertivel": ele NAO promove a troca,
        # apenas deixa de a bloquear. Promover trocas e o padrao que gerava o ciclo
        # de alternancia infinita (ver 6.16). Aqui a policy limita-se a parar de
        # interferir e o modo tatico decide, que e o comportamento normal.
        boosts = getattr(active, "boosts", {}) or {}
        buff_ofensivo = any(boosts.get(k, 0) > 0 for k in ("atk", "spa", "spe"))
        buff_defensivo = any(boosts.get(k, 0) > 0 for k in ("def", "spd"))

        # Fronteira de HP alinhada com o BUCKET do StateParser, para que o cerebro
        # veja exatamente a mesma fronteira que a policy usa (antes eram 0.35 aqui e
        # buckets la, duas reguas diferentes para a mesma nocao).
        hp_bucket = self.parser.get_hp_bucket(active)
        hp_abaixo_de_safe = hp_bucket in ("DANGER", "CRIT")

        # --- Buff OFENSIVO: converte-se em DANO, logo precisa de agir ---
        # Nao convertivel se: matchup de desvantagem ofensiva, HP abaixo de SAFE, ou
        # o adversario ameaca e joga primeiro (nesse caso o buff nem chega ao proximo
        # turno: ficar e dar o Pokemon de graca).
        ofensivo_convertivel = (
            buff_ofensivo
            and matchup_atual not in {MatchupState.CRITICAL_DIS, MatchupState.OFFENSIVE_DIS}
            and not hp_abaixo_de_safe
            and not (is_threat and not is_faster)
        )

        # --- Buff DEFENSIVO: converte-se em TURNOS, logo precisa de aguentar ---
        # Tres condicoes proprias, porque um buff defensivo faz o Pokemon querer FICAR
        # em campo, que e exatamente o oposto do ofensivo.
        defensivo_convertivel = False
        if buff_defensivo:
            # 1. A defesa subida e a CERTA para este atacante? Subir Def contra um
            #    atacante especial nao serve de nada.
            opp_fisico = self.physics._is_physical(opp)
            defesa_certa = (boosts.get("def", 0) > 0) if opp_fisico else (boosts.get("spd", 0) > 0)

            # 2. Desvantagem drastica de tipagem anula a preservacao. VOLATILE fica
            #    de fora deste conjunto porque e super-efetivo dos DOIS lados: sendo
            #    mais rapido e um bom matchup para um buff defensivo. E tratado em (4).
            matchup_drastico = matchup_atual in {MatchupState.CRITICAL_DIS,
                                                 MatchupState.DEFENSIVE_DIS}

            # 3. Limiar de vida pelo DANO REAL do ultimo turno, no MESMO confronto.
            #    Margem: sobreviver a 2 golpes; 1 basta com Regenerator, porque o
            #    Pokemon recupera 1/3 ao sair e pode dar-se ao luxo de levar mais um.
            aguenta = True
            if history is not None:
                mesmo_confronto = (
                    history.get("last_active_id") == getattr(active, "species", None)
                    and history.get("last_opponent_id") == getattr(opp, "species", None)
                )
                hp_antes = history.get("last_my_hp")
                if mesmo_confronto and hp_antes is not None:
                    dano_sofrido = float(hp_antes) - float(hp_frac)
                    if dano_sofrido > 0:
                        abi = str(getattr(active, "ability", "")).lower()
                        golpes_de_margem = 1 if abi == "regenerator" else 2
                        aguenta = hp_frac > dano_sofrido * golpes_de_margem

            # 4. Mais lento em VOLATILE: os dois lados batem super-efetivo e ele bate
            #    primeiro. Nenhum buff defensivo compensa levar o golpe antes de agir.
            #    Vale sobretudo para tanks, mas aplica-se a qualquer papel.
            volatile_lento = (matchup_atual == MatchupState.VOLATILE and not is_faster)

            defensivo_convertivel = (
                defesa_certa and not matchup_drastico and aguenta and not volatile_lento
            )

        if ofensivo_convertivel or defensivo_convertivel:
            for intent in reversed(["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_TECH"]):
                if intent in ranking_list:
                    ranking_list.remove(intent)
                    ranking_list.insert(0, intent)
            # Trocar apaga o buff: desce para o fim.
            for intent in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]:
                if intent in ranking_list:
                    ranking_list.remove(intent)
                    ranking_list.append(intent)

        # ------------------------------------------------------------------
        # CORRIDA DE BUFFS: nao bufar indefinidamente
        # ------------------------------------------------------------------
        # Sem isto, BUFF pode manter-se no topo turno apos turno e criar um ciclo do
        # mesmo tipo do das trocas: o estado repete-se, a acao repete-se, e a batalha
        # arrasta-se sem dano. Um buff (ou dois em posicao dominante) chega; a partir
        # dai o valor esta em converter a vantagem em dano.
        # REANIMADA EM 24/08/2026: operava sobre `priorities`, que ja nao e lida
        # depois de `ranking_list` ser construida. Passou a operar sobre
        # `ranking_list`, que e o que a funcao devolve. Esta e a UNICA das tres
        # regras mortas que foi reativada (opcao C): nao mexe em intencoes de troca,
        # logo nao pode reintroduzir o ciclo de alternancia.
        if history is not None:
            buffs_seguidos = int(history.get("buffs_consecutivos", 0) or 0)
            limite = 2 if matchup == MatchupState.DOMINANT else 1
            if buffs_seguidos >= limite and "BUFF" in ranking_list:
                ranking_list.remove("BUFF")
                ranking_list.append("BUFF")
                if "ATTACK_STRONG" in ranking_list:
                    ranking_list.remove("ATTACK_STRONG")
                    ranking_list.insert(0, "ATTACK_STRONG")

        # ==================================================================
        # REGRA GLOBAL 6: TRAPPING POR HABILIDADE (28/08/2026)
        # ==================================================================
        # As duas situacoes sao ASSIMETRICAS e nada tem a ver uma com a outra.
        opp_buffado = any((getattr(opp, "boosts", {}) or {}).get(k, 0) > 0
                          for k in ("atk", "spa", "spe", "def", "spd"))

        # Cobre os dois mecanismos: habilidade (Shadow Tag, Arena Trap, Magnet Pull)
        # e efeito em campo (golpes de prisao, Mean Look, Octolock).
        nos_prendemos = self._preso_por(opp, active)
        estamos_presos = self._preso_por(active, opp)

        if nos_prendemos and not estamos_presos:
            # PRENDER E UMA DECISAO NOSSA, e so se toma por vantagem. O adversario
            # nao pode sair, logo o TEMPO E NOSSO: montar primeiro, bater depois.
            #
            # DUAS EXCECOES em que montar e erro:
            #   1. o alvo morre em ~2 turnos -> a janela e para o MATAR, nao para
            #      montar. Magnet Pull existe para eliminar Acos, nao para bufar.
            #   2. nao aguentamos o que ele faz -> prender tambem nos prende ao lado
            #      dele. Um Magnezone que prende um Heatran com Magma Storm morre
            #      enquanto monta.
            # `has_lethal` so e calculado mais abaixo (deteccao de letalidade), logo
            # aqui passa-se False e o _morre_depressa decide pela estimativa de dano.
            # Conservador na direcao certa: na duvida monta-se em vez de se atacar.
            alvo_frageil = self._morre_depressa(opp, active, battle, False)
            seguro = (matchup_atual not in {MatchupState.CRITICAL_DIS,
                                            MatchupState.DEFENSIVE_DIS,
                                            MatchupState.VOLATILE}
                      and self.parser.get_hp_bucket(active) in ("FULL", "SAFE"))

            if alvo_frageil or not seguro:
                self._promover(ranking_list, ["ATTACK_STRONG", "ATTACK_TECH"])
            else:
                self._promover(ranking_list, ["BUFF", "HAZARD", "FIELD_CONTROL",
                                              "ATTACK_STRONG", "ATTACK_TECH"])
            # Sair agora desperdicava a janela que nos proprios criamos.
            self._despromover(ranking_list, ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
                                             "ATTACK_PIVOT"])

        elif estamos_presos:
            # A troca esta fora da mesa: a unica saida e RESOLVER a ameaca. O como
            # depende do que temos em campo.
            #
            # O PHAZE entra aqui como ferramenta de FUGA: expulsar o adversario
            # remove-o de campo e com ele a habilidade que nos prende. E a unica
            # intencao que quebra o trapping sem depender de o matar.
            #
            # E o contexto em que o DEBUFF finalmente compensa: quem nos prende NAO
            # VAI SAIR, logo o debuff nao se perde com a troca. Era este o criterio
            # de "preso" que faltava para o justificar.
            try:
                papel = self.physics.get_role(active).name
            except Exception:
                papel = "TANK"

            if papel == "SWEEPER":
                self._promover(ranking_list, ["ATTACK_STRONG", "ATTACK_TECH",
                                              "PHAZE", "BUFF"])
            else:
                self._promover(ranking_list, ["STATUS", "HEAL", "PHAZE",
                                              "DEBUFF", "ATTACK_TECH"])
            self._despromover(ranking_list, ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
                                             "ATTACK_PIVOT"])

        # ==================================================================
        # REGRA GLOBAL 9: PRESERVAR O SETTER UNICO (03/09/2026)
        # ==================================================================
        # MIGRADA DE `_mod_lead`, ONDE SO EXISTIA NA ABERTURA. Estava presa a
        # `macro_context == "OPENING"` (total_alive >= 10): bastavam tres Pokemon
        # desmaiados no total para desaparecer. E o caso comum e esse — o setter e
        # guardado para por o clima MAIS TARDE, ou seja precisamente quando a
        # regra ja nao corria.
        #
        # DIFERENCA FACE A REGRA 8. A 8 trata do setter que JA POS o clima e cuja
        # utilidade em campo acabou. Esta trata do setter que AINDA NAO POS, ou
        # cujo clima nao esta em campo: o que se protege aqui nao e a posicao, e a
        # PECA que instala a condicao. As duas coexistem e nao se sobrepoem.
        #
        # TRES CONDICOES:
        #   1. E o UNICO vivo que instala aquele clima, e ha quem o aproveite
        #      (ver `_e_setter_unico`).
        #   2. NAO temos vantagem de matchup. Com vantagem, ficar e bater e
        #      correcto: o setter tambem e uma peca e nao se troca por medo.
        #   3. NAO sobrevivemos ao melhor golpe conhecido do adversario.
        #
        # A CONDICAO 3 SUBSTITUI `not is_faster or hp < 0.60` (03/09/2026). A
        # antiga argumentava que ser mais rapido e estar inteiro da margem para
        # agir antes de decidir. **E falso quando o adversario mata num golpe:**
        # ser mais rapido so ajuda se matarmos primeiro ou se sobrevivermos ao
        # troco. Era a mesma pergunta errada que o atalho de pivo fazia com
        # `is_faster`, e a resposta certa esta no `physics.sobrevive_a`.
        #
        # Observado em batalha manual: Mega Tyranitar com Sand Stream, unico
        # setter da equipa, ficou em campo e morreu no turno 1. A areia acabou
        # junto com ele.
        #
        # PROMOVE a troca, NAO remove o ataque: sem candidato de troca (equipa
        # presa, banco vazio) o executor devolve None e o ranking desce.
        #
        # ORDEM: antes da Regra 7, como a Regra 8, porque PROMOVE trocas — e foi
        # promover trocas que gerou o carrossel de tanques da 6.28. Se a posicao
        # se repetir, a Regra 7 despromove-as a seguir e tem a ultima palavra.
        # ==================================================================
        # CORRECCAO DE 04/09/2026: A CONDICAO 3 ERA CEGA NO TURNO DA ENTRADA
        # ==================================================================
        # Em 03/09 substitui `not is_faster or hp < 0.60` por SO a sobrevivencia,
        # com o argumento de que ser rapido nao ajuda contra quem mata num golpe.
        # O argumento continua certo, mas a substituicao foi uma REGRESSAO minha:
        #
        #   `sobrevive_a` mede o dano dos golpes JA REVELADOS. Um Pokemon que
        #   ACABOU DE ENTRAR nao revelou nenhum, logo o dano conhecido e 0, logo
        #   a funcao devolve `se_desconhecido` e a regra NUNCA dispara — e
        #   precisamente no turno da entrada que ela e precisa.
        #
        # Observado em batalha manual: Tyranitar (ROCK/DARK), unico setter de
        # areia, ficou em campo contra um Kommo-o (DRAGON/FIGHTING). Luta e 4x
        # contra Rocha/Sombrio; o Close Combat tirou 71% e a areia acabou com ele.
        # O matchup por TIPO ja dizia CRITICAL_DIS desde o primeiro turno.
        #
        # As duas leituras respondem a perguntas diferentes e nenhuma substitui a
        # outra: o TIPO sabe antes de haver dados, a AMEACA sabe depois de os
        # haver. A condicao passa a ser a UNIAO das duas. E a mesma licao das
        # outras duas regressoes de hoje: nao trocar uma leitura grosseira que
        # funciona por uma precisa que ainda nao tem dados.
        if self._e_setter_unico(active, battle) and matchup_atual not in {
                MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV}:
            _nao_aguenta = self.physics.sobrevive_a(active, opp, battle,
                                                    se_desconhecido=True) is False
            _tipo_perdido = matchup_atual in self._MATCHUPS_PERDIDOS
            if _nao_aguenta or _tipo_perdido:
                self._promover(ranking_list, ["ATTACK_PIVOT", "SWITCH_DEFENSIVE"])

        # ==================================================================
        # REGRA GLOBAL 10: HAZARD VALE PELAS ENTRADAS QUE FALTAM (03/09/2026)
        # ==================================================================
        # A promocao de hazard so existia no `_mod_lead`, ou seja so na abertura,
        # com o argumento de que o turno 1 e o melhor turno para o por. O
        # argumento e verdadeiro mas o proxy estava errado: o valor do Stealth
        # Rock e proporcional ao numero de ENTRADAS que o adversario ainda vai
        # fazer, e esse numero nao cai a zero quando a abertura acaba. Com cinco
        # adversarios vivos no turno 12 o hazard continua a valer cinco entradas.
        #
        # Reaproveita `_vale_a_pena_hazard`, que ja carrega as duas guardas: nao
        # se poe hazard quando nao se sobrevive ao golpe, e nao se troca o setter
        # unico de clima por uma rocha.
        #
        # A viabilidade nao se decide aqui: o filtro 8 do masking poda hazard com
        # o adversario no ultimo Pokemon e o `is_hazard_already_set` impede a
        # reposicao. Aqui so se decide a PRIORIDADE.
        # SONDA [REGRA10] (04/09/2026) — TEMPORARIA. Marca que a regra FOI
        # ALCANCADA. Se um turno tiver HAZARD no ranking e nao tiver esta linha,
        # a causa e um `return` anterior no perfil (hipotese 2), e nao esta
        # funcao. Imprime a posicao do HAZARD ANTES e DEPOIS de promover, o que
        # separa a hipotese 3: se sair 0 aqui e o HAZARD acabar em ultimo no
        # [DECISAO], alguem o despromoveu a seguir.
        _vale = self._vale_a_pena_hazard(active, opp, battle)
        _antes = ranking_list.index("HAZARD") if "HAZARD" in ranking_list else None
        if _vale:
            self._promover(ranking_list, ["HAZARD"])
        elif self._adversario_a_montar(opp):
            # NAO CHEGA NAO POR HAZARD: tem de haver resposta ao setup. O PHAZE e
            # a resposta directa (a expulsao APAGA os boosts e ainda cobra a
            # entrada seguinte); sem golpe de PHAZE no moveset o executor devolve
            # None e o ranking desce sozinho para o ataque.
            self._promover(ranking_list, ["PHAZE", "ATTACK_STRONG"])
        diagnostico.log("REGRA10", f"turno={getattr(battle, 'turn', '?')} "
                                   f"vale={_vale} pos_antes={_antes} pos_depois="
                                   f"{ranking_list.index('HAZARD') if 'HAZARD' in ranking_list else None}")

        # ==================================================================
        # REGRA GLOBAL 8: O SETTER JA CUMPRIU — SAIR PARA O ABUSADOR (29/08/2026)
        # ==================================================================
        # Um setter de clima por HABILIDADE poe o clima ao ENTRAR: a partir desse
        # momento, a funcao principal dele esta feita. Ficar em campo raramente
        # produz — quem tira partido do sol nao e o Torkoal, sao os abusadores no
        # banco. E manter o setter vivo tem valor proprio: e ele que REPOE o clima
        # quando os 5 ou 8 turnos acabarem.
        #
        # Sequencia: acabar a utilidade que so ele tem (hazards), depois sair.
        #
        # EXCECAO — vantagem plena. Com matchup dominante e HP alto, o setter esta a
        # ganhar o confronto e sair seria deitar fora a posicao. Ai fica.
        #
        # ORDEM DELIBERADA: esta regra corre ANTES da Regra 7 porque PROMOVE trocas,
        # e foi promover trocas que gerou o carrossel de tanques diagnosticado em
        # 6.28. Se a posicao se repetir, a Regra 7 despromove-as a seguir e tem a
        # ultima palavra. A guarda continua a valer.
        hab_ativo = str(getattr(active, "ability", "")).lower()
        clima_do_ativo = self.HABILIDADES_POR_CLIMA.get(hab_ativo)

        if clima_do_ativo and self._clima_atual(battle) == clima_do_ativo:
            # Ha quem aproveite este clima, vivo e NO BANCO?
            banco = [m for m in battle.team.values()
                     if not getattr(m, "fainted", False) and m is not active]
            abusadores_no_banco = any(
                clima_do_ativo in self._climas_uteis([m]) for m in banco)

            vantagem_plena = (
                matchup_atual in {MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV}
                and self.parser.get_hp_bucket(active) in ("FULL", "SAFE")
            )

            if abusadores_no_banco and not vantagem_plena:
                # 1. Utilidade que so o setter costuma ter. Se os hazards ja estiverem
                #    postos, o masking trata disso e o executor devolve None, com o
                #    ranking a avancar para a saida.
                # 2. Sair, de preferencia com pivo (mantem o momentum e escolhe quem
                #    entra), e so depois por troca simples.
                self._promover(ranking_list, ["HAZARD", "ATTACK_PIVOT",
                                              "SWITCH_OFFENSIVE"])

        # ==================================================================
        # REGRA GLOBAL 7: IMPASSE — a posicao repetiu-se (28/08/2026)
        # ==================================================================
        # DIAGNOSTICO (10.000 batalhas, InstinctBot vs InstinctBot, secao 6.x):
        #   45 empates, e nos ultimos turnos deles 360 de 360 decisoes foram TROCA
        #   papeis em campo: TANK v TANK em 356 de 360
        #   HP do ativo ao decidir sair: mediana 1,00 — 79% saia com HP >= 85%
        #   matchup: NEUTRAL em 196
        #
        # Nao e fuga nem impasse defensivo: e um TANQUE COM VIDA CHEIA a sair de um
        # confronto equilibrado. A pontuacao de troca ve sempre um tanque intacto no
        # banco a pontuar melhor que o de campo, e o outro lado faz o mesmo porque a
        # politica e identica. Carrossel de 5 ou 6 membros, que a quarentena por PAR
        # (execution.py) nao apanha porque cada par e novo.
        #
        # O QUE NAO SE FAZ: promover ataques. O Blue ja esta em 77,0% de acao de
        # ataque contra 55,4% do Green, e num impasse TANK v TANK bater mais forte
        # nao resolve — ja se sabe que o ataque nao chega. Promovem-se as ferramentas
        # que DESBLOQUEIAM, pela ordem em que produzem efeito duradouro:
        #
        #   STATUS       sobrevive a troca: cola-se aquele Pokemon e muda o relogio
        #   HAZARD       permanente e acumula: resolve as proximas entradas
        #   PHAZE        so com hazards no campo (cobra dano por entrada) OU com o
        #                adversario buffado (a expulsao apaga os boosts). Sem nenhuma
        #                das duas, expulsar da-lhe uma rotacao de graca
        #   ATTACK_TECH  Knock Off, Foul Play: mexem na posicao sem furar a defesa
        se_ativa = (
            not nos_prendemos and not estamos_presos
            and self.parser.get_hp_bucket(active) in ("FULL", "SAFE")
            and matchup_atual not in {MatchupState.CRITICAL_DIS,
                                      MatchupState.DEFENSIVE_DIS,
                                      MatchupState.VOLATILE}
        )
        assinatura = self._assinatura_posicao(battle, active, opp, matchup_atual)
        repetida = self._registar_posicao(battle, assinatura)

        if se_ativa and repetida:
            hazards_no_adversario = bool(nomes_de_enum(
                getattr(battle, "opponent_side_conditions", {}) or {}))
            desbloqueio = ["STATUS", "HAZARD"]
            if hazards_no_adversario or opp_buffado:
                desbloqueio.append("PHAZE")
            desbloqueio.append("ATTACK_TECH")
            self._promover(ranking_list, desbloqueio)
            # DESPROMOVER, nunca remover: se tudo o resto falhar o executor ainda
            # pode trocar, em vez de cair no fallback aleatorio.
            self._despromover(ranking_list, ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
                                             "ATTACK_PIVOT"])

        # ------------------------------------------------------------------
        # REGRA GLOBAL 11: A PREVISAO QUE FALHOU NAO SE REPETE (04/09/2026)
        # ------------------------------------------------------------------
        # `ATTACK_PREDICTIVE` aposta que o adversario vai TROCAR: escolhe o golpe
        # que melhor cobre o BANCO dele, e por isso aceita de proposito um golpe
        # fraco (ate imune) contra quem esta em campo. Isso e correcto UMA vez.
        #
        # Observado em batalha manual: `ATTACK_PREDICTIVE` em `pos=0` tres turnos
        # seguidos contra um adversario que nao trocou nenhuma vez. Nos turnos 31
        # a 33 o Excadrill usou `ironhead` (0,5x) contra um Toxapex que continuava
        # la, tendo `earthquake` (2x) no moveset. Quatro vezes pior, tres vezes
        # seguidas, e nada no sistema penalizava a aposta falhada.
        #
        # O CRITERIO E FACTUAL, nao heuristico: se o adversario e o MESMO do turno
        # em que se previu, a previsao FALHOU. `predicoes_falhadas` conta isso no
        # historico e reinicia sozinho assim que o alvo muda, porque ai a previsao
        # acertou e prever volta a valer.
        #
        # DESPROMOVE, NAO REMOVE. Com o adversario preso (trapping) ou sem banco,
        # prever pode continuar a ser a melhor jogada disponivel; o que nao pode e
        # ser a PRIMEIRA escolha depois de ja ter falhado.
        try:
            if int((history or {}).get('predicoes_falhadas', 0)) >= 1:
                self._despromover(ranking_list, ["ATTACK_PREDICTIVE"])
        except Exception:
            pass

        # ------------------------------------------------------------------
        # REGRA GLOBAL 5: ENDGAME — ULTIMO POKEMON EM CAMPO
        # ------------------------------------------------------------------
        # Sem banco, qualquer intencao que pressuponha sair de campo e um turno
        # perdido: o executor cai no fallback e o agente joga ao acaso. Empurra-as
        # para o fim, da menos pior para a pior.
        if not getattr(battle, "available_switches", None):
            for intent in ("ATTACK_PIVOT", "CLEAN_HAZARD", "PHAZE",
                           "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"):
                if intent in ranking_list:
                    ranking_list.remove(intent)
                    ranking_list.append(intent)

        # REGRA GLOBAL 3 (SEM BANCO, NAO EXISTE TROCA) — REMOVIDA EM 24/08/2026.
        # Era subconjunto da REGRA GLOBAL 5 acima e CONTRADIZIA-A: a 5 poe
        # ATTACK_PIVOT como "o menos pior" da cauda, e esta voltava a empurra-lo para
        # ultimo lugar absoluto. Fica so a 5, que e a versao completa.

        # ------------------------------------------------------------------
        # DETECAO DE LETALIDADE
        # ------------------------------------------------------------------
        # Criterio PRIMARIO: dano REALMENTE observado no turno anterior.
        # A estimativa (estimate_damage_percent) desconhece EVs, IVs, item e nature do
        # adversario, logo tem erro grande. Se no turno anterior os MESMOS dois Pokemon
        # estavam em campo e usamos um ataque de dano, a diferenca de HP do oponente e
        # o dano REAL daquele confronto — facto medido, nao estimativa.
        #
        # So se aplica quando o confronto e identico (mesmo aliado, mesmo oponente);
        # se qualquer um trocou, a medicao nao transita e voltamos a estimativa.
        dano_observado = None
        if history:
            mesmo_ativo = (history.get('last_active_id') is not None and
                           history.get('last_active_id') == getattr(active, 'species', None))
            mesmo_opp = (history.get('last_opponent_id') is not None and
                         history.get('last_opponent_id') == getattr(opp, 'species', None))
            if mesmo_ativo and mesmo_opp and history.get('last_action_was_damage'):
                hp_antes = history.get('last_opp_hp')
                if hp_antes is not None:
                    delta = hp_antes - opp_hp_frac
                    if delta > 0.0:
                        dano_observado = delta

        has_lethal = False
        if "ATTACK_STRONG" in candidate_mask or "ATTACK_PREDICTIVE" in candidate_mask:
            if dano_observado is not None and dano_observado >= opp_hp_frac:
                # O dano que ja causamos neste confronto chega para o rematar.
                has_lethal = True
            else:
                for m in battle.available_moves:
                    if m.base_power > 0 and not self.masker.is_move_useless(m, opp, battle, history):
                        dmg = self.physics.estimate_damage_percent(m, active, opp, battle)
                        # Calibra a estimativa pelo que foi REALMENTE observado neste
                        # confronto: se estimamos 30% e causamos 45%, a estimativa esta
                        # subavaliada por um fator ~1.5 e corrigimo-la.
                        if dano_observado is not None:
                            est_anterior = self.physics.estimate_damage_percent(m, active, opp, battle)
                            if est_anterior > 0.01:
                                fator = dano_observado / est_anterior
                                fator = max(0.5, min(2.0, fator))   # trava contra outliers
                                dmg = dmg * fator
                        if dmg >= opp_hp_frac:
                            has_lethal = True
                            break
        if has_lethal:
            for atk in reversed(["ATTACK_PREDICTIVE", "ATTACK_STRONG"]):
                if atk in ranking_list:
                    ranking_list.remove(atk)
                    ranking_list.insert(0, atk)

        # Anti-fadiga de troca: se acabou de trocar, empurra trocas para baixo
        if history:
            # CORREÇÃO: last_action pode estar PRESENTE com valor None (primeiro turno
            # da batalha). O default de .get() só cobre a chave ausente, não o valor
            # None, por isso normalizamos explicitamente antes de indexar.
            last = history.get('last_action') or (None, None)
            prev_action = last[0]
            if prev_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"]:
                for sw in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"]:
                    if sw in ranking_list:
                        current_idx = ranking_list.index(sw)
                        ranking_list.remove(sw)
                        new_idx = min(len(ranking_list), current_idx + 2)
                        ranking_list.insert(new_idx, sw)

        # ==================================================================
        # REDE DE SEGURANCA (reforcada em 04/09/2026)
        # ==================================================================
        # A versao anterior so cobria a lista VAZIA. Nao cobria o caso observado
        # no turno 24 de uma batalha manual: `ranking=['ATTACK_PREDICTIVE']`, uma
        # unica intencao, que o executor nao conseguiu traduzir — devolveu None e
        # o InstinctBot caiu no ALEATORIO. Jogada sorteada e o pior resultado
        # possivel para uma medicao, porque nao vem da politica.
        #
        # "A mascara nunca deveria podar todas as opcoes." Nao se garante isso
        # dentro de cada filtro, que ve UM golpe de cada vez e nao sabe se e o
        # ultimo a sobrar. Garante-se nos DOIS sitios que veem o conjunto todo:
        # aqui, e no ultimo recurso do `instinct_player.choose_move`.
        #
        # A CAUDA UNIVERSAL. `ATTACK_STRONG` e `SWITCH_DEFENSIVE` sao as duas
        # intencoes que quase sempre traduzem: a primeira porque o executor tem
        # fallback de obediencia, a segunda porque basta haver banco. Ficam no
        # FIM, logo nao alteram nenhuma decisao em que o ranking ja funcionava —
        # so existem para o caso em que tudo o resto falhou.
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

        for reserva in ("ATTACK_STRONG", "SWITCH_DEFENSIVE"):
            if reserva not in ranking_list:
                ranking_list.append(reserva)

        primary = ranking_list[0]
        return (primary, confidence, ranking_list, candidate_mask, has_lethal)
