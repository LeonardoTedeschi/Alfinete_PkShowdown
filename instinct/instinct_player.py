"""
InstinctBot — Agente 2 do projeto ALFINETE.

Joga usando APENAS o conhecimento de domínio (o instinto), sem qualquer aprendizado:
sem Q-table, sem epsilon, sem treino. A mesma situação produz sempre a mesma decisão
(determinístico, a menos do RNG do próprio jogo).

Papel na pesquisa: é a RÉGUA de avaliação. Todos os outros agentes (Green/Q-puro,
Blue/Híbrido, Red/DQN) treinam e são medidos contra ele. Por ser determinístico,
qualquer diferença de Win Rate entre agentes vem DELES, não de flutuação do oponente
— é o que torna a comparação justa e reproduzível.

Este ficheiro é uma CASCA FINA: toda a lógica tática vive nos componentes do pacote
`instinct`. O bot apenas liga o fluxo (troca forçada -> decidir -> executar).
"""

from poke_env.player import Player

# Ao instalar no projeto: `from instinct import build_instinct`
from instinct import build_instinct

from shared import diagnostico as _diag
from shared.diagnostico import ligado as diagnostico_ligado
from shared.mechanics import e_golpe, marcar_uso, mega_valido, z_move_valido


class InstinctBot(Player):
    """Agente que decide exclusivamente pelo instinto (conhecimento de domínio)."""

    def __init__(self, *args, diagnostico=None, **kwargs):  # noqa: A002
        super().__init__(*args, **kwargs)
        # Monta os 6 componentes já ligados (physics, parser, masker, policy, executor).
        self.instinct = build_instinct()
        # HISTORICO POR BATALHA (29/08/2026). Ver `_get_history` abaixo.
        self._history = {}

        # ==============================================================
        # MODO DIAGNOSTICO, DESLIGADO POR OMISSAO (03/09/2026)
        # ==============================================================
        # PORQUE. Nos treinos o InstinctBot corre como ADVERSARIO no MESMO
        # processo do Blue e do Green, e escreve no MESMO stdout. Os avisos
        # `[INSTINTO][IMUNE]` apareciam no meio das metricas de treino e davam
        # a impressao de vir do agente, quando vinham da regua. Poluiam a
        # leitura e ja custaram um diagnostico errado.
        #
        # A INFORMACAO NAO SE PERDE: `_CONTAGEM_IMUNE` conta sempre, ligado ou
        # desligado. O que o modo controla e so se IMPRIME. Ver
        # `resumo_imunes()`, que devolve o acumulado para ser inspeccionado no
        # fim de uma corrida sem ter tido uma unica linha de ruido durante.
        #
        # COMO LIGAR, para batalhas manuais e medicao de ancora:
        #   InstinctBot(..., diagnostico=True)
        #   ou a variavel de ambiente ALFINETE_DIAGNOSTICO=1
        #
        # A variavel existe para os scripts de treino NAO precisarem de mudar:
        # o default e desligado, logo `train_blue.py`, `train_green.py` e
        # `treino_continuo.py` ficam limpos sem uma linha alterada.
        # DELEGA NO `shared.diagnostico` (03/09/2026). A leitura da variavel de
        # ambiente estava aqui em copia propria, e o `masking` e o `execution`
        # precisavam do MESMO interruptor: tres leituras da mesma variavel sao
        # tres sitios para divergirem. O parametro continua a poder forcar.
        self.diagnostico = diagnostico if diagnostico is not None else diagnostico_ligado()

    # ==================================================================
    # HISTORICO POR BATALHA (29/08/2026)
    # ==================================================================
    # ACRESCENTADO PARA CORRIGIR CINCO REGRAS INERTES NA REGUA.
    #
    # O `choose_move` chamava `get_instinct_profile(battle)` e
    # `get_best_execution_object(intent, battle)` SEM `history`. Como as duas
    # assinaturas tem `history=None` por omissao, nao havia erro: as regras que
    # dependem dele simplesmente NUNCA CORRIAM para o InstinctBot.
    #
    #   masking.py   l.410   Protect consecutivo (chance 1/3^n: o 2o uso tem ~33%)
    #   policy.py    l.1392  LETALIDADE POR DANO OBSERVADO — a mais grave: caia
    #                        sempre na estimativa, que desconhece EVs, IVs, item e
    #                        nature do adversario, tendo a medicao real disponivel
    #   policy.py    l.1155  limiar de vida pelo dano REAL sofrido no turno anterior
    #   policy.py    l.1200  guarda da corrida de buffs consecutivos
    #   policy.py    l.1436  anti-fadiga de troca
    #   execution.py l.412   turno de inicio do clima
    #
    # O Blue e o Green TINHAM estas regras (herdam o `_history` do `TabularAgent`);
    # o adversario contra o qual sao medidos, NAO. Duas consequencias:
    #
    #   1. a REGUA media um instinto mais fraco do que o codigo descrevia, e a tese
    #      descreveria um jogador que nunca jogou
    #   2. a anti-fadiga de troca estava desligada dos DOIS lados na corrida
    #      InstinctBot vs InstinctBot que diagnosticou o carrossel de tanques (6.28)
    #
    # As chaves e a semantica sao IDENTICAS as do `TabularAgent._get_history`, de
    # proposito: divergir criaria um TERCEIRO comportamento em vez de dois.
    #
    # NOTA: isto SOBE A ANCORA e nao beneficia os agentes — ambos enfrentam o mesmo
    # adversario, logo a comparacao Blue vs Green nao muda. E correcao de INTEGRIDADE
    # (o instinto passa a ser o jogador que o codigo descreve), nao de desempenho.

    def _get_history(self, battle):
        tag = getattr(battle, "battle_tag", None)
        if tag not in self._history:
            self._history[tag] = {
                'state': None, 'last_action': None, 'prev_action': None,
                'last_was_exploratory': False, 'buffs_consecutivos': 0,
                'last_active_id': None, 'last_opponent_id': None,
                'last_opp_hp': None, 'last_my_hp': None,
                'last_action_was_damage': False,
                'my_fainted': 0, 'opp_fainted': 0,
                'weather_start_turn': 0, 'weather_active_prev': False,
            }
        return self._history[tag]

    def _atualizar_history(self, hist, battle, base_action, obj=None):
        """Mesma ordem e mesmas chaves que o TabularAgent escreve por turno."""
        # `prev_action` tem de ser preservada ANTES de `last_action` ser sobrescrita:
        # a regra anti-Protect do masking le as duas.
        hist['prev_action'] = hist.get('last_action')
        hist['last_action'] = (base_action, None)

        if str(base_action).replace("_MEC", "") == "BUFF":
            hist['buffs_consecutivos'] = int(hist.get('buffs_consecutivos', 0)) + 1
        else:
            hist['buffs_consecutivos'] = 0

        try:
            hist['last_active_id'] = getattr(battle.active_pokemon, 'species', None)
            hist['last_opponent_id'] = getattr(battle.opponent_active_pokemon, 'species', None)
            hist['last_opp_hp'] = battle.opponent_active_pokemon.current_hp_fraction
            hist['last_my_hp'] = battle.active_pokemon.current_hp_fraction
            hist['last_action_was_damage'] = str(base_action).startswith("ATTACK")
        except Exception:
            hist['last_action_was_damage'] = False

        # ==============================================================
        # WISH E PREDITIVO: ESTADO QUE O POKE-ENV NAO DA (04/09/2026)
        # ==============================================================
        # WISH. Verificado em 04/09: a string "WISH" nao existe no `pokemon.py`
        # do poke-env, e o Wish tambem nao consta das `SideCondition`. E uma
        # SLOT condition e a biblioteca simplesmente NAO A EXPOE. O filtro do
        # masking procurava-a em `side_conditions` e em `effects`, e nunca a
        # podia encontrar: o instinto usou Wish nos turnos 40, 42, 43, 44 e 45 da
        # mesma batalha, a falhar. Se a biblioteca nao guarda, guardamos nos.
        #
        # PREDITIVO. Guarda-se tambem a especie adversaria no turno em que se
        # previu, para saber depois se a previsao ACERTOU (ele trocou) ou FALHOU
        # (ficou). Ver a penalizacao em `policy`.
        try:
            _acao = str(base_action).replace("_MEC", "")
            _opp = getattr(battle.opponent_active_pokemon, "species", None)

            if _acao in ("STATUS", "HEAL") and any(
                    getattr(m, "id", "") == "wish"
                    for m in (getattr(battle, "available_moves", None) or [])):
                # Nao se sabe QUE golpe o executor escolheu, so a intencao. Marca-se
                # a intencao de suporte com Wish disponivel, e o masking confirma
                # com o PP do proprio Wish. Sobre-marcar custa dois turnos sem Wish;
                # sub-marcar custa o ciclo inteiro que se observou.
                hist['wish_turno'] = battle.turn

            if _acao == "ATTACK_PREDICTIVE":
                hist['predicao_turno'] = battle.turn
                hist['predicao_alvo'] = _opp
                # Conta previsoes seguidas CONTRA O MESMO adversario. Reinicia
                # quando o alvo muda, porque ai a previsao acertou.
                if hist.get('predicao_alvo_anterior') == _opp:
                    hist['predicoes_falhadas'] = int(hist.get('predicoes_falhadas', 0)) + 1
                else:
                    hist['predicoes_falhadas'] = 0
                hist['predicao_alvo_anterior'] = _opp
            elif hist.get('predicao_alvo_anterior') != _opp:
                hist['predicoes_falhadas'] = 0
                hist['predicao_alvo_anterior'] = None
        except Exception:
            pass


        # GOLPES DE EFEITO DIFERIDO (04/09/2026). Registados pelo GOLPE
        # EXECUTADO e nao pela intencao: a intencao diz o que se queria, o
        # `obj` diz o que saiu. O registo do Wish por intencao (acima) fica,
        # como rede; este e o preciso.
        #
        # PORQUE TEM DE SER NOSSO: Wish, Future Sight e Doom Desire sao SLOT
        # conditions. O poke-env nao as expoe nem em `side_conditions` nem em
        # `effects` (verificado em 04/09: a string "WISH" nao existe no
        # `pokemon.py`). Sem memoria propria, nenhum filtro os pode ver.
        #
        # Observado: `futuresight` em quatro turnos seguidos, com
        # "But it failed!" nos dois ultimos.
        try:
            _gid = getattr(obj, "id", None)
            if _gid == "wish":
                hist['wish_turno'] = battle.turn
            elif _gid in ("futuresight", "doomdesire"):
                hist['diferido_turno'] = battle.turn
        except Exception:
            pass

        hist['my_fainted'] = len([m for m in battle.team.values() if m.fainted])
        hist['opp_fainted'] = len([m for m in battle.opponent_team.values() if m.fainted])

        if battle.weather:
            if not hist.get('weather_active_prev'):
                hist['weather_start_turn'] = battle.turn
            hist['weather_active_prev'] = True
        else:
            hist['weather_active_prev'] = False

    def _battle_finished_callback(self, battle):
        """Liberta o estado por batalha do instinto quando ela termina.

        O InstinctBot nao tem `_history` nem cerebro, mas a policy e o executor
        guardam dicionarios indexados por `battle_tag`:

          executor._saidas    quarentena anti-ciclo por par (quem saiu, contra quem)
          policy._posicoes    historico de posicoes da Regra 7 (impasse)

        Sem limpeza crescem uma entrada por batalha e nunca encolhem. Numa corrida de
        10.000 batalhas do medir_regua isso e memoria desperdicada; num treino de
        400.000, e fuga a serio.
        """
        tag = getattr(battle, "battle_tag", None)
        self._history.pop(tag, None)
        for componente, metodo in ((self.instinct.executor, "limpar_saidas"),
                                   (self.instinct.policy, "limpar_posicoes")):
            try:
                getattr(componente, metodo)(tag)
            except AttributeError:
                pass
        try:
            super()._battle_finished_callback(battle)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # BLINDAGEM CONTRA GOLPE IMUNE (30/08/2026)
    # ------------------------------------------------------------------
    # PORQUE EXISTE. O filtro 0 do masking (multiplicador ZERO -> podar) e a regra
    # mais simples do sistema, e mesmo assim golpes imunes chegaram a campo DUAS
    # vezes em batalha manual: Plasma Fists contra Steelix (Aco/Terra) e Earth Power
    # contra Drifblim (Fantasma/Voador). Nao foi possivel determinar por LEITURA por
    # que caminho sairam — os tres caminhos de golpe do executor passam pela mascara.
    #
    # Esta guarda nao substitui o filtro 0: e a ultima linha, no unico ponto por onde
    # TODAS as decisoes passam. Cobre inclusive o `choose_random_move`, que sorteia
    # entre golpes legais sem olhar para efetividade e por isso pode sortear um imune.
    #
    # E DIAGNOSTICA E CORRIGE AO MESMO TEMPO. O aviso diz qual INTENCAO produziu o
    # golpe, que e exatamente o dado que falta para achar o caminho furado. Quando os
    # avisos pararem de aparecer, a guarda passa a ser redundante e pode sair; ate la,
    # cada aviso e uma pista.

    # Combinacoes ja avisadas nesta sessao (nao inundar um log de 400k) e
    # contagem acumulada por combinacao, que e escrita SEMPRE, mesmo com o
    # modo diagnostico desligado. Ambas de classe: partilhadas por todas as
    # instancias do processo, que e o que se quer num orquestrador de sessoes.
    _AVISOS_IMUNE = set()
    _CONTAGEM_IMUNE = {}

    @classmethod
    def resumo_imunes(cls):
        """Acumulado de golpes imunes descartados, por (intencao, golpe, alvo).

        Serve para inspeccionar no fim de uma corrida silenciosa. Enquanto
        devolver dicionario vazio, nenhum caminho do executor produziu golpe de
        multiplicador zero e a guarda esta redundante (ver 6.44, seccao E).
        """
        return dict(cls._CONTAGEM_IMUNE)

    def _golpe_imune(self, obj, battle):
        """O objeto e um golpe de dano com multiplicador ZERO contra o alvo?"""
        try:
            if not e_golpe(obj) or not getattr(obj, "base_power", 0):
                return False
            opp = battle.opponent_active_pokemon
            return bool(opp) and opp.damage_multiplier(obj) == 0
        except Exception:
            # Na duvida NAO se descarta: melhor jogar um golpe possivelmente mau do
            # que perder o turno por causa de uma excecao na verificacao.
            return False

    def _avisar_imune(self, obj, battle, intent):
        """Contabiliza sempre; imprime so em modo diagnostico.

        A contagem e por (intencao, golpe, alvo) e nao tem tecto: e o dado que
        identifica QUAL intencao produziu o golpe imune, que foi o que fechou o
        diagnostico do atalho de pivo. A impressao e que e um aviso por
        combinacao, para nao inundar um log de 400k batalhas.

        O nome do jogador vai na linha: o InstinctBot corre no mesmo processo e
        no mesmo stdout que o agente em treino, e sem isto era impossivel
        distinguir a regua do agente so pelo log.
        """
        try:
            opp = battle.opponent_active_pokemon
            chave = (str(intent), getattr(obj, "id", "?"), getattr(opp, "species", "?"))
            self._CONTAGEM_IMUNE[chave] = self._CONTAGEM_IMUNE.get(chave, 0) + 1
            if not (self.diagnostico or _diag.ligado()):
                return
            if chave in self._AVISOS_IMUNE:
                return
            self._AVISOS_IMUNE.add(chave)
            tipos = "/".join(t.name for t in opp.types if t)
            print(f"[IMUNE][{getattr(self, 'username', 'InstinctBot')}] "
                  f"intencao={intent} golpe={getattr(obj,'id','?')} "
                  f"alvo={getattr(opp,'species','?')} ({tipos}) -> descartado, "
                  f"o ranking avanca")
        except Exception:
            pass

    def _aleatorio_seguro(self, battle):
        """`choose_random_move` sem golpes imunes.

        O aleatorio e o ultimo recurso e nao consulta a mascara. Filtra-se o que e
        garantidamente inutil; se nao sobrar nada, cai-se no aleatorio puro, porque
        nesse ponto qualquer escolha e igualmente ma.
        """
        try:
            import random
            opcoes = list(getattr(battle, "available_switches", None) or [])
            opcoes += [m for m in (getattr(battle, "available_moves", None) or [])
                       if not self._golpe_imune(m, battle)]
            if opcoes:
                return self.create_order(random.choice(opcoes))
        except Exception:
            pass
        return self.choose_random_move(battle)

    def teampreview(self, battle):
        """Escolhe a ordem de time inicial pela heurística de lead do instinto."""
        return self.instinct.executor.get_best_lead(battle)

    def choose_move(self, battle):
        try:
            # 1. Troca forçada (Pokémon ativo desmaiou ou foi forçado a sair).
            hist = self._get_history(battle)

            if battle.force_switch or (battle.active_pokemon and battle.active_pokemon.fainted):
                switch = self.instinct.executor.get_post_faint_switch(battle, hist)
                return self.create_order(switch) if switch else self.choose_random_move(battle)

            # 2. Sem ativo/oponente definido (turno de transição) -> aleatório seguro.
            if not battle.active_pokemon or not battle.opponent_active_pokemon:
                return self.choose_random_move(battle)

            # 3. DECISÃO: a policy devolve o ranking de intenções (5 valores).
            #    Usamos o ranking_list (índice 2), não a tupla inteira.
            _primary, _conf, ranking_list, _mask, _has_lethal = \
                self.instinct.policy.get_instinct_profile(battle, hist)

            # 4. EXECUÇÃO: percorre o ranking até o executor devolver um objeto válido.
            # `fallback_obediencia=False`: o InstinctBot nao tem Q-table, logo o
            # fallback que entrega um golpe inutil "para o cerebro ser punido"
            # nao tem a quem punir. Sem isto ele gastava o turno num golpe de
            # dano zero (Plasma Fists contra Terra) em vez de deixar o ranking
            # avancar para trocar ou por status. Ver execution.py, bloco de
            # ataque. O Blue e o Green NAO passam o parametro e mantem o
            # comportamento antigo, que para eles e o correto.
            for intent in ranking_list:
                obj = self.instinct.executor.get_best_execution_object(
                    intent, battle, hist,
                    fallback_obediencia=False, comparar_ataques=True)
                if obj:
                    # ULTIMA LINHA CONTRA GOLPE IMUNE (30/08/2026).
                    # Ver `_golpe_imune`. Se algum caminho do executor devolver
                    # um golpe de multiplicador ZERO, descarta-se e o ranking
                    # avanca, em vez de gastar o turno. O aviso identifica QUAL
                    # intencao o produziu, que e a informacao que faltava para
                    # diagnosticar (Earth Power contra Voador, Plasma Fists
                    # contra Terra) sem inspecionar o executor inteiro.
                    if self._golpe_imune(obj, battle):
                        self._avisar_imune(obj, battle, intent)
                        continue
                    # Registar SO a intencao que foi mesmo EXECUTADA. Registar a
                    # primeira do ranking daria `buffs_consecutivos` e
                    # `last_action_was_damage` errados sempre que o executor
                    # recusasse a intencao de topo e o ranking avancasse.
                    # SONDA [DECISAO] (03/09/2026) — TEMPORARIA.
                    # Diz QUE INTENCAO produziu o objecto que foi mesmo jogado, e
                    # quantas intencoes o ranking teve de descer para la chegar.
                    # E a sonda que responde quando o caminho furado NAO e nenhum
                    # dos dois instrumentados: se o Stealth Rock repetido sair sob
                    # `intencao=HAZARD`, a poda nao correu; se sair sob outra
                    # intencao ou pelo aleatorio, o problema esta noutro sitio.
                    # `opp` e `last` acrescentados em 04/09/2026. Sem eles, duas
                    # perguntas ficavam sem resposta possivel no log:
                    #   - a REGRA 11 nao travou a previsao repetida, ou o
                    #     adversario trocou e o contador reiniciou com razao?
                    #     So a especie adversaria por turno distingue os casos.
                    #   - o filtro anti-Protect esta certo por leitura e nao
                    #     disparou. A unica explicacao compativel e `last_action`
                    #     chegar sem PROTECT, e isso ve-se aqui.
                    _diag.log("DECISAO", f"turno={getattr(battle, 'turn', '?')} "
                                         f"intencao={intent} "
                                         f"obj={getattr(obj, 'id', None) or getattr(obj, 'species', obj)} "
                                         f"pos={ranking_list.index(intent)} "
                                         f"opp={getattr(battle.opponent_active_pokemon, 'species', None)} "
                                         f"last={hist.get('last_action')} "
                                         f"pred_falhadas={hist.get('predicoes_falhadas')} "
                                         f"ranking={list(ranking_list)[:5]}")
                    self._atualizar_history(hist, battle, intent, obj)
                    return self._ordem_com_mecanica(obj, battle, _has_lethal)

            # ==========================================================
            # ULTIMO RECURSO DETERMINISTICO, ANTES DO ALEATORIO (04/09/2026)
            # ==========================================================
            # Observado no turno 24 de uma batalha manual: a mascara podou tudo,
            # sobrou `ranking=['ATTACK_PREDICTIVE']`, o executor devolveu None e
            # jogou-se ao ACASO. Uma vez em quatro batalhas, mas e o caminho que
            # mais contamina uma medicao, porque a jogada nao vem da politica.
            #
            # "A mascara nunca deveria podar todas as opcoes": nao se pode
            # garantir isso dentro de cada filtro, que so ve UM golpe de cada vez
            # e nao sabe se e o ultimo. Garante-se AQUI e na rede de seguranca da
            # `policy`, que sao os dois sitios que veem o conjunto todo.
            #
            # ATTACK_STRONG com o fallback de obediencia LIGADO devolve o melhor
            # golpe de dano mesmo que a mascara reprove todos: uma jogada ma
            # escolhida por criterio vale mais que uma jogada sorteada, e nao
            # contamina a leitura do comportamento.
            try:
                obj = self.instinct.executor.get_best_execution_object(
                    "ATTACK_STRONG", battle, hist, fallback_obediencia=True)
                if obj is not None:
                    _diag.log("DECISAO", f"turno={getattr(battle, 'turn', '?')} "
                                         f"nenhuma intencao traduziu -> ULTIMO RECURSO "
                                         f"ATTACK_STRONG obj="
                                         f"{getattr(obj, 'id', None) or getattr(obj, 'species', obj)} "
                                         f"(ranking={list(ranking_list)[:5]})")
                    self._atualizar_history(hist, battle, "ATTACK_STRONG", obj)
                    return self._ordem_com_mecanica(obj, battle, _has_lethal)
            except Exception:
                pass

            _diag.log("DECISAO", f"turno={getattr(battle, 'turn', '?')} "
                                 f"NENHUMA intencao produziu objecto -> ALEATORIO "
                                 f"(ranking={list(ranking_list)[:5]})")
            return self._aleatorio_seguro(battle)

        except Exception:
            # Falha defensiva: nunca trava a batalha, joga algo legal.
            return self._aleatorio_seguro(battle)

    # ------------------------------------------------------------------
    # MECANICAS DE TURNO (adicionado 24/08/2026, ver 6.19)
    # ------------------------------------------------------------------

    def _ordem_com_mecanica(self, obj, battle, has_lethal):
        """Decide Mega e Z-Move. E conhecimento de dominio como o resto do instinto.

        ALTERA A REGUA. Antes desta adicao o InstinctBot nunca usava mecanica
        nenhuma, enquanto o MaxDamage mega-evoluia em 13 dos 15 times do pool. As
        ancoras de 56,26% e 55,82% foram medidas SEM isto e nao servem de comparacao.

        A forma e a de um JOGADOR MEDIO, nao a de um jogador bom:

          MEGA   — usar assim que disponivel. A Mega sobe stats quase sempre sem
                   contrapartida, e adia-la para preservar a habilidade base
                   (Intimidate, Regenerator) e leitura de jogador experiente.
                   Deliberadamente FORA de ambito.

          Z-MOVE — recurso de uso unico. Duas guardas contra desperdicio, ambas
                   erros que um jogador medio nao comete:
                     1. nao gastar quando o ataque normal ja e letal (overkill)
                     2. nao gastar num adversario quase morto (bucket CRIT)
                   Fora disso, usa-se no primeiro ataque compativel.

        Tera nao e tratada: esta banida no gen9nationaldex por Terastal Clause.
        """
        # Trocas nunca usam mecanica.
        if not e_golpe(obj):
            return self.create_order(obj)

        try:
            # --- MEGA: propriedade do Pokemon, nao do golpe ---
            if mega_valido(battle):
                # marcar ANTES de devolver: a partir de poke-env 0.12.0 a pedra
                # permanece no item depois da transformacao, logo o proprio item ja
                # nao serve de guarda contra uma segunda tentativa.
                marcar_uso(battle, "mega")
                return self.create_order(obj, mega=True)

            # --- Z-MOVE: propriedade do golpe, e recurso de uso unico ---
            if getattr(battle, "can_z_move", False):
                # GUARDA 0 (30/08/2026): SO EM GOLPE DE ATAQUE.
                #
                # O `e_golpe(obj)` la em cima aceita golpes de STATUS, e a
                # docstring desta funcao ja dizia "primeiro ATAQUE compativel".
                # A intencao estava escrita e a condicao nao existia.
                #
                # Observado em batalha manual: Garchomp @ Rockium Z (Time-13 do
                # pool) queimou o Z em STEALTH ROCK. O cristal estava la para
                # virar Stone Edge em Continental Crush; o instinto gastou o
                # recurso de uso unico do jogo inteiro num hazard.
                #
                # PORQUE SO O INSTINTO. Para o Blue e o Green, gastar o Z num
                # golpe de status e EXPLORACAO: eles nao sabem que e mau ate
                # experimentarem, e a Q-table existe para aprender isso. Um
                # Z-status da +1 de atributo, logo nao e sequer sempre errado.
                # Para o InstinctBot nao ha o que aprender: e so desperdicio.
                # Mesmo criterio do `fallback_obediencia` no executor.
                if not getattr(obj, "base_power", 0):
                    return self.create_order(obj)

                if has_lethal:
                    return self.create_order(obj)  # guarda 1: overkill

                opp = battle.opponent_active_pokemon
                if opp and self.instinct.parser.get_hp_bucket(opp) == "CRIT":
                    return self.create_order(obj)  # guarda 2: alvo quase morto

                # O servidor rejeita "X como Z-move" se o Pokemon nao carregar o
                # cristal do tipo certo. Ver z_move_valido acima.
                if z_move_valido(obj, battle):
                    # Idem para o cristal Z: desde 0.12.0 ("Don't get rid of z item
                    # when using z move") o item persiste apos o uso.
                    marcar_uso(battle, "z")
                    return self.create_order(obj, z_move=True)

        except Exception:
            # Uma ordem invalida custa o turno inteiro. Na duvida, joga-se normal.
            pass

        return self.create_order(obj)
