"""
qlearning/hybrid_agent.py — Agente BLUE (Híbrido).

O instinto contribui de DUAS formas independentes:

  1. ACTION MASKING — poda as ações taticamente inválidas antes de o cérebro decidir.
     Reduz o espaço de exploração inútil. Controlado por USAR_MASKING.

  2. RANKING (prior) — ordena as ações restantes e enviesa a exploração para as
     preferências do instinto. Controlado por USAR_RANKING.

Separá-las permite a ABLAÇÃO: medir quanto do ganho vem de cada uma. A comparação
Blue-vs-Green mede apenas o efeito CONJUNTO, porque o Green não tem nenhuma das duas.

Motivo para desligar o ranking: ele enviesa a exploração para perto da política do
mestre, o que constrange a divergência do aluno (o "teto mestre-aluno"). Evidência:
a distribuição de ações preferidas é muito mais estreita no Blue (81,3% ataque) do que
no Green (61,1% ataque, 22,0% suporte), o que é consistente com o ranking a estreitar
o comportamento explorado.

Codinome: "Blue". Cérebro salvo em blue_brain.pkl.
"""

from qlearning.base_agent import TabularAgent


class HybridAgent(TabularAgent):
    codename = "Blue"

    # ---- Ablação do instinto ----
    # USAR_MASKING=True,  USAR_RANKING=True   -> Blue completo (híbrido)   <== ATUAL
    # USAR_MASKING=True,  USAR_RANKING=False  -> Blue sem ranking (só poda)
    # USAR_MASKING=False, USAR_RANKING=False  -> equivale ao Green
    USAR_MASKING = True
    USAR_RANKING = True

    def __init__(self, *args, brain_file="blue_brain.pkl", **kwargs):
        super().__init__(*args, brain_file=brain_file, **kwargs)

    def _get_actions_and_ranking(self, battle, hist):
        if self.USAR_MASKING:
            # O instinto poda as ações inválidas (e, se USAR_RANKING, também ordena).
            _p, _c, ranking_list, candidate_mask, has_lethal = \
                self.instinct.policy.get_instinct_profile(battle, hist)
            valid_actions = self._expand_with_mechanic(candidate_mask, battle)
            valid_actions = self._podar_mec_por_overkill(valid_actions, battle, has_lethal)
        else:
            # Sem poda: todas as categorias legais, como o Green.
            categorias = set()
            for move in battle.available_moves:
                # Contexto, pela mesma razao (29/08/2026). Este ramo so corre com
                # USAR_MASKING=False, mas fica coerente com os restantes.
                cat = self.instinct.physics.classify_move(
                    move, battle.opponent_active_pokemon, battle)
                if cat.name in self.brain.actions:
                    categorias.add(cat.name)
            if battle.available_switches:
                categorias.add("SWITCH_DEFENSIVE")
                categorias.add("SWITCH_OFFENSIVE")
            valid_actions = self._expand_with_mechanic(list(categorias), battle)
            ranking_list = []

        if not self.USAR_RANKING:
            # Ranking vazio: o cérebro decide sem prior do instinto. A exploração
            # deixa de ser canalizada para as preferências do mestre.
            ranking_list = []

        if not valid_actions:
            valid_actions = ["ATTACK_STRONG"]
        return valid_actions, ranking_list

    # ------------------------------------------------------------------
    # Guarda de overkill de recurso consumível (24/08/2026)
    # ------------------------------------------------------------------

    def _podar_mec_por_overkill(self, valid_actions, battle, has_lethal):
        """Retira as variantes _MEC quando gastar o recurso seria desperdício.

        ONDE ESTA GUARDA VIVE, E PORQUÊ. Vive na MÁSCARA, não na execução. Se o
        executor recusasse uma mecânica que o cérebro escolheu, a Q-table receberia
        a recompensa de `X_MEC` num turno em que a mecânica não saiu: aprenderia uma
        associação falsa e o inspect_brain passaria a mentir. A máscara define o que
        é viável, o cérebro escolhe, o executor obedece.

        SÓ PARA RECURSO CONSUMÍVEL. A distinção importa:

          Z-MOVE — um uso por batalha. Gastá-lo num turno em que o golpe normal já
                   mata é desperdício puro. PODAR.
          MEGA   — permanente, dura o resto da batalha. Mega-evoluir e matar no
                   mesmo turno sai DE GRAÇA: ganha-se a forma para o resto do jogo
                   sem custo. Podar aqui seria o erro.

        Como o _order_with_mechanic resolve `_MEC` por prioridade (tera, mega, z),
        `_MEC` significa MEGA sempre que houver pedra disponível. Por isso a poda
        exige `can_z_move AND NOT can_mega_evolve`.

        Esta guarda é do BLUE e não do Green por desenho: depende do `has_lethal` do
        instinto, e a máscara É a diferença entre os dois. Dá-la ao Green destruiria
        o grupo de controlo. O Green pode gastar um Z num alvo condenado, e vai
        gastar — é isso que se está a medir.
        """
        if not has_lethal:
            return valid_actions

        pode_z = bool(getattr(battle, "can_z_move", False))
        pode_mega = bool(getattr(battle, "can_mega_evolve", False))
        if not (pode_z and not pode_mega):
            return valid_actions

        podadas = [a for a in valid_actions if not a.endswith("_MEC")]
        # Nunca devolver lista vazia: se por algum motivo só houvesse _MEC, mantém-se
        # o conjunto original em vez de forçar o fallback de emergência.
        return podadas if podadas else valid_actions
