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
    # USAR_MASKING=True,  USAR_RANKING=True   -> Blue completo (híbrido original)
    # USAR_MASKING=True,  USAR_RANKING=False  -> Blue sem ranking (só poda)   <== ATUAL
    # USAR_MASKING=False, USAR_RANKING=False  -> equivale ao Green
    USAR_MASKING = True
    USAR_RANKING = True

    def __init__(self, *args, brain_file="blue_brain.pkl", **kwargs):
        super().__init__(*args, brain_file=brain_file, **kwargs)

    def _get_actions_and_ranking(self, battle, hist):
        if self.USAR_MASKING:
            # O instinto poda as ações inválidas (e, se USAR_RANKING, também ordena).
            _p, _c, ranking_list, candidate_mask, _lethal = \
                self.instinct.policy.get_instinct_profile(battle, hist)
            valid_actions = self._expand_with_mechanic(candidate_mask, battle)
        else:
            # Sem poda: todas as categorias legais, como o Green.
            categorias = set()
            for move in battle.available_moves:
                cat = self.instinct.physics.classify_move(move)
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
