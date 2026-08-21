"""
qlearning/pure_agent.py — Agente GREEN (Q-Learning puro).

Grupo de CONTROLO da pesquisa. Usa exatamente o mesmo estado, cérebro, reward,
execução e espaço de 37 ações que o Blue (tudo herdado da base TabularAgent), MAS
NÃO usa o instinto para decidir:
  - sem RANKING (ranking_list vazio -> nenhum prior de exploração),
  - sem ACTION MASK (todas as ações abstratas legais entram, sem poda tática).

Isola a contribuição do instinto: qualquer diferença Blue-vs-Green vem do instinto,
não de código diferente. Codinome: "Green". Cérebro salvo em green_brain.pkl.

Nota de paridade: o Green ainda usa o StateParser (para o estado), o classify_move da
física (para NOMEAR as ações legais) e o executor (para traduzir intenção->golpe).
Isto NÃO é "usar o instinto para decidir" — é usar a mesma representação de dados que
o Blue, para que a comparação seja justa. A decisão em si é 100% do Q-Learning.
"""

from qlearning.base_agent import TabularAgent


class PureAgent(TabularAgent):
    codename = "Green"

    def __init__(self, *args, brain_file="green_brain.pkl", **kwargs):
        super().__init__(*args, brain_file=brain_file, **kwargs)

    def _get_actions_and_ranking(self, battle, hist):
        # Sem instinto: todas as categorias abstratas LEGAIS neste turno, sem poda.
        categories = set()
        for move in battle.available_moves:
            cat = self.instinct.physics.classify_move(move)
            if cat.name in self.brain.actions:
                categories.add(cat.name)
        if battle.available_switches:
            categories.add("SWITCH_DEFENSIVE")
            categories.add("SWITCH_OFFENSIVE")

        valid_actions = self._expand_with_mechanic(list(categories), battle)
        if not valid_actions:
            valid_actions = ["ATTACK_STRONG"]

        # ranking_list vazio: o cérebro NÃO recebe prior do instinto.
        return valid_actions, []
