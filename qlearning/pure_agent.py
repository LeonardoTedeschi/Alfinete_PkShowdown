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

import random

from qlearning.base_agent import TabularAgent


class PureAgent(TabularAgent):
    codename = "Green"

    def __init__(self, *args, brain_file="green_brain.pkl", **kwargs):
        super().__init__(*args, brain_file=brain_file, **kwargs)

    def _get_actions_and_ranking(self, battle, hist):
        # Sem instinto: todas as categorias abstratas LEGAIS neste turno, sem poda.
        categories = set()
        for move in battle.available_moves:
            # CONTEXTO (29/08/2026), pela mesma razao do masking e do executor: sem
            # ele, a categoria com que a acao entra na Q-table pode nao ser a do golpe
            # que o executor acaba por jogar, e a recompensa ia parar a acao errada.
            #
            # Isto NAO da conhecimento tatico ao Green: e a mesma NOMEACAO de acoes
            # que ele ja fazia, agora correta. Mantem-se a nota de paridade acima —
            # usar a mesma representacao de dados que o Blue e o que torna a
            # comparacao justa; usa-la mal e que a tornava enviesada.
            cat = self.instinct.physics.classify_move(
                move, battle.opponent_active_pokemon, battle)
            if cat.name in self.brain.actions:
                categories.add(cat.name)
        if battle.available_switches:
            categories.add("SWITCH_DEFENSIVE")
            categories.add("SWITCH_OFFENSIVE")

        valid_actions = self._expand_with_mechanic(list(categories), battle)
        if not valid_actions:
            valid_actions = ["ATTACK_STRONG"]

        # Sem ranking do instinto, a ORDEM da lista nao pode funcionar como prior
        # acidental. O brain escolhe valid_ranked[0] em estado virgem e usa posicoes
        # da lista durante exploracao; uma lista vinda de set() teria ordem de hash.
        # Embaralhar a cada decisao torna o cold start e a exploracao marginais
        # uniformes entre as acoes legais, sem mexer no argmax Q da exploracao zero.
        random.shuffle(valid_actions)

        # ranking_list vazio: o cérebro NÃO recebe prior do instinto.
        return valid_actions, []
