"""
qlearning/ash_agent.py — Agente ASH (Q-Learning com o minimo indispensavel).

TERCEIRO BRACO da comparacao. Existe para decompor a contribuicao do conhecimento de
dominio em duas parcelas mensuraveis:

| Agente | Policy (ranking + poda abstrata) | Executor do instinto | Mede |
|---|---|---|---|
| **Blue**  | sim | sim | conhecimento completo |
| **Green** | nao | sim | so a camada TATICA |
| **Ash**   | nao | nao | Q-Learning quase puro |

Blue − Green  = contribuicao da camada ESTRATEGICA
Green − Ash   = contribuicao da camada TATICA
Blue − Ash    = contribuicao TOTAL do instinto

Sem o Ash, a afirmacao "com instinto vs sem instinto" seria imprecisa: o Green herda
~66% do conhecimento de dominio escrito a mao (executor + masking + physics), logo
nao aprende do zero.

O QUE O ASH PARTILHA (percepcao — sem isto nao ha comparacao direta):
  - `StateParser`: a mesma tupla de estado de 15 dimensoes
  - `physics.classify_move`: o mesmo espaco de 37 acoes abstratas
  - o mesmo cerebro (BlueBrain), os mesmos hiperparametros

O QUE O ASH NAO TEM (decisao):
  - ranking de intencoes e poda do espaco de acoes abstratas (como o Green)
  - E, ao contrario do Green, tambem NAO tem o executor do instinto:
      * escolhe o lead ao acaso (nao pontua matchups)
      * troca pos-faint ao acaso (nao pontua sobrevivencia nem ameaca)
      * escolhe ataques por poder base x precisao (nao estima dano real)
      * so respeita imunidades por tipo (regra do jogo, nao estrategia)
      * mantem a distincao defensivo/ofensivo nas trocas apenas por tabela de tipos,
        para que as duas acoes abstratas nao colapsem numa so

Codinome: "Ash". Cerebro salvo em ash_brain.pkl.
"""

from qlearning.base_agent import TabularAgent
from shared.minimal_execution import ExecutorMinimo


class AshAgent(TabularAgent):
    codename = "Ash"

    def __init__(self, *args, brain_file="ash_brain.pkl", **kwargs):
        super().__init__(*args, brain_file=brain_file, **kwargs)
        # Substitui o executor do instinto pelo minimo. O estado e o espaco de acoes
        # continuam identicos aos dos outros agentes.
        self.executor = ExecutorMinimo(self.instinct.physics)

    def _get_actions_and_ranking(self, battle, hist):
        """Todas as categorias legais, sem poda e sem ranking (como o Green).

        A diferenca face ao Green esta no EXECUTOR, nao aqui.
        """
        categorias = set()
        for move in battle.available_moves:
            try:
                cat = self.instinct.physics.classify_move(move)
            except Exception:
                continue
            if cat.name in self.brain.actions:
                categorias.add(cat.name)
        if battle.available_switches:
            categorias.add("SWITCH_DEFENSIVE")
            categorias.add("SWITCH_OFFENSIVE")

        valid_actions = self._expand_with_mechanic(list(categorias), battle)
        if not valid_actions:
            valid_actions = ["ATTACK_STRONG"]
        return valid_actions, []
