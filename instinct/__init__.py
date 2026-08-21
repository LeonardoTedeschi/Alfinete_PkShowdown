"""
Pacote `instinct` — o conhecimento de domínio do projeto ALFINETE, decomposto em
componentes independentes e injetáveis.

O antigo monólito `InstinctCore` (2099 linhas, uma classe com ~50 métodos acoplados
por self.) foi substituído por SEIS componentes, cada um testável isoladamente:

  Camada 0  definitions.py  enums (Role, MatchupState, TacticalMode, MoveCategory)
  Camada 1  physics.py      GamePhysics    — regras do jogo (dano, stats, papéis)
  Camada 2  state.py        StateParser    — battle -> tupla de estado da Q-table
  Camada 3  masking.py      ActionMasker   — poda de ações inválidas
  Camada 4  policy.py       InstinctPolicy — decide a INTENÇÃO (ranking de categorias)
  Camada 4b execution.py    InstinctExecutor — traduz intenção em GOLPE/TROCA concreto

Regra de dependência: cada camada só depende das de cima. As setas apontam todas na
mesma direção, o que torna o grafo legível e o comportamento de cada agente
verificável ("o Green usa Camadas 1-2; o Blue usa 1-4b; o Instinto-puro usa 1-4b").

Este __init__ oferece duas formas de usar o instinto:

1. `build_instinct()` — fábrica que monta e liga os 6 componentes com a injeção
   correta, devolvendo-os prontos. É o que os agentes usam.

2. `InstinctCore` — uma FACHADA opcional que agrupa os componentes num único objeto
   com a mesma interface pública do monólito antigo (get_state, get_instinct_profile,
   get_best_execution_object...). Serve para código legado que espera "um core" e
   para testes de equivalência com o monólito. Internamente é só composição — não há
   lógica nova aqui.
"""

from shared.definitions import Role, MatchupState, TacticalMode, MoveCategory
from shared.physics import GamePhysics
from shared.state import StateParser, STATE_DIM
from instinct.masking import ActionMasker
from instinct.policy import InstinctPolicy
from instinct.execution import InstinctExecutor

__all__ = [
    "Role", "MatchupState", "TacticalMode", "MoveCategory", "STATE_DIM",
    "GamePhysics", "StateParser", "ActionMasker", "InstinctPolicy", "InstinctExecutor",
    "build_instinct", "InstinctComponents", "InstinctCore",
]


class InstinctComponents:
    """Contentor simples dos 6 componentes já ligados. Um agente pega só o que precisa."""

    def __init__(self, physics, parser, masker, policy, executor):
        self.physics = physics    # Camada 1
        self.parser = parser      # Camada 2
        self.masker = masker      # Camada 3
        self.policy = policy      # Camada 4  (decisão)
        self.executor = executor  # Camada 4b (execução)


def build_instinct():
    """Fábrica: instancia e liga os componentes na ordem de dependência correta.

    Uma única instância de GamePhysics é partilhada por todos (é sem estado), o que
    garante que parser, masker, policy e executor raciocinam sobre a mesma física.
    """
    physics = GamePhysics()
    parser = StateParser(physics)
    masker = ActionMasker(physics)
    policy = InstinctPolicy(physics, parser, masker)
    executor = InstinctExecutor(physics, parser, masker)
    return InstinctComponents(physics, parser, masker, policy, executor)


class InstinctCore:
    """Fachada de compatibilidade: expõe a interface pública do monólito antigo,
    delegando para os componentes. Útil para o Agente Instinto-puro e para testes de
    equivalência. NÃO contém lógica — apenas encaminha chamadas.
    """

    def __init__(self):
        c = build_instinct()
        self.physics = c.physics
        self.parser = c.parser
        self.masker = c.masker
        self.policy = c.policy
        self.executor = c.executor

    # -- interface de estado (Camada 2) ------------------------------------
    def get_state(self, battle):
        return self.parser.get_state(battle)

    # -- interface de masking (Camada 3) -----------------------------------
    def get_available_actions(self, battle):
        return self.masker.get_available_actions(battle)

    def is_move_useless(self, move, opponent, battle, history=None):
        return self.masker.is_move_useless(move, opponent, battle, history)

    # -- interface de decisão (Camada 4) -----------------------------------
    def get_instinct_profile(self, battle, history=None):
        return self.policy.get_instinct_profile(battle, history)

    # -- interface de execução (Camada 4b) ---------------------------------
    def get_best_execution_object(self, base_action, battle, history=None):
        return self.executor.get_best_execution_object(base_action, battle, history)

    def get_best_lead(self, battle):
        return self.executor.get_best_lead(battle)

    def get_post_faint_switch(self, battle, history=None):
        return self.executor.get_post_faint_switch(battle, history)
