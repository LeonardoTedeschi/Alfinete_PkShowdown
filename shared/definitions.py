"""
Definições de dados do Instinto (codinome do domínio tático do projeto ALFINETE).

Este módulo contém apenas enumerações e não depende de nenhum outro módulo do
projeto nem do poke-env. É a base partilhada por todas as camadas do instinto
(parsing de estado, física, action masking e política) e também pela camada de
aprendizado, que precisa dos nomes de Role/Matchup para compor o estado da Q-table.

Manter estas definições isoladas garante que qualquer módulo pode importá-las sem
arrastar dependências pesadas.
"""

from enum import Enum


class Role(Enum):
    """Papel tático de um Pokémon no time."""
    SWEEPER = 1
    UTILITY = 2
    TANK = 3


class MatchupState(Enum):
    """
    Estado do confronto direto entre o Pokémon ativo e o oponente ativo,
    derivado da relação de efetividade de tipos (SE = super efetivo,
    N = neutro, NVE = não muito efetivo).
    """
    DOMINANT = 1       # SE vs NVE
    VOLATILE = 2       # SE vs SE
    OFFENSIVE_ADV = 3  # SE vs N
    DEFENSIVE_ADV = 4  # N vs NVE
    DEFENSIVE_DIS = 5  # N vs SE
    OFFENSIVE_DIS = 6  # NVE vs N
    STALEMATE = 7      # NVE vs NVE
    NEUTRAL = 8        # N vs N
    CRITICAL_DIS = 9   # NVE vs SE


class TacticalMode(Enum):
    """Modo tático de alto nível que seleciona o template de prioridades (Camada 2)."""
    PRESS = 1
    CONTEST = 2
    GRIND = 3
    ESCAPE = 4
    LEAD = 5
    WALLBREAK = 6


class MoveCategory(Enum):
    """Categoria funcional de um golpe ou ação, usada no action masking e no ranking."""
    ATTACK_STRONG = 1
    ATTACK_PREDICTIVE = 2
    ATTACK_PIVOT = 3
    ATTACK_TECH = 4
    BUFF = 5
    STATUS = 6
    HEAL = 7
    CLEAN_HAZARD = 8
    PROTECT = 9
    DEBUFF = 10
    STAT_CLEAN = 11
    HEAL_STATUS = 12
    PHAZE = 13
    FIELD_CONTROL = 14
    HAZARD = 15
    BARRIER = 16
    DISRUPTION = 17
    SWITCH_DEFENSIVE = 18
    SWITCH_OFFENSIVE = 19
    UNKNOWN = 20
