"""Regras de Last Resort/Sleep Talk, sem preferencias de politica.

O replay do poke-env 0.16.1 guarda eventos mesmo com save_replays=False.
O acesso privado fica isolado aqui: sem eventos, a prontidao e desconhecida,
nao inventada a partir de PP totais ou de ordens apenas escolhidas.
"""


class MovePoolBattle:
    """Visao somente de leitura com candidatos restritos; nao altera a batalha."""

    def __init__(self, battle, moves):
        self._source_battle = battle
        self.available_moves = list(moves)

    def __getattr__(self, name):
        return getattr(self._source_battle, name)


def protocol_events(battle):
    events = getattr(battle, '_replay_data', None)
    return events if isinstance(events, list) else None


def _move_id(name):
    return ''.join(c for c in str(name).lower() if c.isalnum())


def last_resort_ready(pokemon, battle):
    """True/False/None: outros slots usados desde entrada/Transform.

    Le o moveset inteiro, inclusive golpes bloqueados por Choice/sem PP.
    Golpes chamados por Sleep Talk nao gastam PP nem marcam o slot 'used'.
    Falhas apos o inicio de um golpe contam; cant/flinch/paralisia nao contam.
    """
    moves = getattr(pokemon, 'moves', {}) or {}
    if 'lastresort' not in moves or len(moves) < 2:
        return False
    if battle is None or pokemon is not getattr(battle, 'active_pokemon', None):
        return None
    events = protocol_events(battle)
    side = getattr(battle, 'player_role', None)
    if events is None or not side:
        return None
    used = set()
    for event in reversed(events):
        if len(event) < 3 or not str(event[2]).startswith(side + 'a:'):
            continue
        kind = event[1]
        if kind in ('switch', 'drag', '-transform'):
            return (set(moves) - {'lastresort'}).issubset(used)
        if kind == 'move' and len(event) >= 4:
            if not any(str(x).startswith('[from]') for x in event[4:]):
                used.add(_move_id(event[3]))
    # Um replay parcial nao prova que os outros slots nunca foram usados.
    return None


def sleep_talk_attack(move, pokemon):
    """Unico resultado ofensivo no conjunto exato Sleep Talk + Last Resort.

    Sleep Talk gasta PP (marcando seu slot) ANTES de chamar Last Resort.
    Portanto o ataque chamado ja satisfaz o requisito no primeiro uso.
    Outros conjuntos de Sleep Talk mantem a abstracao existente.
    """
    if getattr(move, 'id', None) != 'sleeptalk' or pokemon is None:
        return None
    asleep = getattr(getattr(pokemon, 'status', None), 'name', '') == 'SLP'
    if not asleep and getattr(pokemon, 'ability', None) != 'comatose':
        return None
    moves = getattr(pokemon, 'moves', {}) or {}
    if set(moves) == {'sleeptalk', 'lastresort'}:
        return moves['lastresort']
    return None
