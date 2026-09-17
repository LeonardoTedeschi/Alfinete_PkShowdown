"""
shared/mechanics.py — validacao de mecanicas de turno (Mega Evolucao e Z-Move).

PORQUE ESTE FICHEIRO EXISTE
---------------------------
Os atributos `battle.can_mega_evolve` e `battle.can_z_move` do poke-env NAO
significam o que o nome sugere. Foram observadas ordens invalidas em producao no
servidor local, a 24/08/2026:

    [Invalid choice] Can't move: Tyranitar can't use Stone Edge as a Z-move
    [Invalid choice] Can't move: Tapu Koko can't mega evolve
    [Invalid choice] Can't move: Porygon-Z can't mega evolve

Nenhum destes Pokemon carregava o item respetivo.

  - `Move.can_z_move` e propriedade do GOLPE NOS DADOS ("existe uma versao Z deste
    golpe?"), nao da situacao ("tenho o cristal certo?").
  - `battle.can_mega_evolve` comporta-se como flag ao nivel da BATALHA e nao do
    Pokemon ativo: mantem-se verdadeiro depois de trocar para um Pokemon sem pedra.

Uma ordem invalida custa o TURNO INTEIRO (o agente cai no fallback aleatorio) e
envenena os dados. Por isso a verificacao e feita pelo ITEM, que e informacao que
temos com certeza sobre o nosso proprio Pokemon.

Regra de ouro: na duvida, devolver False e jogar sem mecanica.

A TERASTALIZACAO nao aparece aqui: esta banida no gen9nationaldex por Terastal
Clause, logo `battle.can_tera` e sempre falso neste formato.


ATUALIZACAO DE COMPATIBILIDADE — poke-env 0.12.0+ (27/08/2026)
-------------------------------------------------------------
A versao 0.12.0 corrigiu um bug com efeito colateral direto neste ficheiro:

    "Don't get rid of z item when using z move"  (PR #801)

Na versao antiga (0.11.0) o cristal Z DESAPARECIA do `pokemon.item` depois de ser
usado. A verificacao por item funcionava por acidente como guarda de uso unico: sem
item, nao havia segunda tentativa.

Com o bug corrigido, o item PERMANECE. A verificacao por item continua correta para
"tenho o cristal certo?", mas deixa de impedir uma SEGUNDA tentativa de Z-move na
mesma batalha, que o servidor rejeita com [Invalid choice].

O mesmo raciocinio vale para a Mega: a pedra nunca desaparecia, e a unica guarda era
o `can_mega_evolve`, que ja sabemos ser pouco de confiar.

Solucao: um registo proprio de uso por batalha (`_USOS`). Nao depende de nenhum
atributo do poke-env, logo e imune a mudancas de versao nos dois sentidos.
"""

from collections import OrderedDict

# Cristais Z -> tipo do golpe que ativam.
Z_CRYSTAL_MAP = {
    'wateriumz': 'water', 'normaliumz': 'normal', 'flyiniumz': 'flying',
    'rockiumz': 'rock', 'electriumz': 'electric', 'darkiniumz': 'dark',
    'iciumz': 'ice', 'grassiumz': 'grass', 'firiumz': 'fire',
    'poisoniumz': 'poison', 'ghostiumz': 'ghost', 'psychiumz': 'psychic',
    'steeliumz': 'steel', 'groundiumz': 'ground', 'dragoniumz': 'dragon',
    'bugiumz': 'bug', 'fightiniumz': 'fighting', 'fairiumz': 'fairy',
}

# Pedras Mega. Lista EXPLICITA de proposito: uma regra por sufixo ("acaba em -ite")
# apanharia o EVIOLITE, que esta no pool de treino e NAO e pedra Mega.
MEGA_STONES = {
    'abomasite', 'absolite', 'aerodactylite', 'aggronite', 'alakazite',
    'altarianite', 'ampharosite', 'audinite', 'banettite', 'beedrillite',
    'blastoisinite', 'blazikenite', 'cameruptite', 'charizarditex',
    'charizarditey', 'diancite', 'galladite', 'garchompite', 'gardevoirite',
    'gengarite', 'glalitite', 'gyaradosite', 'heracronite', 'houndoominite',
    'kangaskhanite', 'latiasite', 'latiosite', 'lopunnite', 'lucarionite',
    'manectite', 'mawilite', 'medichamite', 'metagrossite', 'mewtwonitex',
    'mewtwonitey', 'pidgeotite', 'pinsirite', 'sablenite', 'salamencite',
    'sceptilite', 'scizorite', 'sharpedonite', 'slowbronite', 'steelixite',
    'swampertite', 'tyranitarite', 'venusaurite',
}

# ---------------------------------------------------------------------------
# Registo de uso por batalha
# ---------------------------------------------------------------------------
# Mega e Z sao recursos de UMA UTILIZACAO por batalha. Este registo e a unica
# garantia que nao depende do poke-env.
#
# Limite de entradas em vez de limpeza por callback: evita ter de ligar isto ao
# ciclo de vida da batalha em tres ficheiros diferentes. Com concorrencia 3, 4096
# entradas cobrem folgadamente as batalhas vivas, e as antigas caem por antiguidade.
_USOS = OrderedDict()
_MAX_BATALHAS_REGISTADAS = 4096


def _tag(battle):
    return getattr(battle, "battle_tag", None) or id(battle)


def ja_usou(battle, tipo):
    """Ja gastamos `tipo` ('mega' ou 'z') nesta batalha?"""
    return tipo in _USOS.get(_tag(battle), ())


def marcar_uso(battle, tipo):
    """Regista que `tipo` foi gasto nesta batalha.

    Chamar SEMPRE imediatamente antes de devolver a ordem com a flag ligada.
    """
    t = _tag(battle)
    if t in _USOS:
        _USOS[t].add(tipo)
        _USOS.move_to_end(t)
    else:
        _USOS[t] = {tipo}
        if len(_USOS) > _MAX_BATALHAS_REGISTADAS:
            _USOS.popitem(last=False)


def _item_do_ativo(battle):
    active = getattr(battle, "active_pokemon", None)
    if not active:
        return None, ""
    return active, str(getattr(active, "item", "") or "").lower()


def mega_valido(battle):
    """O Pokemon ATIVO pode mega-evoluir NESTE turno?

    Tres condicoes: a batalha permitir, o ativo carregar uma pedra Mega, e a Mega
    ainda nao ter sido gasta nesta batalha.

    A Mega nao tem restricao de GOLPE: funciona com qualquer ataque. Por isso esta
    funcao recebe so a batalha.
    """
    if ja_usou(battle, "mega"):
        return False
    if not getattr(battle, "can_mega_evolve", False):
        return False
    active, item = _item_do_ativo(battle)
    if not active or getattr(active, "fainted", False):
        return False
    return item in MEGA_STONES


def z_move_valido(obj, battle):
    """Este golpe pode SAIR como Z-move NESTE turno?

    Ao contrario da Mega, o Z-move depende do GOLPE: o cristal so ativa golpes do
    seu tipo. Verifica-se por duas vias, pela ordem de fiabilidade.
    """
    if ja_usou(battle, "z"):
        return False
    if not getattr(battle, "can_z_move", False):
        return False
    active, item = _item_do_ativo(battle)
    if not active or getattr(active, "fainted", False):
        return False

    # 1. Lista oficial do poke-env, quando a versao instalada a expoe.
    lista = getattr(active, "available_z_moves", None)
    if lista:
        try:
            return any(getattr(m, "id", None) == getattr(obj, "id", None) for m in lista)
        except TypeError:
            pass

    # 2. Fallback: cristal no item E tipo do golpe igual ao tipo do cristal.
    tipo_exigido = Z_CRYSTAL_MAP.get(item)
    if not tipo_exigido:
        return False
    tipo_golpe = getattr(obj, "type", None)
    return bool(tipo_golpe and tipo_golpe.name.lower() == tipo_exigido)


def e_golpe(obj):
    """Distingue um Move de um Pokemon (trocas nunca usam mecanica)."""
    return hasattr(obj, "id") and hasattr(obj, "base_power")
