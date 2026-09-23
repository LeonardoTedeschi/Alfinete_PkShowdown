"""
scripts/jogar_contra_cynthia.py — joga TU contra a CynthiaPlatinumBot no navegador.

OBJETIVO
--------
Ferramenta de teste manual da régua externa Cynthia. O bot fica ligado ao servidor
e espera desafios diretos enviados pelo utilizador no cliente do Pokémon Showdown.

Não há treino, não há cérebro e nenhum ficheiro de modelo é alterado.

USO LOCAL
---------
1. Noutro terminal, iniciar o Showdown local:

       node pokemon-showdown start --no-security

2. Iniciar a Cynthia:

       python -m scripts.jogar_contra_cynthia --humano Vylleon --batalhas 5

3. No navegador, abrir o Showdown local, procurar o nome mostrado pelo script e
enviar um desafio no formato [Gen 9] National Dex.

Exemplos:

    # Pool de treino, times sorteados
    python -m scripts.jogar_contra_cynthia \
        --humano Vylleon \
        --pool treino \
        --batalhas 5 \
        --diagnostico-cynthia

    # Mesmo time da Cynthia em todas as batalhas
    python -m scripts.jogar_contra_cynthia \
        --humano Vylleon \
        --pool treino \
        --time 3 \
        --semente 42 \
        --batalhas 5 \
        --diagnostico-cynthia

    # Pool holdout
    python -m scripts.jogar_contra_cynthia \
        --humano Vylleon \
        --pool holdout \
        --batalhas 5

SERVIDOR OFICIAL
----------------
Por omissão o script usa localhost:8000.

Para o servidor oficial:

    $env:SHOWDOWN_PASS = "password-da-conta-do-bot"
    python -m scripts.jogar_contra_cynthia \
        --oficial \
        --bot AlfineteCynthia \
        --humano Vylleon \
        --batalhas 5

Se SHOWDOWN_PASS não estiver definida, o login só funciona se o nome escolhido
estiver livre e puder entrar como conta não registada.

O script apenas aceita desafios diretos. Não entra no ladder.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration

from shared.env.cynthia_platinum import (
    CynthiaPlatinumBot,
    BASELINE_NAME,
    BASELINE_SPEC_VERSION,
    AI_PROFILE,
    BATTLE_FORMAT,
)
from shared.env.teams_train import (
    RandomTeamFromPool,
    TEAMS_LIST as TIMES_TREINO,
)
from shared.env.teams_eval import TEAMS_LIST as TIMES_HOLDOUT


LOCAL = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/",
)

FORMATO = BATTLE_FORMAT
BOT_NOME_PADRAO = "AlfineteCynthia"

POOLS = {
    "treino": TIMES_TREINO,
    "holdout": TIMES_HOLDOUT,
}


def _ai_profile_texto() -> str:
    if isinstance(AI_PROFILE, (tuple, list)):
        return " | ".join(str(x) for x in AI_PROFILE)
    return str(AI_PROFILE)


def _resolver_time(pool_nome: str, indice: int | None):
    pool = POOLS[pool_nome]

    if indice is None:
        # Mantém o mesmo comportamento usado nas avaliações automáticas:
        # cada batalha recebe um time sorteado deste pool.
        return RandomTeamFromPool(pool), f"sorteado do pool '{pool_nome}' ({len(pool)} times)"

    if indice < 0 or indice >= len(pool):
        raise SystemExit(
            f"--time fora do intervalo 0..{len(pool) - 1} para o pool '{pool_nome}'."
        )

    # Passar diretamente a string do time fixa o mesmo time em todas as batalhas.
    return pool[indice], f"fixo: pool '{pool_nome}', índice {indice}"


async def main(args):
    random.seed(args.semente)
    np.random.seed(args.semente)

    if args.oficial:
        servidor = ShowdownServerConfiguration
        password = os.environ.get("SHOWDOWN_PASS") or None
        localizacao = "play.pokemonshowdown.com"

        if password:
            print(f"[LOGIN] a autenticar '{args.bot}' usando SHOWDOWN_PASS")
        else:
            print("[AVISO] SHOWDOWN_PASS não definida.")
            print("        O modo oficial só entrará se o nome estiver livre")
            print("        ou puder autenticar como conta não registada.")
            print()
    else:
        servidor = LOCAL
        password = None
        localizacao = "localhost:8000"

    equipa, descricao_time = _resolver_time(args.pool, args.time)

    cynthia = CynthiaPlatinumBot(
        account_configuration=AccountConfiguration(args.bot, password),
        server_configuration=servidor,
        battle_format=FORMATO,
        team=equipa,
        max_concurrent_battles=1,
        start_timer_on_battle_start=True,
        log_level=logging.WARNING,
        seed=args.semente,
        diagnostico=args.diagnostico_cynthia,
    )

    print()
    print("=" * 78)
    print(f"  {BASELINE_NAME} v{BASELINE_SPEC_VERSION} pronta para desafios")
    print(f"  Servidor      : {localizacao}")
    print(f"  Nome do bot   : {args.bot}")
    print(f"  Aceita de     : {args.humano or 'QUALQUER PESSOA'}")
    print(f"  Formato       : {FORMATO}")
    print(f"  Pool          : {args.pool}")
    print(f"  Time          : {descricao_time}")
    print(f"  Semente       : {args.semente}")
    print(f"  AI Profile    : {_ai_profile_texto()}")
    print(f"  Diagnóstico   : {'LIGADO' if args.diagnostico_cynthia else 'desligado'}")
    print(f"  Batalhas      : {args.batalhas}")
    print("=" * 78)
    print()
    print("No navegador:")
    print(f"  1. Procure '{args.bot}' em 'Find a user'")
    print("  2. Clique em 'Challenge'")
    print("  3. Escolha [Gen 9] National Dex")
    print("  4. Envie o desafio")
    print()

    if args.diagnostico_cynthia:
        print(
            "[DIAGNÓSTICO] A Cynthia imprimirá a decisão selecionada e o score "
            "durante as batalhas."
        )
        print()

    # opponent=None significa aceitar desafios de qualquer utilizador.
    # Com --humano, só o nome indicado é aceito.
    await cynthia.accept_challenges(args.humano, args.batalhas)

    print()
    print("=" * 78)
    print(
        f"  Terminado: {cynthia.n_won_battles} vitórias em "
        f"{cynthia.n_finished_battles} batalhas da Cynthia."
    )
    print("=" * 78)


def construir_parser():
    ap = argparse.ArgumentParser(
        description="Joga manualmente contra a CynthiaPlatinumBot no Pokémon Showdown."
    )
    ap.add_argument(
        "--humano",
        default=None,
        help=(
            "aceitar desafios apenas deste nome. "
            "Por omissão aceita de qualquer pessoa"
        ),
    )
    ap.add_argument(
        "--bot",
        default=BOT_NOME_PADRAO,
        help=f"nome da conta usada pela Cynthia (default: {BOT_NOME_PADRAO})",
    )
    ap.add_argument(
        "--oficial",
        action="store_true",
        help="usar play.pokemonshowdown.com em vez de localhost:8000",
    )
    ap.add_argument(
        "--batalhas",
        type=int,
        default=1,
        help="número de desafios a aceitar antes de sair (default: 1)",
    )
    ap.add_argument(
        "--pool",
        choices=sorted(POOLS),
        default="treino",
        help="pool de times da Cynthia (default: treino)",
    )
    ap.add_argument(
        "--time",
        type=int,
        default=None,
        help=(
            "índice do time dentro do pool. "
            "Omitir para usar RandomTeamFromPool"
        ),
    )
    ap.add_argument(
        "--semente",
        type=int,
        default=42,
        help="semente da Cynthia e do sorteio de times (default: 42)",
    )
    ap.add_argument(
        "--diagnostico-cynthia",
        action="store_true",
        help="liga as mensagens internas de decisão da Cynthia",
    )
    return ap


if __name__ == "__main__":
    args = construir_parser().parse_args()

    if args.batalhas <= 0:
        raise SystemExit("--batalhas deve ser maior que zero.")

    if args.humano and args.bot.lower() == args.humano.lower():
        raise SystemExit(
            "O nome do bot deve ser diferente do nome humano: "
            "o servidor não aceita duas ligações com a mesma conta."
        )

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(args))
    else:
        asyncio.run(main(args))
