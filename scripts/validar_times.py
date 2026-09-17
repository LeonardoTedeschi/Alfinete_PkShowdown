"""
scripts/validar_times.py — descobre QUAL time de um pool o servidor rejeita.

PORQUE EXISTE
-------------
Um time invalido no formato nao produz excecao no poke-env: o servidor rejeita o
desafio e o processo fica a espera de uma batalha que nunca comeca. O sintoma e um
travamento silencioso, sem erro, depois de N batalhas bem sucedidas — porque o
RandomTeamFromPool so sorteia o time mau ao fim de algumas tentativas.

Foi o que aconteceu com `shared/env/teams_eval.py` a 25/08/2026: os 15 times de
holdout foram escritos mas NUNCA validados contra o servidor, e a avaliacao de
generalizacao travava sempre que um deles saia no sorteio.

Este script testa os times UM A UM, com timeout. Um time que nao consiga comecar uma
batalha dentro do tempo e marcado como invalido, e o processo continua para o
seguinte em vez de pendurar.

NAO usa cerebro nenhum e nao altera ficheiros. So testa times.

USO
---
    python -m scripts.validar_times --pool eval
    python -m scripts.validar_times --pool treino
    python -m scripts.validar_times --pool eval --timeout 45

Correr com o servidor local a funcionar.
"""

import argparse
import asyncio
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder

from shared.env.maxdamage import MaxDamagePlayer

LOCAL = ServerConfiguration("ws://localhost:8000/showdown/websocket",
                            "http://localhost:8000/")
FORMATO = "gen9nationaldex"


def carregar_pool(nome):
    if nome == "eval":
        from shared.env.teams_eval import TEAMS_LIST
        return TEAMS_LIST, "teams_eval.py"
    from shared.env.teams_train import TEAMS_LIST
    return TEAMS_LIST, "teams_train.py"


def primeira_linha(equipa):
    """Nome do primeiro Pokemon, para identificar o time na saida."""
    for linha in str(equipa).strip().splitlines():
        linha = linha.strip()
        if linha:
            return linha.split("@")[0].strip()[:28]
    return "(vazio)"


async def testar(indice, equipa, timeout, nivel_log):
    """Tenta UMA batalha com este time fixo dos dois lados.

    Time fixo dos dois lados de proposito: se a batalha comecar, o time e aceite pelo
    servidor. Se nao comecar dentro do timeout, foi rejeitado.
    """
    a = MaxDamagePlayer(
        account_configuration=AccountConfiguration(f"ValidaA{indice}", None),
        server_configuration=LOCAL, battle_format=FORMATO,
        team=ConstantTeambuilder(equipa),
        max_concurrent_battles=1,
        start_timer_on_battle_start=True,   # rede de seguranca: nunca pendurar
        log_level=nivel_log,
    )
    b = MaxDamagePlayer(
        account_configuration=AccountConfiguration(f"ValidaB{indice}", None),
        server_configuration=LOCAL, battle_format=FORMATO,
        team=ConstantTeambuilder(equipa),
        max_concurrent_battles=1,
        start_timer_on_battle_start=True,
        log_level=nivel_log,
    )
    try:
        await asyncio.wait_for(a.battle_against(b, n_battles=1), timeout=timeout)
        return (a.n_finished_battles >= 1), ""
    except asyncio.TimeoutError:
        return False, f"nenhuma batalha comecou em {timeout}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def main(args):
    equipas, ficheiro = carregar_pool(args.pool)
    nivel = logging.WARNING if args.verboso else logging.CRITICAL

    print()
    print("=" * 78)
    print(f"  VALIDACAO DE TIMES — {ficheiro} ({len(equipas)} times)")
    print(f"  Formato: {FORMATO} | timeout por time: {args.timeout}s")
    print("=" * 78)
    if not args.verboso:
        print("  (usa --verboso para ver a mensagem do servidor em cada rejeicao)")
    print()

    maus = []
    for i, equipa in enumerate(equipas):
        etiqueta = primeira_linha(equipa)
        print(f"  [{i:2d}] {etiqueta:<30} ", end="", flush=True)
        ok, motivo = await testar(i, equipa, args.timeout, nivel)
        if ok:
            print("OK")
        else:
            print(f"REJEITADO  <- {motivo}")
            maus.append((i, etiqueta, motivo))

    print()
    print("=" * 78)
    if not maus:
        print("  Todos os times sao aceites pelo servidor.")
        print("  Se a avaliacao continua a travar, a causa NAO esta nos times.")
    else:
        print(f"  {len(maus)} time(s) REJEITADO(S) em {len(equipas)}:")
        for i, etiqueta, motivo in maus:
            print(f"    indice {i}: {etiqueta}  ({motivo})")
        print()
        print("  Corrigir ou remover estes times de", ficheiro)
        print("  Causas tipicas: Pokemon banido em National Dex OU, item ilegal,")
        print("  golpe que a especie nao aprende, ou erro de sintaxe no set.")
        print("  Para ver a mensagem exata do servidor, repetir com --verboso.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Valida os times de um pool contra o servidor.")
    ap.add_argument("--pool", choices=["eval", "treino"], default="eval")
    ap.add_argument("--timeout", type=int, default=40,
                    help="segundos a esperar por time antes de o dar como rejeitado")
    ap.add_argument("--verboso", action="store_true",
                    help="mostra as mensagens do servidor (log a WARNING)")
    args = ap.parse_args()

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main(args))
