"""
scripts/jogar_contra_instinto.py — joga TU contra o InstinctBot, no navegador.

PARA QUE SERVE
--------------
O InstinctBot e a regua do projeto, e vale 74,37% contra o MaxDamage. Mas nenhuma
metrica substitui ver o que ele faz turno a turno contra um humano: os erros de
politica que se notam a jogar sao os que nenhum CSV mostra.

Este script poe o bot a ESPERAR DESAFIOS teus. Tu abres o Showdown no navegador,
desafias o nome dele, e jogas normalmente.


CREDENCIAIS DO BOT
------------------
Estao FIXAS nas constantes `BOT_NOME` e `BOT_SENHA` no topo do codigo.

ANTES DE USAR NO SERVIDOR OFICIAL, e preciso REGISTAR esse nome uma vez:

    1. abrir play.pokemonshowdown.com
    2. escolher o nome que esta em BOT_NOME
    3. Settings -> Register, e definir exatamente a password de BOT_SENHA

Sem esse registo, o servidor responde `|nametaken|... token was invalid` se o nome ja
pertencer a outra pessoa, ou entra como nao registado se estiver livre.

SAO DUAS CONTAS DIFERENTES. O bot usa BOT_NOME; tu entras no navegador com o TEU nome.
Se ambos usarem o mesmo, o servidor recusa a segunda ligacao.

`SHOWDOWN_PASS`, se definida, TEM PRECEDENCIA sobre BOT_SENHA. Serve para trocar de
conta sem editar o ficheiro.

DUAS NOTAS QUE VALEM O QUE VALEM. Uma credencial escrita num ficheiro de codigo fica
no historico do git mesmo depois de removida — se este projeto for para um repositorio
publico, convem passar para a variavel de ambiente. E a password do bot devia ser
DIFERENTE da da tua conta pessoal: sao contas com valor diferente e uma delas vai
correr sozinha durante horas.


USO
---
Local (recomendado: mais rapido, sem regras de ladder, e e onde o projeto ja corre):

    node pokemon-showdown start --no-security          # noutro terminal
    python -m scripts.jogar_contra_instinto

Servidor oficial (play.pokemonshowdown.com):

    python -m scripts.jogar_contra_instinto --oficial --bot AlfineteBot

Por omissao aceita desafios de QUALQUER PESSOA. Para so aceitar de um nome:

    python -m scripts.jogar_contra_instinto --humano Vylleon


NOMES REGISTADOS NO SERVIDOR OFICIAL
------------------------------------
Nomes NAO REGISTADOS entram sem password. Se o nome escolhido ja estiver registado
por alguem, o servidor responde:

    |nametaken|<nome>|Your authentication token was invalid.

e o poke-env rebenta com um `AssertionError: Expected <nome> to be logged in`, que
nao diz o que se passa. Nesse caso: ou se escolhe um nome livre (`--bot` com uns
digitos ao acaso), ou se regista o nome no Showdown e se define `SHOWDOWN_PASS`.

Depois, no navegador: procura o nome do bot em "Find a user" e clica em "Challenge".
Escolhe o formato `[Gen 9] National Dex` e usa um dos teus times.

    --batalhas N   quantas aceitar antes de sair (default 1). Se vais deixa-lo
                   aberto um bocado, poe um numero maior — ao fim de N desafios o
                   script sai e deixa de responder
    --time N       indice do time do pool de treino (default: sorteia)
    --humano NOME  aceitar SO deste nome (default: de qualquer pessoa)

NOTA SOBRE O SERVIDOR OFICIAL
-----------------------------
Aceitar DESAFIOS de quem quer que seja e uma coisa; por o bot a jogar NO LADDER e
outra. O Smogon tem regras proprias sobre bots em ladder, e este script nunca ladderiza
— so responde a desafios diretos.

Se o deixares aberto no servidor oficial, conta que estranhos o desafiem. Nao ha risco
de dados: nao ha aprendizado, nao ha ficheiro a gravar, o cerebro nem sequer existe
neste agente. O unico custo e o teu tempo e o do processo.
"""

import argparse
import asyncio
import logging
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration

from instinct.instinct_player import InstinctBot
from shared import diagnostico
from shared.env.teams_train import RandomTeamFromPool, TEAMS_LIST

LOCAL = ServerConfiguration("ws://localhost:8000/showdown/websocket",
                            "http://localhost:8000/")
FORMATO = "gen9nationaldex"

# ----------------------------------------------------------------------------
# CREDENCIAIS FIXAS DO BOT
# ----------------------------------------------------------------------------
# Registar UMA VEZ em play.pokemonshowdown.com: escolher este nome, ir a
# Settings -> Register e definir exatamente esta password.
#
# So sao usadas com `--oficial`. O servidor local com `--no-security` nao pede nada.
# A variavel de ambiente SHOWDOWN_PASS, se existir, tem precedencia sobre BOT_SENHA.
BOT_NOME = "AlfineteInstinto"
BOT_SENHA = "Cast2527"

# AS MAIUSCULAS IMPORTAM AQUI, ao contrario do que parece.
#
# Para IDENTIFICAR A CONTA, o Showdown normaliza: "alfineteinstinto" e
# "AlfineteInstinto" sao a mesma conta e a autenticacao funciona nas duas formas.
#
# MAS o poke-env compara por STRING EXATA o nome que pediu com o que o servidor
# devolve, e o servidor devolve o NOME DE EXIBICAO REGISTADO (com maiusculas). Se
# nao baterem certo:
#
#     WARNING - Trying to login as alfineteinstinto, showdown returned
#               AlfineteInstinto - this might prevent future actions
#     AssertionError: Expected alfineteinstinto to be logged in.
#
# O login teve SUCESSO, mas o evento `logged_in` nunca e marcado, logo o
# `accept_challenges` nunca arranca e o desafio fica a espera para sempre.
#
# BOT_NOME tem de ser a grafia EXATA com que a conta foi registada.


async def main(args):
    if args.oficial:
        servidor = ShowdownServerConfiguration
        # A variavel de ambiente ganha a constante: permite trocar de conta sem
        # editar o ficheiro.
        password = os.environ.get("SHOWDOWN_PASS") or (BOT_SENHA or None)
        if password:
            origem = "SHOWDOWN_PASS" if os.environ.get("SHOWDOWN_PASS") else "BOT_SENHA"
            print(f"[LOGIN] a autenticar '{args.bot}' (password de {origem})")
            print("        Se der 'nametaken ... token was invalid', o nome ainda NAO")
            print("        esta registado com esta password. Registar em")
            print("        play.pokemonshowdown.com -> Settings -> Register.")
        if password is None:
            print("[AVISO] SHOWDOWN_PASS nao definida: a entrar como nome NAO")
            print("        REGISTADO. So funciona se o nome estiver LIVRE.")
            print()
            print("        Se aparecer:")
            print("            |nametaken|<nome>|Your authentication token was invalid.")
            print("        o nome JA ESTA REGISTADO por alguem. Duas saidas:")
            print()
            print("          1. usar um nome livre:   --bot AlfineteQV7391")
            print("             (confirma antes no navegador em 'Find a user')")
            print("          2. registar o nome no Showdown e definir a variavel:")
            print('             $env:SHOWDOWN_PASS = "a-password-do-bot"')
            print()
            print("        O traceback do poke-env nao explica nada disto: mostra um")
            print("        AssertionError 'Expected <nome> to be logged in'.")
    else:
        servidor = LOCAL
        password = None      # o servidor local com --no-security nao pede nada

    # Time: fixo se `--time` for dado, sorteado do pool caso contrario. Fixar e util
    # para repetir a mesma situacao varias vezes e ver se a decisao do bot muda.
    if args.time is not None:
        if not 0 <= args.time < len(TEAMS_LIST):
            raise SystemExit(f"--time fora do intervalo 0..{len(TEAMS_LIST) - 1}")
        equipa = TEAMS_LIST[args.time]
        print(f"[TIME] fixo, indice {args.time}")
    else:
        equipa = RandomTeamFromPool(TEAMS_LIST)
        print(f"[TIME] sorteado de {len(TEAMS_LIST)} times do pool de treino")

    bot = InstinctBot(
        account_configuration=AccountConfiguration(args.bot, password),
        server_configuration=servidor,
        battle_format=FORMATO,
        team=equipa,
        max_concurrent_battles=1,
        # DIAGNOSTICO SEMPRE LIGADO NAS BATALHAS MANUAIS (03/09/2026).
        #
        # Em 03/09 os avisos passaram a estar DESLIGADOS por omissao, para nao
        # poluirem o log de um treino de 400k onde o InstinctBot corre como
        # adversario no mesmo stdout do agente. O efeito colateral foi apagar a
        # instrumentacao AQUI — e uma batalha manual sem instrumentacao nao serve
        # para o fim a que se destina. Este script e a UNICA ferramenta que apanha
        # defeitos que nenhuma metrica mostra (6.35, 6.44), e o `print` nao e
        # ruido: e o produto.
        diagnostico=True,
        # TIMER LIGADO, ao contrario do treino: com um humano do outro lado, uma
        # batalha sem timer pode ficar pendurada indefinidamente se fechares o
        # separador. Nos treinos esta desligado para o timeout nao contaminar os
        # dados; aqui nao ha dados a contaminar.
        start_timer_on_battle_start=True,
        # WARNING e nao CRITICAL: aqui queres VER o que se passa. As mensagens de
        # rejeicao de equipa e os [Invalid choice] aparecem, e sao precisamente o
        # que interessa quando algo corre mal numa batalha manual.
        log_level=logging.WARNING,
    )

    # Liga tambem o interruptor GLOBAL: o `diagnostico=True` acima so alcanca o
    # InstinctBot, e as sondas do `masking` e do `execution` leem daqui.
    diagnostico.ligar()

    onde = "play.pokemonshowdown.com" if args.oficial else "localhost:8000"
    print()
    print("=" * 70)
    print(f"  InstinctBot ligado a {onde}")
    print(f"  Nome do bot   : {args.bot}")
    print(f"  A aceitar de  : {args.humano or 'QUALQUER PESSOA'}")
    print(f"  Formato       : {FORMATO}")
    print(f"  Batalhas      : {args.batalhas}")
    print("=" * 70)
    print()
    print("  DIAGNOSTICO LIGADO: vais ver [DECISAO], [HAZ], [QUAR] e [IMUNE].")
    print()
    print("  No navegador: procura o nome do bot em 'Find a user' -> 'Challenge',")
    print(f"  escolhe o formato [Gen 9] National Dex e envia o desafio.")
    print()

    # `opponent=None` no poke-env significa ACEITAR DE QUALQUER PESSOA. Com um nome,
    # so esse nome consegue desafiar.
    await bot.accept_challenges(args.humano, args.batalhas)

    print()
    print("=" * 70)
    print(f"  Terminado. {bot.n_won_battles} vitorias em {bot.n_finished_battles} "
          f"batalhas do bot.")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Joga contra o InstinctBot no Pokemon Showdown.")
    ap.add_argument("--humano", default=None,
                    help="restringe os desafios a este nome. Por omissao aceita de "
                         "QUALQUER PESSOA")
    ap.add_argument("--bot", default=BOT_NOME,
                    help=f"nome do BOT. Tem de ser DIFERENTE do teu: o servidor nao "
                         f"aceita duas ligacoes com o mesmo nome (default: "
                         f"{BOT_NOME}). Mudar o nome sem mudar BOT_SENHA so funciona "
                         f"se o nome novo estiver livre ou registado com a mesma "
                         f"password")
    ap.add_argument("--oficial", action="store_true",
                    help="ligar a play.pokemonshowdown.com em vez do servidor local")
    ap.add_argument("--batalhas", type=int, default=1,
                    help="quantos desafios aceitar antes de sair (default 1)")
    ap.add_argument("--time", type=int, default=None,
                    help="indice do time a usar. Omitir para sortear do pool")
    args = ap.parse_args()

    if args.humano and args.bot.lower() == args.humano.lower():
        raise SystemExit("O nome do bot tem de ser DIFERENTE do teu: o servidor nao "
                         "aceita duas ligacoes com o mesmo nome.")

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(args))
    else:
        asyncio.run(main(args))
