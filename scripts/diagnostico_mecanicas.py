"""
scripts/diagnostico_mecanicas.py

DIAGNOSTICO, NAO CORRECAO. Este script nao altera nenhum ficheiro do projeto e nao
treina nada. So OBSERVA e regista.

Responde a uma unica pergunta empirica, da qual depende todo o trabalho de mecanicas:

    O poke-env atualiza `pokemon.types`, `pokemon.base_stats` e `pokemon.species`
    quando ocorre Protean/Libero, Terastalizacao ou Mega Evolucao?

Se atualizar, o instinto ja ve tudo certo (todas as camadas leem ao vivo, sem cache) e
nao ha nada a reescrever nessa frente. Se nao atualizar, ha um bug ativo: o Greninja
com Protean esta no pool de treino desde sempre.

DOIS MODOS
----------
1) observar  — InstinctBot vs MaxDamage, automatico. Cobre o LADO PROPRIO (Protean do
               Greninja) porque o bot ve os seus proprios Pokemon com informacao
               completa.

2) humano    — levanta um MaxDamage no servidor local a aceitar desafios. Tu jogas pelo
               browser (http://localhost:8000) e usas Tera, Mega e Z-Move a mao. O
               script regista o que o poke-env ve do TEU lado, que e o "opponent" na
               perspetiva do bot. E o teste mais importante dos dois, porque em batalha
               real e assim que o instinto ve o adversario: por inferencia do protocolo.

Correr da raiz do projeto:
    python -m scripts.diagnostico_mecanicas --modo observar --batalhas 30
    python -m scripts.diagnostico_mecanicas --modo humano --utilizador OTeuNick

SAIDA
-----
    artefatos/logs/diagnostico_mecanicas_NN.csv   uma linha por snapshot com evento
    consola                                        eventos em tempo real + veredicto
"""

import argparse
import asyncio
import csv
import glob
import logging
import os
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from poke_env import AccountConfiguration, ServerConfiguration

from instinct.instinct_player import InstinctBot
from shared.env.maxdamage import MaxDamagePlayer
from shared.env.teams_train import RandomTeamFromPool, TEAMS_LIST

LOCAL = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

BATTLE_FORMAT = "gen9nationaldex"
CONCORRENCIA = 1  # sequencial: o log fica legivel e a ordem dos turnos preservada
LOGS_DIR = os.path.join(ROOT, "artefatos", "logs")

CABECALHO = [
    "Data", "Batalha", "Turno", "Lado", "Especie", "Tipos", "Terastalizado",
    "TeraType", "Atk", "Spa", "Spe", "Habilidade", "Item", "HP", "EVENTO", "Detalhe",
]


# ---------------------------------------------------------------------------
# Leitura defensiva: os nomes dos atributos variam entre versoes do poke-env,
# e descobrir QUAIS existem faz parte do diagnostico.
# ---------------------------------------------------------------------------

def _tipos(mon):
    try:
        return "/".join(sorted(t.name for t in mon.types if t))
    except Exception:
        return "?"


def _tera_flag(mon):
    """Devolve (terastalizado, tipo_tera) tentando varios nomes de atributo."""
    for nome in ("terastallized", "is_terastallized", "terastalized"):
        v = getattr(mon, nome, None)
        if v is not None:
            flag = bool(v)
            break
    else:
        flag = None  # o atributo nao existe nesta versao do poke-env

    tipo = None
    for nome in ("terastallized_type", "tera_type", "teratype"):
        v = getattr(mon, nome, None)
        if v is not None:
            tipo = getattr(v, "name", str(v))
            break

    return flag, tipo


def _snapshot(mon):
    if mon is None:
        return None
    tera_flag, tera_tipo = _tera_flag(mon)
    bs = getattr(mon, "base_stats", {}) or {}
    return {
        "especie": str(getattr(mon, "species", "?")),
        "tipos": _tipos(mon),
        "tera": "SIM" if tera_flag else ("NAO" if tera_flag is False else "N/D"),
        "tera_tipo": tera_tipo or "",
        "atk": bs.get("atk", ""),
        "spa": bs.get("spa", ""),
        "spe": bs.get("spe", ""),
        "abilidade": str(getattr(mon, "ability", "") or ""),
        "item": str(getattr(mon, "item", "") or ""),
        "hp": f"{getattr(mon, 'current_hp_fraction', 0.0):.2f}",
    }


def _comparar(antes, agora):
    """Devolve (evento, detalhe) descrevendo o que mudou entre dois snapshots."""
    if antes is None or agora is None:
        return "", ""
    eventos, detalhes = [], []

    if antes["especie"] != agora["especie"]:
        eventos.append("ESPECIE_MUDOU")
        detalhes.append(f"{antes['especie']} -> {agora['especie']}")

    if antes["tipos"] != agora["tipos"]:
        eventos.append("TIPOS_MUDARAM")
        detalhes.append(f"{antes['tipos']} -> {agora['tipos']}")

    if (antes["atk"], antes["spa"], antes["spe"]) != (agora["atk"], agora["spa"], agora["spe"]):
        eventos.append("STATS_MUDARAM")
        detalhes.append(f"atk/spa/spe {antes['atk']}/{antes['spa']}/{antes['spe']}"
                        f" -> {agora['atk']}/{agora['spa']}/{agora['spe']}")

    if antes["tera"] != agora["tera"]:
        eventos.append("TERA_ATIVADA")
        detalhes.append(f"tera {antes['tera']} -> {agora['tera']} ({agora['tera_tipo']})")

    if antes["abilidade"] != agora["abilidade"]:
        eventos.append("HABILIDADE_MUDOU")
        detalhes.append(f"{antes['abilidade']} -> {agora['abilidade']}")

    return "+".join(eventos), " | ".join(detalhes)


class Observador:
    """Guarda snapshots por (batalha, lado) e escreve o CSV. Partilhado pelos modos."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._anterior = {}
        self.contagem = {
            "TIPOS_MUDARAM": 0, "STATS_MUDARAM": 0, "ESPECIE_MUDOU": 0,
            "TERA_ATIVADA": 0, "HABILIDADE_MUDOU": 0,
        }
        self.linhas = 0
        self.tera_atributo_existe = None

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CABECALHO)

    def registar(self, battle, mon, lado):
        s = _snapshot(mon)
        if s is None:
            return
        if self.tera_atributo_existe is None:
            self.tera_atributo_existe = (s["tera"] != "N/D")

        chave = (battle.battle_tag, lado)
        evento, detalhe = _comparar(self._anterior.get(chave), s)
        self._anterior[chave] = s

        for e in evento.split("+"):
            if e in self.contagem:
                self.contagem[e] += 1

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%H:%M:%S"), battle.battle_tag,
                getattr(battle, "turn", 0), lado, s["especie"], s["tipos"],
                s["tera"], s["tera_tipo"], s["atk"], s["spa"], s["spe"],
                s["abilidade"], s["item"], s["hp"], evento, detalhe,
            ])
        self.linhas += 1

        if evento:
            print(f"  [t{getattr(battle, 'turn', 0):>3}] {lado:<10} {s['especie']:<16} "
                  f"{evento}  {detalhe}", flush=True)

    def veredicto(self):
        print()
        print("=" * 78)
        print("  VEREDICTO")
        print("=" * 78)
        print(f"  Snapshots registados : {self.linhas}")
        for k, v in self.contagem.items():
            print(f"  {k:<20} : {v}")
        print()

        if self.tera_atributo_existe is False:
            print("  ATENCAO: nenhum atributo de Terastalizacao encontrado no objeto")
            print("  Pokemon (tentou terastallized, is_terastallized, terastalized).")
            print("  A versao do poke-env instalada pode nao expor a Tera. Confirmar")
            print("  antes de escrever qualquer regra que dependa disso.")
            print()

        if self.contagem["TIPOS_MUDARAM"] > 0:
            print("  BOA NOTICIA: o poke-env PROPAGA mudancas de tipagem.")
            print("  Como todas as camadas do instinto leem os tipos ao vivo (sem")
            print("  cache), o matchup e o survival score ja se adaptam sozinhos.")
        else:
            print("  NENHUMA mudanca de tipagem observada.")
            print("  Nao conclusivo por si so: pode ser que nao tenha havido Protean")
            print("  nem Tera nas batalhas observadas. Verificar na coluna Habilidade")
            print("  se o Greninja (Protean) chegou a entrar em campo. Se entrou e")
            print("  atacou com tipo diferente e os tipos NAO mudaram, e bug ativo.")
        print("=" * 78)


class InstinctObservado(InstinctBot):
    """InstinctBot com observador. NAO altera nenhuma decisao: chama o super()."""

    def definir_observador(self, obs):
        self._obs = obs

    def choose_move(self, battle):
        obs = getattr(self, "_obs", None)
        if obs is not None:
            obs.registar(battle, battle.active_pokemon, "PROPRIO")
            obs.registar(battle, battle.opponent_active_pokemon, "ADVERSARIO")
        return super().choose_move(battle)


class MaxDamageObservado(MaxDamagePlayer):
    """MaxDamage com observador. Usado no modo humano: o 'adversario' es tu."""

    def definir_observador(self, obs):
        self._obs = obs

    def choose_move(self, battle):
        obs = getattr(self, "_obs", None)
        if obs is not None:
            obs.registar(battle, battle.active_pokemon, "BOT")
            obs.registar(battle, battle.opponent_active_pokemon, "HUMANO")
        return super().choose_move(battle)


def proximo_indice():
    os.makedirs(LOGS_DIR, exist_ok=True)
    ids = []
    for caminho in glob.glob(os.path.join(LOGS_DIR, "diagnostico_mecanicas_*.csv")):
        m = re.fullmatch(r"diagnostico_mecanicas_(\d+)\.csv", os.path.basename(caminho))
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 1


async def modo_observar(obs, n_batalhas):
    bot = InstinctObservado(
        account_configuration=AccountConfiguration("DiagInstinct", None),
        server_configuration=LOCAL, battle_format=BATTLE_FORMAT,
        team=RandomTeamFromPool(TEAMS_LIST),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False, log_level=logging.CRITICAL,
    )
    bot.definir_observador(obs)

    adversario = MaxDamagePlayer(
        account_configuration=AccountConfiguration("DiagMaxDmg", None),
        server_configuration=LOCAL, battle_format=BATTLE_FORMAT,
        team=RandomTeamFromPool(TEAMS_LIST),
        max_concurrent_battles=CONCORRENCIA,
        start_timer_on_battle_start=False, log_level=logging.CRITICAL,
    )

    print(f"Modo OBSERVAR: InstinctBot vs MaxDamage, {n_batalhas} batalhas.")
    print("Procura-se sobretudo o Greninja (Protean) a entrar em campo.")
    print("-" * 78)
    inicio = time.time()
    await bot.battle_against(adversario, n_battles=n_batalhas)
    print("-" * 78)
    print(f"Concluido em {time.time() - inicio:.0f} s.")


async def modo_humano(obs, utilizador, n_batalhas):
    bot = MaxDamageObservado(
        account_configuration=AccountConfiguration("DiagMaxDmg", None),
        server_configuration=LOCAL, battle_format=BATTLE_FORMAT,
        team=RandomTeamFromPool(TEAMS_LIST),
        max_concurrent_battles=CONCORRENCIA,
        # timer LIGADO: com um humano do outro lado, sem timer a batalha pode ficar
        # pendurada indefinidamente se fechares o browser.
        start_timer_on_battle_start=True,
        log_level=logging.CRITICAL,
    )
    bot.definir_observador(obs)

    print("Modo HUMANO. O bot esta a aceitar desafios.")
    print()
    print("  1. Abre http://localhost:8000 no browser")
    print(f"  2. Entra com o nick: {utilizador}")
    print("  3. Desafia o utilizador 'DiagMaxDmg'")
    print(f"  4. Formato: {BATTLE_FORMAT}")
    print("  5. Joga e usa Tera, Mega e Z-Move a mao. Cada uso aparece abaixo.")
    print()
    print("  O que interessa e a linha 'HUMANO': e assim que o instinto ve um")
    print("  adversario em batalha real, por inferencia do protocolo.")
    print("-" * 78)

    await bot.accept_challenges(utilizador, n_batalhas)

    print("-" * 78)
    print("Desafios concluidos.")


def main():
    p = argparse.ArgumentParser(
        description="Diagnostico de propagacao de tipagem e stats (Protean/Tera/Mega).")
    p.add_argument("--modo", choices=["observar", "humano"], default="observar")
    p.add_argument("--batalhas", type=int, default=30,
                   help="numero de batalhas (observar) ou de desafios a aceitar (humano)")
    p.add_argument("--utilizador", type=str, default="",
                   help="o teu nick no servidor local (obrigatorio no modo humano)")
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    if args.modo == "humano" and not args.utilizador:
        p.error("--modo humano exige --utilizador com o teu nick do servidor local")

    import random
    random.seed(args.semente)
    np.random.seed(args.semente)

    sessao = proximo_indice()
    csv_path = os.path.join(LOGS_DIR, f"diagnostico_mecanicas_{sessao:02d}.csv")
    obs = Observador(csv_path)

    print()
    print("=" * 78)
    print("  DIAGNOSTICO DE MECANICAS — o poke-env propaga tipagem e stats?")
    print("=" * 78)
    print(f"  CSV: {csv_path}")
    print()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)

    try:
        if args.modo == "observar":
            loop.run_until_complete(modo_observar(obs, args.batalhas))
        else:
            loop.run_until_complete(modo_humano(obs, args.utilizador, args.batalhas))
    except KeyboardInterrupt:
        print("\nInterrompido. O veredicto abaixo cobre o que foi observado ate agora.")

    obs.veredicto()
    print(f"\n  Registado em: {csv_path}")


if __name__ == "__main__":
    main()
