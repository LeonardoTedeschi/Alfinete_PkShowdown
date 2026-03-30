import asyncio
import logging
import os
import sys
import time
import random
import gc

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
instinct_path = os.path.join(project_root, 'Instinto')
sys.path.append(instinct_path)
sys.path.append(current_dir)

# --- 2. IMPORTAÇÕES ---
try:
    from instinct_bot import InstinctBot
    from bot_agent import RED 
    from teams import RandomTeamFromPool, TEAMS_LIST 
except ImportError as e:
    print(f"[ERRO] {e}")
    sys.exit(1)

from poke_env import ServerConfiguration, AccountConfiguration

# Silencia logs chatos
logging.getLogger("poke-env").setLevel(logging.ERROR)
LOCAL_CONFIG = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

# CONFIGURAÇÕES
MAX_BATALHAS = 1000
TIMEOUT_BATALHA = 180  # 3 minutos máximos por luta. Se passar, mata.

async def main():
    sess_id = random.randint(100, 999)
    red_name = f"Red_Train_{sess_id}"
    blue_name = f"Sensei_{sess_id}"

    print(f"\n[SISTEMA] Iniciando Treino com Watchdog (Anti-Travamento)")
    print(f"[CONFIG] Timeout por luta: {TIMEOUT_BATALHA}s")
    
    # CRIANDO OS BOTS COM A TRAVA DE CONCORRÊNCIA
    red_bot = RED(
        account_configuration=AccountConfiguration(red_name, None),
        server_configuration=LOCAL_CONFIG,
        battle_format="gen9nationaldex",
        team=RandomTeamFromPool(TEAMS_LIST),
        max_concurrent_battles=1
    )
    red_bot.brain.epsilon = 0.15

    rival = InstinctBot(
        account_configuration=AccountConfiguration(blue_name, None),
        server_configuration=LOCAL_CONFIG,
        battle_format="gen9nationaldex",
        team=RandomTeamFromPool(TEAMS_LIST),
        max_concurrent_battles=1
    )

    battles_done = 0

    while battles_done < MAX_BATALHAS:
        try:
            # LIMPEZA DE ESTADO FANTASMA
            # Se o bot acha que está em batalha mas ela travou, limpamos aqui.
            if red_bot.battles:
                for b_id in list(red_bot.battles.keys()):
                    if red_bot.battles[b_id].finished:
                        del red_bot.battles[b_id]

            # 1. Defensor aceita
            defensor_task = asyncio.create_task(
                rival.accept_challenges(opponent=red_name, n_challenges=1)
            )
            await asyncio.sleep(0.5)

            # 2. Atacante envia (COM WATCHDOG/TIMEOUT)
            # Aqui está a mágica: asyncio.wait_for vai contar o tempo.
            # Se passar de 180s, ele lança um erro e destrava o script.
            await asyncio.wait_for(
                red_bot.send_challenges(opponent=blue_name, n_challenges=1),
                timeout=TIMEOUT_BATALHA
            )
            
            await defensor_task
            battles_done += 1

            # SALVAMENTO
            red_bot.save_brain_silently()
            
            sys.stdout.write(f"\r[DOJO] Batalhas: {battles_done} | WinRate: {(red_bot.n_won_battles/battles_done)*100:.1f}% | Estados: {len(red_bot.brain.q_table)}")
            sys.stdout.flush()

            await asyncio.sleep(0.2)
            gc.collect()

        except asyncio.TimeoutError:
            # OCORREU O TRAVAMENTO DO TURNO 44
            print(f"\n[ALERTA] Batalha travou (Timeout > {TIMEOUT_BATALHA}s)!")
            print("[AÇÃO] Ignorando esta luta e reiniciando bots...")
            
            # Força o fechamento para destravar os sockets
            # Na próxima volta do loop, ele tenta de novo limpo
            continue
            
        except Exception as e:
            print(f"\n[ERRO] {e}. Reiniciando em 3s...")
            await asyncio.sleep(3)
            continue

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())