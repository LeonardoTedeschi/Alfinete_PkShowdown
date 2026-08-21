import asyncio
import logging
import os
import sys

# --- 1. CORREÇÃO DE CAMINHOS (Para importar bots de pastas diferentes) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
instinct_path = os.path.join(project_root, 'Instinto')

# Adiciona os caminhos ao Python
sys.path.append(instinct_path)
sys.path.append(current_dir)

# --- 2. IMPORTAÇÕES ---
try:
    from bot_agent import RED
    from instinct_bot import InstinctBot
    from teams import RandomTeamFromPool, TEAMS_LIST 
except ImportError as e:
    print(f"[ERRO DE IMPORTAÇÃO] {e}")
    print("Verifique se as pastas 'Q-learning' e 'Instinto' estão corretas.")
    sys.exit(1)

from poke_env import ServerConfiguration, AccountConfiguration

# Limpa o terminal de logs desnecessários
logging.getLogger("poke-env").setLevel(logging.ERROR)

# Configuração do Servidor Local
LOCAL_CONFIG = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

async def run_arena(n_battles=10):
    print(f"\n{'='*60}")
    print(f"       ARENA: RED (RL) vs INSTINCT (LÓGICA)")
    print(f"{'='*60}")
    print(f"Batalhas Programadas: {n_battles}")

    # --- CONFIGURAR RED (O Desafiante) ---
    red_username = "Red_RL_Bot"
    red_config = AccountConfiguration(red_username, None) # Senha None para local
    
    red_bot = RED(
        account_configuration=red_config,
        server_configuration=LOCAL_CONFIG,
        battle_format="gen9nationaldex",
        team=RandomTeamFromPool(TEAMS_LIST)
    )

    # Carregar Cérebro
    brain_file = os.path.join(current_dir, "red_brain.pkl")
    if os.path.exists(brain_file):
        red_bot.brain.load_model(brain_file)
        red_bot.brain.epsilon = 0.0 # Modo Competitivo
        print(f"[RED] Cérebro carregado. Epsilon: 0.0")
    else:
        print("[RED] JOGANDO ALEATORIAMENTE (Cérebro não encontrado).")

    # --- CONFIGURAR INSTINCT (O Defensor) ---
    blue_username = "Blue_Instinct"
    instinct_config = AccountConfiguration(blue_username, None) # Senha None para local
    
    instinct_bot = InstinctBot(
        account_configuration=instinct_config,
        server_configuration=LOCAL_CONFIG,
        battle_format="gen9nationaldex",
        team=RandomTeamFromPool(TEAMS_LIST)
    )
    print(f"[BLUE] InstinctBot pronto para defender.")

    print(f"\n>>> INICIANDO COMBATE...")

    # --- CORREÇÃO DO TRAVAMENTO (Lógica Assíncrona) ---
    
    # 1. Coloca o InstinctBot para "ouvir" desafios em segundo plano
    # Ele vai aceitar exatamente 'n_battles' desafios
    defensor_task = asyncio.create_task(
        instinct_bot.accept_challenges(opponent=None, n_challenges=n_battles)
    )

    # 2. Espera um pouco para garantir que o Blue logou e está pronto
    print("... Aguardando login dos bots ...")
    await asyncio.sleep(2)

    # 3. O Red começa a bombardear desafios para o nome exato do Blue
    # O await aqui garante que o script só avança quando o Red terminar suas lutas
    await red_bot.send_challenges(opponent=blue_username, n_challenges=n_battles)

    # 4. Espera o defensor terminar de processar (caso tenha atraso)
    await defensor_task

    # --- PLACAR FINAL ---
    wins = red_bot.n_won_battles
    total = n_battles
    win_rate = (wins / total) * 100

    print(f"\n{'='*60}")
    print(f"              RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"RED (IA Aprendizado):   {wins} vitórias")
    print(f"BLUE (Lógica Fixa):     {total - wins} vitórias")
    print(f"--------------------------------------")
    print(f"TAXA DE VITÓRIA DA IA:  {win_rate:.2f}%")
    
    if win_rate > 50:
        print("\n>>> SUCESSO: A IA SUPEROU O INSTINTO! 🤖🏆")
    else:
        print("\n>>> RESULTADO: A LÓGICA HUMANA AINDA VENCEU.")

if __name__ == "__main__":
    # Executa o loop assíncrono
    asyncio.run(run_arena(n_battles=10))