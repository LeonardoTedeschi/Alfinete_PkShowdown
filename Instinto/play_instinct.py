import asyncio
import logging
import sys
import os

# Configuração de caminhos para mapear a estrutura do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
if current_dir not in sys.path: sys.path.append(current_dir)
if repo_root not in sys.path: sys.path.append(repo_root)

# Importações do Poke-env
from poke_env import ServerConfiguration, AccountConfiguration

# Importação do Agente Híbrido (RL + Instinto)
try:
    from blue_agent import BLUE
    from Suporte.teams import RandomTeamFromPool, TEAMS_LIST
except ImportError as e:
    print(f"[ERRO] Falha de importação: {e}")
    print("Certifique-se de que a pasta 'Suporte' e 'blue_agent.py' estão acessíveis.")
    sys.exit(1)

# Configuração para ignorar logs excessivos
logging.getLogger("poke-env").setLevel(logging.ERROR)

# Configuração do Servidor Local
LOCAL_CONFIG = ServerConfiguration("ws://127.0.0.1:8000/showdown/websocket", "http://127.0.0.1:8000/")

async def main():
    print("\n=================================================")
    print("     D O J O   -   HUMANO VS AGENTE BLUE (V6)")
    print("=================================================")

    # 1. Configurar Identidade
    bot_username = "BlueBot"
    bot_config = AccountConfiguration(bot_username, None)

    # 2. Criar o Bot
    print(f">>> Carregando pool com {len(TEAMS_LIST)} times disponíveis.")
    
    bot = BLUE(
        account_configuration=bot_config,
        server_configuration=LOCAL_CONFIG,
        battle_format="gen9nationaldex",
        team=RandomTeamFromPool(TEAMS_LIST)
    )

    print(">>> Agente BLUE Inicializado.")
    print(">>> Módulos carregados: Matriz 4D (Instinto) + Q-Table (Cérebro).")

    # 3. Instruções
    print("\n--- COMO LUTAR ---")
    print("1. Certifique-se que o servidor Showdown local está rodando.")
    print(f"2. Procure o usuário '{bot_username}' no Lobby (http://127.0.0.1:8000).")
    print("3. Desafie para o formato: [Gen 9] National Dex")
    print("4. O bot aprenderá e salvará a Q-Table ao fim de cada partida.")
    print("\n>>> AGUARDANDO DESAFIO... (Ctrl+C para sair)")

    # 4. Loop de Batalha Manual
    while True:
        try:
            await bot.accept_challenges(opponent=None, n_challenges=1)
            print("\n>>> Batalha concluída!")
            
            # Processa o resultado final (vitória/derrota) e salva a matriz na hora
            bot.check_finished_battles()
            bot.save_brain_silently()
            print(">>> Progresso do Cérebro (Q-Table) salvo. Preparando próxima luta...")
            
        except KeyboardInterrupt:
            print("\n>>> Encerrando o bot e garantindo salvamento do progresso...")
            bot.check_finished_battles()
            bot.save_brain_silently()
            break
        except Exception as e:
            print(f"Erro durante a conexão ou batalha: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass