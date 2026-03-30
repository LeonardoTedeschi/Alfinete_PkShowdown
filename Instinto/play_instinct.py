import asyncio
import logging
import sys
import os

# Importações do Poke-env
from poke_env import ServerConfiguration, AccountConfiguration

# Importação do Nosso Bot de Instinto
from instinct_bot import InstinctBot

# Importação do Gerenciador de Times (Do seu arquivo teams.py)
from teams import RandomTeamFromPool, TEAMS_LIST

# Configuração para ignorar logs excessivos
logging.getLogger("poke-env").setLevel(logging.ERROR)

# Configuração do Servidor Local
LOCAL_CONFIG = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

async def main():
    print("\n=================================================")
    print("     D O J O   -   HUMANO VS INSTINCT (POOL)")
    print("=================================================")

    # 1. Configurar Identidade
    bot_username = "InstinctBot"
    bot_config = AccountConfiguration(bot_username, None)

    # 2. Criar o Bot usando o Pool de Times
    # O RandomTeamFromPool vai escolher um time aleatório a cada batalha
    print(f">>> Carregando pool com {len(TEAMS_LIST)} times disponíveis.")
    
    bot = InstinctBot(
        account_configuration=bot_config,
        server_configuration=LOCAL_CONFIG,
        battle_format="gen9nationaldex",
        team=RandomTeamFromPool(TEAMS_LIST) # <--- AQUI ESTÁ A MUDANÇA
    )

    print(">>> Bot Inicializado com Sucesso.")
    print(">>> Lógica de Instinto: ATIVADA.")

    # 3. Instruções
    print("\n--- COMO LUTAR ---")
    print("1. Certifique-se que o servidor local está rodando.")
    print(f"2. Procure o usuário '{bot_username}' no Lobby (http://localhost:8000).")
    print("3. Desafie para o formato: [Gen 9] National Dex")
    print("4. OBS: O Bot trocará de time a cada batalha aleatoriamente.")
    print("\n>>> O BOT ESTÁ ESPERANDO SEU DESAFIO... (Ctrl+C para sair)")

    # 4. Loop de Espera
    while True:
        try:
            # Aceita 1 desafio, joga, e volta para o loop para trocar de time na próxima
            await bot.accept_challenges(opponent=None, n_challenges=1)
            print("\n>>> Batalha concluída! Preparando novo time para a próxima...")
        except KeyboardInterrupt:
            print("\n>>> Encerrando o Bot...")
            break
        except Exception as e:
            print(f"Erro na conexão ou batalha: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass