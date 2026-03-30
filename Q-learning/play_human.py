import asyncio
import logging
import sys
import os

# Importações do Poke-env
from poke_env import ServerConfiguration, AccountConfiguration
from poke_env.player import Player

# Importações do seu projeto
from teams import RandomTeamFromPool, TEAMS_LIST
from bot_agent import RED

# Configuração para ignorar logs chatos
logging.getLogger("poke-env").setLevel(logging.ERROR)

# Configuração do Servidor Local
LOCAL_CONFIG = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")

async def main():
    print("\n=================================================")
    print("       D O J O   -   HUMANO VS RED (IA)")
    print("=================================================")

    # 1. Configurar Identidade
    # CORREÇÃO CRÍTICA: Senha = None.
    # Isso diz ao servidor: "Sou um convidado/usuário local, não verifique senha."
    bot_username = "RedBot"
    bot_config = AccountConfiguration(bot_username, None)

    # 2. Criar o Bot
    bot = RED(
        account_configuration=bot_config,
        server_configuration=LOCAL_CONFIG,
        battle_format="gen9nationaldex",
        team=RandomTeamFromPool(TEAMS_LIST)
    )

    # 3. Carregar o Cérebro
    if os.path.exists("red_brain.pkl"):
        bot.brain.load_model("red_brain.pkl")
        print(f">>> Cérebro Carregado: {len(bot.brain.q_table)} estados.")
        
        # MODO SÉRIO: Epsilon 0.0 para ele usar todo o conhecimento
        bot.brain.epsilon = 0.0 
        print(">>> MODO COMPETITIVO: Epsilon 0.0 (Sem chutes, só técnica).")
    else:
        print(">>> ERRO CRÍTICO: 'red_brain.pkl' não encontrado!")
        return

    # 4. Instruções
    print("\n--- COMO LUTAR ---")
    print("1. Abra seu navegador em: http://localhost:8000")
    print(f"2. Procure o usuário '{bot_username}' no Lobby.")
    print("   (Se não aparecer, clique em 'Find a User' e digite RedBot)")
    print("3. Clique em 'Challenge'.")
    print("4. Formato: [Gen 9] National Dex")
    print("5. Time: Escolha seu time favorito.")
    print("\n>>> O BOT ESTÁ ESPERANDO SEU DESAFIO... (Ctrl+C para sair)")

    # 5. Loop de Espera
    while True:
        try:
            # Aceita 1 desafio por vez
            await bot.accept_challenges(opponent=None, n_challenges=1)
            print("\n>>> Batalha concluída! Esperando próximo desafio...")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erro na conexão: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())