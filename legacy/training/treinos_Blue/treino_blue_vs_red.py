import asyncio
import sys
import os
import time
import logging
import gc
import warnings

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Limpa a saída do console
logging.getLogger("poke-env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from poke_env import ServerConfiguration

# --- 2. IMPORTAÇÕES ---
try:
    from Suporte.teams import RandomTeamFromPool, TEAMS_LIST
    import Suporte.plot_graph as plot_graph
    from bot_agent import RED
    from q_instinct_bot import Blue_bot as BLUE
except ImportError as e:
    print(f"[ERRO] Falha ao importar os bots: {e}")
    sys.exit(1)

# --- CONFIGURAÇÕES DE TREINO ---
LOCAL_CONFIG = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")
TOTAL_BATTLES = 5000      # Meta total
BATCH_SIZE = 100          # Salva a cada 100 batalhas
CONCURRENCY = 5           # Batalhas simultâneas

async def main():
    # 1. Preparação de Times
    team_builder = RandomTeamFromPool(TEAMS_LIST)
    
    # --- INSTANCIA O BLUE (Seu Bot Novo) ---
    blue_bot = BLUE(
        battle_format="gen9nationaldex",
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )

    # --- INSTANCIA O RED (Oponente) ---
    red_bot = RED(
        battle_format="gen9nationaldex",
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )

    # Configura diretórios de log se não existirem
    if not hasattr(blue_bot, 'paths'):
        blue_bot.paths = plot_graph.setup_training_files()
    
    print(f"{'='*60}")
    print(f"      MESTRE DE TREINO: BLUE vs RED")
    print(f"{'='*60}")
    print(f"[-] Concorrência: {CONCURRENCY} threads")
    print(f"[-] Meta:         {TOTAL_BATTLES} batalhas")
    print(f"[-] Log:          {blue_bot.paths['csv']}")
    print(f"{'='*60}\n")

    start_time = time.time()
    battles_played = 0
    blue_wins_cumulative = 0

    try:
        while battles_played < TOTAL_BATTLES:
            # Define o tamanho do lote atual
            current_batch_size = min(BATCH_SIZE, TOTAL_BATTLES - battles_played)
            
            # Executa as batalhas em paralelo
            await blue_bot.battle_against(red_bot, n_battles=current_batch_size)
            
            battles_played += current_batch_size
            
            # Estatísticas
            current_total_wins = blue_bot.n_won_battles
            batch_wins = current_total_wins - blue_wins_cumulative
            blue_wins_cumulative = current_total_wins
            
            batch_wr = (batch_wins / current_batch_size) * 100
            total_wr = (blue_wins_cumulative / battles_played) * 100
            
            # --- SALVAMENTO ---
            # Salva a Q-Table do Blue
            blue_brain_path = os.path.join(root_dir, "blue_brain.pkl")
            blue_bot.save_model(blue_brain_path)
            
            # Salva o Q-Learning do Red (importante para ele continuar aprendendo também)
            red_bot.save_brain_silently()
            
            # Limpa memória RAM
            gc.collect()

            # Status no Terminal
            elapsed = time.time() - start_time
            print(f"[Treino {battles_played}/{TOTAL_BATTLES}] "
                  f"Lote WR: {batch_wr:5.1f}% | "
                  f"Global WR: {total_wr:5.1f}% | "
                  f"Epsilon: {blue_bot.epsilon:.4f}")

    except KeyboardInterrupt:
        print("\n\n[!] Parando...")
    finally:
        print(f"\n{'='*60}")
        print("FIM DO TREINO")
        
        # Salvamento final garantido
        blue_brain_path = os.path.join(root_dir, "blue_brain.pkl")
        blue_bot.save_model(blue_brain_path)
        red_bot.save_brain_silently()
        
        try:
            plot_graph.generate_graph(blue_bot.paths['csv'], blue_bot.paths['graph'], str(blue_bot.paths['id']))
            print(f"[-] Gráfico salvo: {blue_bot.paths['graph']}")
        except: pass

        print(f"[-] Win Rate Final: {(blue_bot.n_won_battles/battles_played*100):.2f}%")
        os._exit(0)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())