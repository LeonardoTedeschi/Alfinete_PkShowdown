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

# Limpa logs
logging.getLogger("poke-env").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from poke_env import ServerConfiguration

# --- 2. IMPORTAÇÕES ---
try:
    from Suporte.teams import RandomTeamFromPool, TEAMS_LIST
    import Suporte.plot_graph as plot_graph
    from Suporte.rivals import MaxDamagePlayer
    from q_instinct_bot import Blue_bot as BLUE
except ImportError as e:
    print(f"[ERRO] Falha ao importar: {e}")
    sys.exit(1)

# --- CONFIGURAÇÕES ---
LOCAL_CONFIG = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")
TOTAL_BATTLES = 2000       # 2000 batalhas geralmente bastam para dominar o MaxDamage
BATCH_SIZE = 100           # Salva a cada 100
CONCURRENCY = 5            # Batalhas simultâneas

async def main():
    # 1. Preparação
    team_builder = RandomTeamFromPool(TEAMS_LIST)
    
    # Instancia o BLUE (Aprendiz)
    # Ele vai carregar o "blue_brain.pkl" se existir e continuar de onde parou
    blue_bot = BLUE(
        battle_format="gen9nationaldex",
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )

    # Instancia o MAX DAMAGE (Oponente)
    # Ele não aprende, é apenas um script fixo agressivo
    rival_bot = MaxDamagePlayer(
        battle_format="gen9nationaldex",
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )

    # Garante estrutura de logs
    if not hasattr(blue_bot, 'paths'):
        blue_bot.paths = plot_graph.setup_training_files()
    
    print(f"{'='*60}")
    print(f"      TREINO DE CHOQUE: BLUE vs MAX_DAMAGE")
    print(f"{'='*60}")
    print(f"[-] Concorrência: {CONCURRENCY}")
    print(f"[-] Meta:         {TOTAL_BATTLES} batalhas")
    print(f"[-] Objetivo:     Atingir > 90% de Win Rate")
    print(f"{'='*60}\n")

    start_time = time.time()
    battles_played = 0
    blue_wins_cumulative = 0

    try:
        while battles_played < TOTAL_BATTLES:
            # Define tamanho do lote
            current_batch_size = min(BATCH_SIZE, TOTAL_BATTLES - battles_played)
            
            # Executa batalhas
            await blue_bot.battle_against(rival_bot, n_battles=current_batch_size)
            
            battles_played += current_batch_size
            
            # Estatísticas
            current_total_wins = blue_bot.n_won_battles
            batch_wins = current_total_wins - blue_wins_cumulative
            blue_wins_cumulative = current_total_wins
            
            batch_wr = (batch_wins / current_batch_size) * 100
            total_wr = (blue_wins_cumulative / battles_played) * 100
            
            # --- SALVAMENTO ---
            # Só precisamos salvar o Blue (MaxDamage não tem cérebro para salvar)
            blue_brain_path = os.path.join(root_dir, "blue_brain.pkl")
            blue_bot.save_model(blue_brain_path)
            
            # Limpeza de RAM
            gc.collect()

            elapsed = time.time() - start_time
            print(f"[Treino {battles_played}/{TOTAL_BATTLES}] "
                  f"Lote WR: {batch_wr:5.1f}% | "
                  f"Global WR: {total_wr:5.1f}% | "
                  f"Epsilon: {blue_bot.epsilon:.4f}")

    except KeyboardInterrupt:
        print("\n\n[!] Interrompido pelo usuário.")
    finally:
        print(f"\n{'='*60}")
        print("FIM DO TREINO CONTRA MAX DAMAGE")
        
        blue_brain_path = os.path.join(root_dir, "blue_brain.pkl")
        blue_bot.save_model(blue_brain_path)
        
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