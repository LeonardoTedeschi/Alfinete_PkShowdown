import asyncio
import logging
import csv
import os
import sys
import time
import contextlib
import warnings
import gc  # Garbage Collector fundamental

# --- IMPORTAÇÕES DA PASTA SUPORTE ---
import Suporte.plot_graph as plot_graph

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("poke-env").setLevel(logging.ERROR)

from poke_env import ServerConfiguration
from poke_env.player import Player
from brain import RLBrain
from Suporte.rivals import MaxDamagePlayer
from Suporte.teams import RandomTeamFromPool, TEAMS_LIST

# Configuração Local
LOCAL_CONFIG = ServerConfiguration("ws://localhost:8000/showdown/websocket", "http://localhost:8000/")
BLOCK_SIZE = 200 

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try: yield
        finally: sys.stdout = old_stdout

class RED(Player):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.paths = plot_graph.setup_training_files()
        
        self.brain = RLBrain(alpha=0.1, epsilon=1.0, min_epsilon=0.05, decay=0.999)
        
        with suppress_stdout():
            if self.brain.load_model(self.paths["model"]):
                sys.stderr.write(f"\n[BOT] CÉREBRO CARREGADO ({len(self.brain.q_table)} estados)\n")
            else:
                sys.stderr.write(f"\n[BOT] CÉREBRO NOVO (Iniciando do zero)\n")
                
        self._init_csv()
        self.battle_history = {}
        
        self.total_completed_battles = 0
        self.total_wins = 0
        self.total_reward_sum = 0.0
        
        self.block_wins = 0
        self.block_rewards_sum = 0.0

    def _init_csv(self):
        try:
            with open(self.paths["csv"], "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Batalhas", "WinRate_Bloco", "Epsilon", "Avg_Reward_Bloco"])
        except: pass

    def save_brain_silently(self):
        with suppress_stdout():
            self.brain.save_model(self.paths["model"])

    def check_finished_battles(self):
        # Snapshot seguro para deletar chaves durante iteração
        battles_snapshot = list(self.battles.items())
        
        for battle_id, battle in battles_snapshot:
            if battle.finished:
                self._process_end_battle(battle)
                # Limpeza de RAM imediata
                if battle_id in self.battles:
                    del self.battles[battle_id]

    def _process_end_battle(self, battle):
        try:
            reward = 0
            if battle.battle_tag in self.battle_history:
                last_turn = self.battle_history[battle.battle_tag]
                reward = self.brain.calculate_reward(battle, last_turn)
                self.brain.update_knowledge(
                    prev=last_turn['state'],
                    act=last_turn['action_idx'],
                    reward=reward,
                    curr=last_turn['state']
                )
                del self.battle_history[battle.battle_tag]

            self.total_completed_battles += 1
            self.total_reward_sum += reward 
            if battle.won:
                self.total_wins += 1
                self.block_wins += 1
            self.block_rewards_sum += reward

            # Salva a cada 100 batalhas, mas SEM PRINT para não travar o I/O
            if self.total_completed_battles % BLOCK_SIZE == 0:
                block_wr = (self.block_wins / BLOCK_SIZE) * 100
                avg_reward = self.block_rewards_sum / BLOCK_SIZE
                
                try:
                    with open(self.paths["csv"], "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.total_completed_battles, block_wr, self.brain.epsilon, avg_reward])
                except: pass
                
                self.block_wins = 0
                self.block_rewards_sum = 0.0
                self.save_brain_silently()
                
                # Limpa RAM silenciosamente
                gc.collect()
                
        except Exception:
            pass # Silencia erros de log para não parar o treino

    def choose_move(self, battle):
        self.check_finished_battles()
        
        try:
            current_state = self.brain.get_state_key(battle)
            
            # Passa 'battle' para o cérebro checar imunidades se necessário
            chosen_move, chosen_idx, dec_type = self.brain.choose_action(
                state=current_state,
                moves=battle.available_moves,
                switches=battle.available_switches,
                battle=battle 
            )

            if chosen_move is None:
                return self.choose_random_move(battle)

            self.battle_history[battle.battle_tag] = {
                'state': current_state,
                'action_idx': chosen_idx,
                'move_obj': chosen_move,
                'was_switch': chosen_move in battle.available_switches,
                'my_hp': battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0,
                'opp_hp': battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0,
                'threat': current_state[0],
                'offense': current_state[1],
                'opp_status': str(battle.opponent_active_pokemon.status) if battle.opponent_active_pokemon else "None",
                'opp_hazards': list(battle.opponent_side_conditions.keys()), 
                'my_boosts': battle.active_pokemon.boosts.copy() if battle.active_pokemon else {}
            }

            do_mega = battle.can_mega_evolve
            do_tera = False
            if hasattr(battle, "can_tera") and battle.can_tera: do_tera = True

            return self.create_order(chosen_move, mega=do_mega, terastallize=do_tera)
            
        except Exception:
            return self.choose_random_move(battle)

async def main():
    n_battles = 5000    
    CONCURRENCY = 5
    
    team_builder = RandomTeamFromPool(TEAMS_LIST)
    
    bot = RED (
        battle_format="gen9nationaldex", 
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )
    
    rival = MaxDamagePlayer(
        battle_format="gen9nationaldex", 
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )
    
    print(f"{'='*40}")
    print(f" SESSÃO: {bot.paths['id']}")
    print(f" LOG:    {bot.paths['csv']}")
    print(f" META:   {n_battles} batalhas (Modo Silencioso)")
    print(f"{'='*40}")
    print("Treinando... (Aguarde o término, o terminal ficará sem output)")
    print("Pressione Ctrl+C APENAS para abortar.")

    start_time = time.time()

    try:
        await bot.battle_against(rival, n_battles=n_battles)
    except KeyboardInterrupt:
        print("\n\n[!] Interrompido pelo usuário. Salvando...")
    finally:
        # Garante que tudo foi processado e salvo
        bot.check_finished_battles()
        bot.save_brain_silently()
        
        end_time = time.time()
        
        try:
            plot_graph.generate_graph(
                bot.paths['csv'], 
                bot.paths['graph'], 
                title_suffix=str(bot.paths['id'])
            )
        except: pass
        
        valid = bot.total_completed_battles
        wins = bot.total_wins
        win_rate = (wins / valid * 100) if valid > 0 else 0.0
        
        print(f"\n{'='*40}")
        print(f"           RESULTADO FINAL")
        print(f"{'='*40}")
        print(f"Tempo:       {end_time - start_time:.2f}s")
        print(f"Estados:     {len(bot.brain.q_table)}")
        print(f"Batalhas:    {valid}")
        print(f"Vitórias:    {wins}")
        print(f"Win Rate:    {win_rate:.2f}%")
        print(f"Score Total: {bot.total_reward_sum:.1f}")
        print(f"Epsilon:     {bot.brain.epsilon:.5f}")
        print(f"{'='*40}")
        print("Modelo salvo com sucesso.")
        os._exit(0)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())