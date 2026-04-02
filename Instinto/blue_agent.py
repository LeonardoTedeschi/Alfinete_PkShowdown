import asyncio
import sys
import os
import time
import logging
import csv
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
suporte_treinamento_dir = os.path.join(repo_root, 'Suporte_Treinamento')

if current_dir not in sys.path: sys.path.append(current_dir)
if suporte_treinamento_dir not in sys.path: sys.path.append(suporte_treinamento_dir)

logging.basicConfig(level=logging.ERROR)
logging.getLogger("poke-env").setLevel(logging.ERROR) 
warnings.filterwarnings("ignore")

from poke_env import ServerConfiguration
from poke_env.player import Player

try:
    from instinct_core import InstinctCore
    from blue_brain import BlueBrain
    from Suporte.teams import RandomTeamFromPool, TEAMS_LIST
    import Suporte.plot_graph as plot_graph
    from Suporte.rivals import MaxDamagePlayer
except ImportError as e:
    print(f"[ERRO] Falha de importação: {e}")
    sys.exit(1)

LOCAL_CONFIG = ServerConfiguration("ws://127.0.0.1:8000/showdown/websocket", "http://127.0.0.1:8000/")
BLOCK_SIZE = 200

class BLUE(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = InstinctCore()
        self.brain = BlueBrain()
        self.paths = plot_graph.setup_training_files()
        
        self.battle_history = {} 
        self.total_completed_battles = 0
        self.total_wins = 0
        self.total_reward_sum = 0.0
        self.block_wins = 0
        self._init_csv()

    def _init_csv(self):
        try:
            with open(self.paths['csv'], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Batalhas", "WinRate_Bloco", "Epsilon", "Reward"])
        except: pass

    def save_brain_silently(self):
        self.brain.save_model("blue_brain.pkl")

    def check_finished_battles(self):
        battles_snapshot = list(self.battles.items())
        for b_id, battle in battles_snapshot:
            if battle.finished:
                self._process_end_battle(battle)
                if b_id in self.battles:
                    del self.battles[b_id]

    # --- FLUXO FIM DE BATALHA: Agente -> Cérebro ---
    def _process_end_battle(self, battle):
        self.total_completed_battles += 1
        if battle.won:
            self.total_wins += 1
            self.block_wins += 1

        try:
            history = self.battle_history.get(battle.battle_tag, {})
            if history:
                reward = self.brain.calculate_reward(battle, history)
                self.total_reward_sum += reward
                
                last_state = history.get('state')
                last_action = history.get('last_action')
                
                if last_state and last_action:
                    current_state = self.core.get_state(battle) 
                    self.brain.update_feedback(current_state, last_state, last_action, reward)
                
                del self.battle_history[battle.battle_tag]
        except Exception:
            pass

        try:
            if self.total_completed_battles % BLOCK_SIZE == 0:
                block_wr = (self.block_wins / BLOCK_SIZE) * 100
                try:
                    with open(self.paths["csv"], "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.total_completed_battles, block_wr, self.brain.epsilon, self.total_reward_sum])
                except: pass
                
                print(f"[Progresso] {self.total_completed_battles} Batalhas | Win Rate (Bloco): {block_wr:.1f}% | Estados: {len(self.brain.q_table)}")
                
                self.block_wins = 0
                self.save_brain_silently()
        except Exception:
            pass

    # --- FLUXO INÍCIO DE BATALHA: Agente -> Instinto -> Agente ---
    def teampreview(self, battle):
        try:
            return self.core.get_best_lead(battle)
        except Exception:
            return "/team 123456"

    # --- FLUXO TURNOS ---
    def choose_move(self, battle):
        self.check_finished_battles()
        
        try:
            history = self.battle_history.get(battle.battle_tag, {})

            # 1. TROCA FORÇADA INTELIGENTE (Usa o histórico para prever quem entra)
            switch_forced = False
            if isinstance(battle.force_switch, list): switch_forced = any(battle.force_switch)
            else: switch_forced = bool(battle.force_switch)

            if switch_forced or (battle.active_pokemon and battle.active_pokemon.fainted):
                best_switch = self.core.get_best_switch(battle, history)
                if best_switch: return self.create_order(best_switch)
                return self.choose_random_move(battle)

            # 2. FEEDBACK PARA O Q-LEARNING (Agente -> Cérebro)
            current_state = self.core.get_state(battle)
            
            if history:
                reward = self.brain.calculate_reward(battle, history)
                self.total_reward_sum += reward
                last_state = history.get('state')
                last_action = history.get('last_action')
                if last_state and last_action:
                    self.brain.update_feedback(current_state, last_state, last_action, reward)

            # 3. TOMADA DE DECISÃO (Agente -> Instinto -> Cérebro)
            instinct_intent = self.core.get_intent(battle)
            final_decision = self.brain.decide_action(current_state, instinct_intent)

            opp_species = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else None

            self.battle_history[battle.battle_tag] = {
                'state': current_state,
                'last_action': final_decision,
                'my_fainted': len([m for m in battle.team.values() if m.fainted]),
                'opp_fainted': len([m for m in battle.opponent_team.values() if m.fainted]),
                'last_opp_species': opp_species 
            }

            # 4. EXECUÇÃO RÍGIDA (Anti-Random)
            # A lista garante que ele tenta a decisão do Cérebro -> Instinto -> Dano Bruto -> Troca Segura
            execution_list = [final_decision, instinct_intent, "ATTACK", "SWITCH"]
            best_object = self.core.get_best_execution_object(execution_list, battle)

            if best_object:
                return self.create_order(best_object)
            else:
                # O bot só chega aqui se TODAS as opções do cérebro/instinto falharem
                # e ele não tiver nenhum ataque válido ou Pokémon vivo para trocar.
                print(f"[ALERTA DADOS] Q-Table ignorada! Fallback Random forçado no estado: {current_state}")
                return self.choose_random_move(battle)

        except Exception as e:
            # Em vez de floodar a tela, mostra o erro e avisa a impureza
            print(f"[ERRO NO TURNO - DADO IMPURO] {e}")
            return self.choose_random_move(battle)


# =============================================================================
# BLOCO DE EXECUÇÃO PADRÃO COM PROTEÇÃO DE REDE (WATCHDOG)
# =============================================================================
async def main():
    n_battles = 10000
    CONCURRENCY = 5
    
    team_builder = RandomTeamFromPool(TEAMS_LIST)
    
    bot = BLUE(
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
    print(f" META:   {n_battles} batalhas")
    print(f"{'='*40}")
    print(f"Treinando em blocos de {BLOCK_SIZE}... Pressione Ctrl+C APENAS para abortar.")

    start_time = time.time()

    try:
        # Loop seguro: Só encerra quando as 5000 batalhas forem reais e concluídas
        while bot.total_completed_battles < n_battles:
            faltam = n_battles - bot.total_completed_battles
            current_batch = min(BLOCK_SIZE, faltam)
            
            try:
                # WATCHDOG: Batalhas locais duram segundos. 
                # Se o lote demorar 5 minutos (300s), o servidor perdeu a conexão.
                await asyncio.wait_for(
                    bot.battle_against(rival, n_battles=current_batch),
                    timeout=300
                )
            except asyncio.TimeoutError:
                print("\n[WATCHDOG] O servidor Showdown engoliu um pacote (Ghost Battle). Destravando rede...")
                bot.battles.clear()
                rival.battles.clear()
                await asyncio.sleep(2) # Pausa para o servidor Node.js respirar
            except Exception:
                pass
                
    except KeyboardInterrupt:
        print("\n\n[!] Interrompido pelo usuário. Salvando...")
    finally:
        bot.check_finished_battles()
        bot.save_brain_silently()
        
        end_time = time.time()
        
        try:
            plot_graph.generate_graph(bot.paths['csv'], bot.paths['graph'], title_suffix=str(bot.paths['id']))
        except: pass
        
        valid = bot.total_completed_battles
        wins = bot.total_wins
        win_rate = (wins / valid * 100) if valid > 0 else 0.0
        
        print(f"\n{'='*40}")
        print(f"           RESULTADO FINAL")
        print(f"{'='*40}")
        print(f"Tempo:       {(end_time - start_time) / 60:.1f} minutos")
        print(f"Estados:     {len(bot.brain.q_table)}")
        print(f"Batalhas:    {valid}")
        print(f"Vitórias:    {wins}")
        print(f"Win Rate:    {win_rate:.2f}%")
        os._exit(0)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())