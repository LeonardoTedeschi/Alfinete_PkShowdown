import asyncio
import sys
import os
import time
import logging
import csv
import warnings
import traceback

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
suporte_treinamento_dir = os.path.join(repo_root, 'Suporte_Treinamento')

# ADICIONE ESTA LINHA: Mapeia a pasta exata onde o bot antigo está
pasta_antiga = os.path.join(current_dir, 'Vesão antiga') 

if current_dir not in sys.path: sys.path.append(current_dir)
if suporte_treinamento_dir not in sys.path: sys.path.append(suporte_treinamento_dir)

# ADICIONE ESTA LINHA: Obriga o Python a ler o que tem dentro da pasta
if pasta_antiga not in sys.path: sys.path.append(pasta_antiga)

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
    
    # === ARSENAL DE RIVAIS ===
    from Suporte.rivals import MaxDamagePlayer 
    from instinct_bot import InstinctBot 
    
except ImportError as e:
    print(f"[ERRO] Falha de importação: {e}")
    sys.exit(1)

LOCAL_CONFIG = ServerConfiguration("ws://127.0.0.1:8000/showdown/websocket", "http://127.0.0.1:8000/")
BLOCK_SIZE = 500

class BLUE(Player):
    def __init__(self, *args, **kwargs):
        # 1. Cria todas as variáveis de estado PRIMEIRO para evitar Race Conditions
        self.core = InstinctCore()
        self.brain = BlueBrain()
        self.paths = plot_graph.setup_training_files()
        
        self.battle_history = {} 
        self.total_completed_battles = 0
        self.total_wins = 0
        self.total_reward_sum = 0.0
        self.block_wins = 0
        self._init_csv()
        
        # 2. Conecta ao servidor e se declara pronto POR ÚLTIMO
        super().__init__(*args, **kwargs)

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

    def teampreview(self, battle):
        try:
            return self.core.get_best_lead(battle)
        except Exception:
            return "/team 123456"

    def choose_move(self, battle):
        self.check_finished_battles()
        
        try:
            history = self.battle_history.get(battle.battle_tag, {})

            # Gerenciamento de Troca Forçada
            switch_forced = False
            if isinstance(battle.force_switch, list): switch_forced = any(battle.force_switch)
            else: switch_forced = bool(battle.force_switch)

            if switch_forced or (battle.active_pokemon and battle.active_pokemon.fainted):
                best_switch = self.core.get_best_switch(battle, history)
                if best_switch: return self.create_order(best_switch)
                return self.choose_random_move(battle)

            current_state = self.core.get_state(battle)
            
            # Atualização e Feedback da Q-Table no turno a turno
            if history:
                reward = self.brain.calculate_reward(battle, history)
                self.total_reward_sum += reward
                last_state = history.get('state')
                last_action = history.get('last_action')
                if last_state and last_action:
                    self.brain.update_feedback(current_state, last_state, last_action, reward)

            current_state = self.core.get_state(battle)
            instinct_intent = self.core.get_instinct_intent(battle)
            
            # Action Masking restringe as opções visíveis para o Cérebro
            available_actions = self.core.get_available_actions(battle)
            final_decision = self.brain.decide_action(current_state, instinct_intent, available_actions)

            opp_species = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else None

            active = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            
            self.battle_history[battle.battle_tag] = {
                'state': current_state,
                'last_action': final_decision, 
                
                # NOVO: Tira foto da vida e status de todos no banco para avaliar a troca
                'team_hp': {m.species: m.current_hp_fraction for m in battle.team.values()},
                'team_status': {m.species: str(m.status) if m.status else "CLEAN" for m in battle.team.values() if hasattr(m, 'status')},
                
                'my_hp_bucket': self.core.get_hp_bucket(battle.active_pokemon),
                'opp_hp_bucket': self.core.get_opp_hp_bucket(battle.opponent_active_pokemon),
                'my_status': self.core.get_status_state(battle.active_pokemon),
                'opp_status': self.core.get_status_state(battle.opponent_active_pokemon),
                'my_boosts': battle.active_pokemon.boosts.copy() if battle.active_pokemon else {},
                'my_hazards': self.core.get_hazard_state(battle.side_conditions),
                'opp_species': opp_species,
                'my_fainted': len([m for m in battle.team.values() if m.fainted]),
                'opp_fainted': len([m for m in battle.opponent_team.values() if m.fainted])
            }


            # =================================================================
            # EXECUÇÃO MECÂNICA E TRADUÇÃO (Instinto -> Poke-env)
            # =================================================================

            execution_list = [final_decision, instinct_intent, "ATTACK", "SWITCH"]
            
            # Recebe a tupla (Objeto_do_Jogo, Flag_da_Mecânica)
            execution_result = self.core.get_best_execution_object(execution_list, battle, history)
            
            action_obj = execution_result[0]
            mechanic_flag = execution_result[1]

            if action_obj:
                # Checagem direta de objeto (Sem importar a classe Move)
                if action_obj in battle.available_moves:
                    is_tera = (mechanic_flag == "MEC") and battle.can_tera
                    is_mega = (mechanic_flag == "MEC") and battle.can_mega_evolve
                    is_dyna = (mechanic_flag == "MEC") and battle.can_dynamax
                    is_z = (mechanic_flag == "MEC") and battle.can_z_move

                    # --- INTERCEPTADOR E CORREÇÃO DE Z-MOVES ---
                    if is_z and battle.active_pokemon.item:
                        item_str = str(battle.active_pokemon.item).lower()
                        if item_str.endswith('z'):
                            crystal_map = {
                                'normaliumz': 'normal', 'firiumz': 'fire', 'wateriumz': 'water',
                                'electriumz': 'electric', 'grassiumz': 'grass', 'iciumz': 'ice',
                                'fightiniumz': 'fighting', 'poisoniumz': 'poison', 'groundiumz': 'ground',
                                'flyiniumz': 'flying', 'psychiumz': 'psychic', 'buginiumz': 'bug',
                                'rockiumz': 'rock', 'ghostiumz': 'ghost', 'dragoniumz': 'dragon',
                                'darkiniumz': 'dark', 'steeliumz': 'steel', 'fairiumz': 'fairy'
                            }
                            signature_map = {
                                'aloraichiumz': 'thunderbolt', 'decidiumz': 'spiritshackle', 
                                'eeviumz': 'lastresort', 'inciniumz': 'darkestlariat', 
                                'kommoniumz': 'clangingscales', 'lunaliumz': 'moongeistbeam', 
                                'lycaniumz': 'stoneedge', 'marshadiumz': 'spectralthief', 
                                'mewniumz': 'psychic', 'mimikiumz': 'playrough', 
                                'pikaniumz': 'volttackle', 'pikashuniumz': 'thunderbolt', 
                                'primariumz': 'sparklingaria', 'snorliumz': 'gigaimpact', 
                                'solganiumz': 'sunsteelstrike', 'tapuniumz': 'naturesmadness', 
                                'ultranecroziumz': 'photongeyser'
                            }
                            
                            req_type = crystal_map.get(item_str)
                            req_move_id = signature_map.get(item_str)
                            valid_z = None
                            
                            if req_type:
                                if str(action_obj.type).split('.')[-1].lower() != req_type:
                                    valid_z = next((m for m in battle.available_moves if str(m.type).split('.')[-1].lower() == req_type), None)
                            elif req_move_id:
                                if action_obj.id != req_move_id:
                                    valid_z = next((m for m in battle.available_moves if m.id == req_move_id), None)
                            
                            if req_type or req_move_id:
                                if valid_z:
                                    action_obj = valid_z 
                                elif (req_type and str(action_obj.type).split('.')[-1].lower() != req_type) or (req_move_id and action_obj.id != req_move_id):
                                    is_z = False 
                            else:
                                is_z = False 

                    # Disparo da ordem de ataque para o servidor
                    return self.create_order(
                        action_obj, 
                        terastallize=is_tera, 
                        mega=is_mega, 
                        z_move=is_z,
                        dynamax=is_dyna
                    )
                else:
                    # É uma ordem de troca de Pokémon
                    return self.create_order(action_obj)
            else:
                return self.choose_random_move(battle)

        except Exception as e:
            print(f"[ERRO NO TURNO] {e}")
            traceback.print_exc()
            return self.choose_random_move(battle)

# =============================================================================
# BLOCO DE EXECUÇÃO PADRÃO
# =============================================================================
async def main():
    n_battles = 10000
    CONCURRENCY = 7
    
    team_builder = RandomTeamFromPool(TEAMS_LIST)
    
    bot = BLUE(
        battle_format="gen9nationaldex", 
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )

    # [FASE 1] - MAX DAMAGE (O Saco de Pancadas Suicida)
    # rival = MaxDamagePlayer(
    #     battle_format="gen9nationaldex", 
    #     server_configuration=LOCAL_CONFIG,
    #     team=team_builder,
    #     max_concurrent_battles=CONCURRENCY
    # )

    # [FASE 2] - INSTINTO SIMPLES (A Ponte Tática)
    rival = InstinctBot(
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
        while bot.total_completed_battles < n_battles:
            faltam = n_battles - bot.total_completed_battles
            current_batch = min(BLOCK_SIZE, faltam)
            
            try:
                await asyncio.wait_for(
                    bot.battle_against(rival, n_battles=current_batch),
                    timeout=300
                )
            except asyncio.TimeoutError:
                print("\n[WATCHDOG] O servidor Showdown engoliu um pacote (Ghost Battle). Destravando rede...")
                bot.battles.clear()
                rival.battles.clear()
                await asyncio.sleep(2)
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