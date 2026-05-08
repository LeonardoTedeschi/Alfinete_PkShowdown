import asyncio
import sys
import os
import time
import logging
import csv
import warnings
import traceback
import threading  # Usado para o Lock do Pickle
import json
import argparse

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
suporte_treinamento_dir = os.path.join(repo_root, 'Suporte_Treinamento')

# Mapeia a pasta exata onde o bot antigo está
pasta_antiga = os.path.join(current_dir, 'Vesão antiga') 

if current_dir not in sys.path: sys.path.append(current_dir)
if suporte_treinamento_dir not in sys.path: sys.path.append(suporte_treinamento_dir)

# Obriga o Python a ler o que tem dentro da pasta do bot antigo
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

# --- 2. CONFIGURAÇÃO DO CIRCUITO ---
def load_circuit_config():
    config_path = os.path.join(current_dir, "circuit_state.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

circuit_cfg = load_circuit_config()

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default=circuit_cfg.get("phase", "maxdamage"))
parser.add_argument("--opponent", type=str, default=circuit_cfg.get("opponent", "maxdamage"))
parser.add_argument("--n-battles", type=int, default=circuit_cfg.get("n_battles", 10000))
# --- NOME LIMPO ---
parser.add_argument("--brain", type=str, default=circuit_cfg.get("brain_filename", "blue_brain.pkl"))
args, _ = parser.parse_known_args()

# --- CONFIGURAÇÕES DE CONEXÃO ---
LOCAL_CONFIG = ServerConfiguration("ws://127.0.0.1:8000/showdown/websocket", "http://127.0.0.1:8000/")
BLOCK_SIZE = 500 # A cada 500 batalhas teremos o log no CSV e o Print

class BLUE(Player):
    def __init__(self, *args, **kwargs):
        # 1. Cria todas as variáveis de estado PRIMEIRO para evitar Race Conditions
        self.core = InstinctCore()
        self.brain = BlueBrain()
        
        # --- Passamos o core para o cérebro ---
        self.brain.core = self.core 
        
        self.paths = plot_graph.setup_training_files()
        
        self.battle_history = {} 
        self.total_completed_battles = 0
        self.aborted_battles = 0 # Contador para Ghost Battles
        self.total_wins = 0
        self.total_reward_sum = 0.0
        self.block_wins = 0
        
        # Lock de thread para evitar corrupção do arquivo PKL com alta concorrência
        self._save_lock = threading.Lock() 
        
        self._init_csv()
        
        # 2. Conecta ao servidor e se declara pronto POR ÚLTIMO
        super().__init__(*args, **kwargs)

    def _init_csv(self):
        try:
            with open(self.paths['csv'], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Batalhas", "WinRate_Bloco", "Epsilon", "Reward", "Ghost_Battles"])
        except: pass

    def save_brain_silently(self):
        with self._save_lock:
            # Salva usando o nome de arquivo passado pelo Circuito
            filename = getattr(self.brain, 'filename', "blue_brain.pkl")
            self.brain.save_model(filename)

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
            
            # --- CORREÇÃO: Só processa se a recompensa NÃO foi processada no turno ---
            if history and not history.get('reward_processed', False):
                reward = self.brain.calculate_reward(battle, history)
                self.total_reward_sum += reward
                
                last_state = history.get('state')
                last_action_tuple = history.get('last_action')
                
                if last_state and last_action_tuple:
                    current_state = ("TERMINAL_WIN",) if battle.won else ("TERMINAL_LOSS",)
                    self.brain.update_feedback(current_state, last_state, last_action_tuple, reward)
                
                history['reward_processed'] = True
            
            # Remove do history independente
            if battle.battle_tag in self.battle_history:
                del self.battle_history[battle.battle_tag]
                
        except Exception as e:
            print(f"[Aviso] Erro ao processar fim de batalha: {e}")

        # --- REMOVER save_brain_silently duplicado ---
        self.save_brain_silently()
 
        if hasattr(self.brain, 'replay_experience'):
            self.brain.replay_experience()

    def teampreview(self, battle):
        try:
            return self.core.get_best_lead(battle)
        except Exception:
            return "/team 123456"

    def choose_move(self, battle):
        # Trava de Segurança Imediata para Race Condition
        if battle.finished:
            return self.choose_random_move(battle)

        self.check_finished_battles()
        
        if battle.finished:
            return self.choose_random_move(battle)

        try:
            history = self.battle_history.get(battle.battle_tag, {})
            current_state = self.core.get_state(battle)
            
            if history and not battle.finished and not history.get('reward_processed', False):
                reward = self.brain.calculate_reward(battle, history)
                self.total_reward_sum += reward
                last_state = history.get('state')
                last_action_tuple = history.get('last_action')
                if last_state and last_action_tuple:
                    self.brain.update_feedback(current_state, last_state, last_action_tuple, reward)

                if hasattr(self.brain, 'replay_experience'):
                        self.brain.replay_experience()
                    
                # Acende a flag informando que a recompensa dessa transição já foi absorvida
                history['reward_processed'] = True

            # 1. Gerenciamento de Troca Forçada (Morte do Pokémon)
            switch_forced = False
            if isinstance(battle.force_switch, list): switch_forced = any(battle.force_switch)
            else: switch_forced = bool(battle.force_switch)

            if switch_forced or (battle.active_pokemon and battle.active_pokemon.fainted):
                best_switch = self.core.get_post_faint_switch(battle, history)
                if best_switch and best_switch in battle.available_switches:
                    return self.create_order(best_switch)
                return self.choose_random_move(battle)

            # 4. Puxa o Perfil completo do Instinto (Agora recebe o has_lethal)
            primary, conf, ranking_list, valid_base_macros, has_lethal = self.core.get_instinct_profile(battle, history)
            
            valid_actions_for_brain = []
            is_mec_avail = battle.can_tera or battle.can_mega_evolve or battle.can_z_move or battle.can_dynamax
            
            for macro in valid_base_macros:
                # --- O FILTRO DA MEGA EVOLUÇÃO ---
                if battle.can_mega_evolve and "SWITCH" not in macro:
                    valid_actions_for_brain.append(f"{macro}_MEC")
                else:
                    valid_actions_for_brain.append(macro)
                    if is_mec_avail and "SWITCH" not in macro:
                        valid_actions_for_brain.append(f"{macro}_MEC")
            
            # 5. O Cérebro toma a decisão com a matemática blindada
            final_decision_tuple = self.brain.decide_action(current_state, valid_actions_for_brain, ranking_list)
            action_str, mec_decision = final_decision_tuple

            # =================================================================
            # EXECUÇÃO MECÂNICA E TRADUÇÃO (Instinto -> Poke-env)
            # =================================================================
            
            action_obj = self.core.get_best_execution_object(action_str, battle, history)
            
            if not action_obj:
                return self.choose_random_move(battle)

            # --- CORREÇÃO: PIVOT OVERRIDE ANTES DO HISTÓRICO ---
            if hasattr(action_obj, 'id') and action_obj.id in ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport']:
                if "SWITCH" in action_str:
                    action_str = "ATTACK_PIVOT"
                    final_decision_tuple = (action_str, mec_decision)

            # --- PREPARAÇÃO DO HISTÓRICO (MOMENTUM) ---
            opp_species = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else None
            active = battle.active_pokemon
            opp = battle.opponent_active_pokemon
            
            opp_switched = False
            if history:
                prev_opp_species = history.get('opp_species')
                prev_opp_hp = history.get('opp_hp_prev', 0.0)
                
                if prev_opp_species and opp_species and (prev_opp_species != opp_species):
                    if prev_opp_hp > 0.0:
                        opp_switched = True

            prev_action = history.get('last_action', (None, None))[0] if history else None

            # Salvamos o histórico com a decisão final e RESETAMOS a flag de processamento
            self.battle_history[battle.battle_tag] = {
                'state': current_state,
                'last_action': final_decision_tuple,
                'prev_action': prev_action,
                'reward_processed': False,
                'has_lethal': has_lethal,
                
                # NOVO: Registra a espécie para impedir bugs no cálculo de Dano/Cura
                'my_species': active.species if active else None,
                'opp_species': opp.species if opp else None,
                
                'team_hp': {m.species: m.current_hp_fraction for m in battle.team.values()},
                'team_status': {m.species: str(m.status) if m.status else "CLEAN" for m in battle.team.values() if hasattr(m, 'status')},
                
                'my_hp_prev': active.current_hp_fraction if active else 0.0,
                'opp_hp_prev': opp.current_hp_fraction if opp else 0.0,
                'my_hp_bucket': self.core.get_hp_bucket(active),
                'opp_hp_bucket': self.core.get_opp_hp_bucket(opp),
                'my_status': self.core.get_status_state(active),
                'opp_status': self.core.get_status_state(opp),
                'my_boosts': active.boosts.copy() if active else {},
                'my_hazards': self.core.get_hazard_state(battle.side_conditions),
                'opp_species': opp_species,
                
                'opp_switched': opp_switched, 
                
                'my_fainted': len([m for m in battle.team.values() if m.fainted]),
                'opp_fainted': len([m for m in battle.opponent_team.values() if m.fainted]),
                
                # Memória de 2 turnos atrás para calcular o Revenge Kill
                'my_fainted_prev_turn': history.get('my_fainted', 0),
                
                # --- NOVAS VARIÁVEIS PARA RECOMPENSA DENSA ---
                'my_alive': len([m for m in battle.team.values() if not m.fainted]),
                'opp_alive': len([m for m in battle.opponent_team.values() if not m.fainted]),
                'matchup': self.core.get_matchup_state(active, opp) if active and opp else None
            }

            if action_obj in battle.available_moves:
                
                # Injeção da decisão de Mecânica (Hierarquia Corrigida)
                should_activate = (mec_decision == "ACTIVATE")
                is_tera = False
                is_mega = False
                is_dyna = False
                is_z = False
                
                if should_activate:
                    if battle.can_tera:
                        is_tera = True
                    elif battle.can_z_move:
                        is_z = True
                    elif battle.can_mega_evolve:
                        is_mega = True
                    elif battle.can_dynamax:
                        is_dyna = True

                # --- INTERCEPTADOR E CORREÇÃO DE Z-MOVES ---
                if is_z and active.item:
                    item_str = str(active.item).lower()
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
                        
                        try:
                            action_type_str = action_obj.type.name.lower() if action_obj.type else ""
                            
                            if req_type:
                                if action_type_str != req_type:
                                    valid_z = next((m for m in battle.available_moves if m.type and m.type.name.lower() == req_type), None)
                            elif req_move_id:
                                if action_obj.id != req_move_id:
                                    valid_z = next((m for m in battle.available_moves if m.id == req_move_id), None)
                            
                            if req_type or req_move_id:
                                if valid_z:
                                    action_obj = valid_z 
                                elif (req_type and action_type_str != req_type) or (req_move_id and action_obj.id != req_move_id):
                                    is_z = False 
                            else:
                                is_z = False 
                        except Exception as ez:
                            print(f"[Aviso Z-Move] Falha na interpretação: {ez}")
                            is_z = False

                return self.create_order(
                    action_obj, 
                    terastallize=is_tera, 
                    mega=is_mega, 
                    z_move=is_z,
                    dynamax=is_dyna
                )
            elif action_obj in battle.available_switches:
                return self.create_order(action_obj)
            else:
                return self.choose_random_move(battle)

        except Exception as e:
            print(f"[ERRO NO TURNO] {e}")
            traceback.print_exc()
            return self.choose_random_move(battle)

# =============================================================================
# BLOCO DE EXECUÇÃO PADRÃO E CIRCUITO
# =============================================================================
async def main():
    n_battles = args.n_battles
    CONCURRENCY = 7
    
    team_builder = RandomTeamFromPool(TEAMS_LIST)
    
    bot = BLUE(
        battle_format="gen9nationaldex", 
        server_configuration=LOCAL_CONFIG,
        team=team_builder,
        max_concurrent_battles=CONCURRENCY
    )

    if hasattr(bot.brain, 'enter_phase'):
        bot.brain.enter_phase(args.phase)
        
    bot.brain.filename = args.brain 
    
    # --- CORREÇÃO DA AMNÉSIA: O CÉREBRO CARREGA DO DISCO AQUI ---
    bot.brain.load_model(args.brain)

    if args.opponent == "maxdamage":
        rival = MaxDamagePlayer(
            battle_format="gen9nationaldex",
            server_configuration=LOCAL_CONFIG,
            team=team_builder,
            max_concurrent_battles=CONCURRENCY
        )
    elif args.opponent == "instinct":
        rival = InstinctBot(
            battle_format="gen9nationaldex",
            server_configuration=LOCAL_CONFIG,
            team=team_builder,
            max_concurrent_battles=CONCURRENCY
        )
    elif args.opponent == "selfplay_frozen":
        rival = BLUE(
            battle_format="gen9nationaldex",
            server_configuration=LOCAL_CONFIG,
            team=team_builder,
            max_concurrent_battles=CONCURRENCY
        )
        rival.brain = BlueBrain() 
        rival.brain.core = rival.core
        
        frozen_path = circuit_cfg.get("frozen_brain", "frozen_blue_brain.pkl")
        
        if os.path.exists(frozen_path):
            rival.brain.load_model(frozen_path)
            rival.brain.epsilon = 0.0 
            print(f"[SELF-PLAY] Oponente congelado: {frozen_path}")
        else:
            print("[SELF-PLAY] AVISO: Frozen não encontrado. Usando brain padrão (blue_brain.pkl).")
            rival.brain.load_model("blue_brain.pkl")
            rival.brain.epsilon = 0.0
    else:
        rival = MaxDamagePlayer(
            battle_format="gen9nationaldex",
            server_configuration=LOCAL_CONFIG,
            team=team_builder,
            max_concurrent_battles=CONCURRENCY
        )
    
    print(f"{'='*40}")
    print(f" SESSÃO: {bot.paths['id']}")
    print(f" FASE:   {args.phase}")
    print(f" RIVAL:  {args.opponent}")
    print(f" LOG:    {bot.paths['csv']}")
    print(f" META:   {n_battles} batalhas")
    print(f"{'='*40}")
    print(f"Treinando em blocos de {BLOCK_SIZE}... Pressione Ctrl+C APENAS para abortar.")

    start_time = time.time()

    try:
        while bot.total_completed_battles < n_battles:
            faltam = n_battles - bot.total_completed_battles
            current_batch = min(BLOCK_SIZE, faltam)
            
            # Salva quantas batalhas tínhamos ANTES de começar este bloco
            completed_before = bot.total_completed_battles
            
            try:
                await asyncio.wait_for(
                    bot.battle_against(rival, n_battles=current_batch),
                    timeout=2400
                )
            except asyncio.TimeoutError:
                bot.aborted_battles += 1
                print(f"\n[WATCHDOG] Servidor engoliu um pacote. Destravando rede... (Falhas: {bot.aborted_battles})")
                
                for b_id in list(bot.battles.keys()):
                    if b_id in bot.battle_history:
                        del bot.battle_history[b_id]
                
                bot.battles.clear()
                if hasattr(rival, 'battles'):
                    rival.battles.clear()
                await asyncio.sleep(2)
            except Exception:
                pass
            
            # --- CORREÇÃO DO 499: Respiro de 1 segundo para a rede descarregar a última mensagem ---
            await asyncio.sleep(1)
            bot.check_finished_battles()
            
            # === BLOCO MATEMÁTICO CORRIGIDO ===
            completed = bot.total_completed_battles
            processed_this_block = completed - completed_before

            if processed_this_block > 0:
                win_rate_bloco = (bot.block_wins / processed_this_block) * 100
                print(f"[Progresso] {completed} Batalhas | Win Rate (Bloco): {win_rate_bloco:.1f}% | Recompensa Total: {bot.total_reward_sum:.0f} | Estados: {len(bot.brain.q_table)} | Epsilon: {bot.brain.epsilon:.3f} | Erros (Rede): {bot.aborted_battles}")
                
                # --- O GATILHO POR BLOCO ESTÁ AQUI ---
                if hasattr(bot.brain, 'decay_epsilon'):
                    bot.brain.decay_epsilon()
                    
                # Salva a linha no CSV
                try:
                    with open(bot.paths['csv'], 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([completed, f"{win_rate_bloco:.1f}", f"{bot.brain.epsilon:.3f}", f"{bot.total_reward_sum:.0f}", bot.aborted_battles])
                except: pass
                
                # Reseta as vitórias do bloco e salva a Q-Table
                bot.block_wins = 0
                bot.save_brain_silently()
                        
    except KeyboardInterrupt:
        print("\n\n[!] Interrompido pelo usuário. Salvando...")
        
    finally:
        bot.check_finished_battles()
        bot.save_brain_silently()
        
        # --- PROTEÇÃO DO FROZEN APENAS NO SELFPLAY ---
        if args.phase == "selfplay":
            frozen_target = circuit_cfg.get("frozen_brain", "frozen_blue_brain.pkl")
            bot.brain.save_model(frozen_target)
            print(f"[SELF-PLAY] Cópia atual salva como snapshot congelado: {frozen_target}")
        
        end_time = time.time()
        
        valid = bot.total_completed_battles
        wins = bot.total_wins
        win_rate = (wins / valid * 100) if valid > 0 else 0.0
        
        try:
            plot_graph.generate_graph(
                csv_path=bot.paths['csv'], 
                img_output_path=bot.paths['graph'], 
                title_suffix=str(bot.paths['id']),
                opponent=args.opponent,
                phase=args.phase,
                total_battles=valid,
                final_win_rate=win_rate,
                final_states=len(bot.brain.q_table)
            )
        except: pass
        
        session_summary = {
            "phase": args.phase,
            "opponent": args.opponent,
            "total_battles": valid,
            "total_wins": wins,
            "win_rate": win_rate,
            "states": len(bot.brain.q_table),
            "total_reward": bot.total_reward_sum,
            "epsilon": bot.brain.epsilon,
            "aborted": bot.aborted_battles,
            "brain_file": args.brain
        }
        
        summary_path = os.path.join(current_dir, "last_session_summary.json")
        try:
            with open(summary_path, "w") as f:
                json.dump(session_summary, f, indent=2)
        except Exception as e:
            print(f"[Aviso] Não foi possível salvar o last_session_summary: {e}")

        print(f"\n{'='*40}")
        print(f"           RESULTADO FINAL")
        print(f"{'='*40}")
        print(f"Tempo:            {(end_time - start_time) / 60:.1f} minutos")
        print(f"Estados:          {len(bot.brain.q_table)}")
        print(f"Batalhas:         {valid}")
        print(f"Vitórias:         {wins}")
        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Recompensa Total: {bot.total_reward_sum:.0f}")
        print(f"Abortadas:        {bot.aborted_battles}")
        os._exit(0)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())