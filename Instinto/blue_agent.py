import asyncio
import sys
import os
import time
import logging
import gc
import traceback
import warnings

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sobe um nível: sai da pasta 'Instinto' e vai para a raiz 'Bot-QV-Pokemon'
repo_root = os.path.abspath(os.path.join(current_dir, '..'))

# Mapeia a pasta 'Suporte_Treinamento'
suporte_treinamento_dir = os.path.join(repo_root, 'Suporte_Treinamento')

# Adiciona os caminhos ao sistema para o Python enxergar os arquivos
if current_dir not in sys.path:
    sys.path.append(current_dir)
if suporte_treinamento_dir not in sys.path:
    sys.path.append(suporte_treinamento_dir)

# Silencia avisos e logs poluentes do Poke-env
warnings.filterwarnings("ignore")
logging.getLogger("poke-env").setLevel(logging.ERROR)

from poke_env import ServerConfiguration
from poke_env.player import Player

# --- 2. IMPORTAÇÕES DA ARQUITETURA ---
try:
    from instinct_core import InstinctCore
    from blue_brain import BlueBrain
    
    # Agora ele vai procurar a pasta 'Suporte' dentro de 'Suporte_Treinamento'
    from Suporte.teams import RandomTeamFromPool, TEAMS_LIST
    import Suporte.plot_graph as plot_graph
    from Suporte.rivals import MaxDamagePlayer
except ImportError as e:
    print(f"[ERRO] Falha de importação: {e}")
    sys.exit(1)

# Configuração Local
LOCAL_CONFIG = ServerConfiguration("ws://127.0.0.1:8000/showdown/websocket", "http://127.0.0.1:8000/")

# =============================================================================
# O AGENTE EXECUTOR (BLUE)
# =============================================================================
class BLUE(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Conecta as Peças
        self.core = InstinctCore()
        self.brain = BlueBrain()
        
        # Prepara Arquivos e Variáveis de Acompanhamento
        self.paths = plot_graph.setup_training_files()
        self.total_reward_sum = 0.0

    # Atalhos para manter a compatibilidade com o log final
    @property
    def total_completed_battles(self): return self.n_finished_battles

    @property
    def total_wins(self): return self.n_won_battles

    def save_brain_silently(self):
        self.brain.save_model("blue_brain.pkl")

    def check_finished_battles(self):
        # Limpeza genérica caso haja batalhas presas na memória
        pass

    def teampreview(self, battle):
        """Passo 2: Core decide o melhor Lead."""
        return self.core.get_best_lead(battle)

    def choose_move(self, battle):
        """Passos 3, 4 e 5: O Ciclo de Turno."""
        # 1. Verificação de Troca Forçada (Fora do Q-Learning)
        switch_forced = False
        if isinstance(battle.force_switch, list): switch_forced = any(battle.force_switch)
        else: switch_forced = bool(battle.force_switch)

        if switch_forced or (battle.active_pokemon and battle.active_pokemon.fainted):
            best_switch = self.core.get_best_switch(battle)
            if best_switch: return self.create_order(best_switch)
            return self.choose_random_move(battle)

        try:
            # 2. Core processa a situação atual
            current_state = self.core.get_state(battle)
            instinct_intent = self.core.get_intent(battle)

            # 3. Brain avalia o resultado do turno anterior (Feedback)
            # Somamos a recompensa para exibir no log final
            reward = self.brain.calculate_reward(battle)
            self.total_reward_sum += reward
            self.brain.update_feedback(current_state, battle)

            # 4. Brain dá a palavra final filtrando a intenção do Core
            final_decision = self.brain.decide_action(current_state, instinct_intent)

            # 5. Agente pede ao Core o melhor objeto para a decisão do Brain e executa
            execution_list = [final_decision, "ATTACK", "SWITCH"]
            best_object = self.core.get_best_execution_object(execution_list, battle)

            if best_object:
                return self.create_order(best_object)
            else:
                return self.choose_random_move(battle)

        except Exception as e:
            traceback.print_exc()
            return self.choose_random_move(battle)


# =============================================================================
# BLOCO DE EXECUÇÃO (TREINAMENTO STANDALONE)
# =============================================================================
async def main():
    n_battles = 2000    
    CONCURRENCY = 3
    
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
        except Exception as e: 
            print(f"Aviso ao gerar gráfico: {e}")
        
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