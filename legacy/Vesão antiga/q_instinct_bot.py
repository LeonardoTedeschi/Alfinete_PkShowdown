import numpy as np
import pickle
import os
import random
import logging
import traceback
from poke_env.player import Player
from instinct_bot import InstinctBot, Role, MatchupState

logging.getLogger("poke-env").setLevel(logging.ERROR)

class Blue_bot(InstinctBot):
    def __init__(self, alpha=0.1, gamma=0.8, epsilon=0.3, min_epsilon=0.01, decay=0.99995, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- HIPERPARÂMETROS ---
        self.alpha = alpha           
        self.gamma = gamma           
        self.epsilon = epsilon       
        self.min_epsilon = min_epsilon 
        self.epsilon_decay = decay   
        
        self.q_table = {}
        
        # --- ESTADOS INTERNOS ---
        self.last_state = None
        self.last_action_idx = None
        self.previous_my_fainted = 0
        self.previous_opp_fainted = 0
        
        # AÇÕES POSSÍVEIS (Mapeadas para as categorias base)
        self.actions = ["ATTACK", "SWITCH", "HEAL", "STATUS", "HAZARD", "BUFF"]

        self.load_model("blue_brain.pkl")

    # =========================================================================
    # 1. PERCEPÇÃO DE ESTADO
    # =========================================================================

    def _get_weather_state(self, battle):
        if "trickroom" in battle.fields: return "TRICK_ROOM"
        
        if battle.weather:
            w_name = next(iter(battle.weather)).name
            if w_name in ["HAIL", "SNOW"]: return "HAIL"
            if w_name == "SANDSTORM": return "SANDSTORM"
            if w_name in ["RAINDANCE", "PRIMORDIALSEA"]: return "RAIN"
            if w_name in ["SUNNYDAY", "DESOLATELAND"]: return "SUN"
            
        return "NORMAL"

    def _get_hp_bucket(self, pokemon):
        if not pokemon or pokemon.fainted: return "DEAD"
        hp = pokemon.current_hp_fraction
        if hp >= 0.8: return "FULL"
        if hp >= 0.5: return "MED"
        if hp >= 0.2: return "LOW"
        return "CRIT"

    def _get_speed_tier(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not active or not opponent: return "UNKNOWN"
        
        my_speed = active.base_stats['spe']
        opp_speed = opponent.base_stats['spe']
        
        if my_speed > opp_speed * 1.1: return "FASTER"
        if opp_speed > my_speed * 1.1: return "SLOWER"
        return "TIED"

    def _get_state(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        if not active or not opponent:
            return ("WAITING", "WAITING", "NONE", "FULL", "NORMAL", "TIED")

        try:
            my_role = self._get_role(active).name
            opp_role = self._get_role(opponent).name
            matchup = self._get_matchup_state(active, opponent).name
        except:
            my_role, opp_role, matchup = "UNKNOWN", "UNKNOWN", "NEUTRAL"

        return (
            my_role,
            opp_role,
            matchup,
            self._get_hp_bucket(active),
            self._get_weather_state(battle),
            self._get_speed_tier(battle)
        )

    def _map_to_q_action(self, instinct_action):
        """Reduz a granularidade do instinto para as ações mapeadas da Q-Table."""
        mapping = {
            "HEAL_50": "HEAL",
            "TEAM_CURE": "HEAL",
            "CLEAN": "HAZARD",
            "STAT_CLEAN": "HAZARD",
            "DEBUFF": "STATUS",
            "PROTECT": "STATUS"
        }
        return mapping.get(instinct_action, instinct_action)

    # =========================================================================
    # 2. SISTEMA DE RECOMPENSA (REWARD)
    # =========================================================================

    def calculate_reward(self, battle):
        R_WIN = 2000.0   
        R_LOSE = -1500.0 
        R_KILL = 300.0   
        R_DEATH = -150.0 
        R_SWITCH_PENALTY = -20.0
        
        reward = 0
        
        if battle.won: reward += R_WIN
        elif battle.lost: reward += R_LOSE
            
        current_my_fainted = len([m for m in battle.team.values() if m.fainted])
        current_opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        
        new_kills = current_opp_fainted - self.previous_opp_fainted
        new_deaths = current_my_fainted - self.previous_my_fainted
        
        if new_kills > 0: reward += (R_KILL * new_kills)
        if new_deaths > 0: reward += (R_DEATH * new_deaths)
        
        if self.last_action_idx is not None:
            if self.actions[self.last_action_idx] == "SWITCH":
                reward += R_SWITCH_PENALTY

        self.previous_my_fainted = current_my_fainted
        self.previous_opp_fainted = current_opp_fainted
        
        return reward

    # =========================================================================
    # 3. CÉREBRO PRINCIPAL (Integração Instinto + Q-Learning)
    # =========================================================================

    def choose_move(self, battle):
        switch_forced = False
        if isinstance(battle.force_switch, list): switch_forced = any(battle.force_switch)
        else: switch_forced = bool(battle.force_switch)

        if switch_forced or (battle.active_pokemon and battle.active_pokemon.fainted):
            return self._choose_switch(battle)

        try:
            # 1. Recupera o estado atual e atualiza a matriz com a recompensa passada
            state = self._get_state(battle)
            reward = self.calculate_reward(battle)
            
            if self.last_state is not None and self.last_action_idx is not None:
                self.update_q_table(self.last_state, self.last_action_idx, reward, state)

            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.actions))

            # 2. Pergunta ao Instinto Puro qual a intenção dele
            active = battle.active_pokemon
            opponent = battle.opponent_active_pokemon
            my_role = self._get_role(active)
            opp_role = self._get_role(opponent)
            matchup = self._get_matchup_state(active, opponent)
            is_faster = self._estimate_stat(active, 'spe') >= self._estimate_stat(opponent, 'spe')
            is_threatening = self._is_threatening(active, opponent)

            instinct_list = ["ATTACK"]
            if my_role == Role.SPEED_SWEEPER:
                if opp_role == Role.SPEED_SWEEPER: instinct_list = self._matrix_sweeper_vs_sweeper(is_faster, matchup, is_threatening)
                elif opp_role == Role.UTILITY: instinct_list = self._matrix_sweeper_vs_utility(matchup, is_faster, is_threatening)
                else: instinct_list = self._matrix_sweeper_vs_tank(active, opponent, matchup)
            elif my_role == Role.UTILITY:
                instinct_list = self._matrix_utility_logic(active, opponent, matchup, opp_role, is_faster)
            elif my_role == Role.TANK_BULK:
                instinct_list = self._matrix_tank_logic(active, opponent, matchup, opp_role)

            # Mapeia a ação do instinto
            primary_instinct = instinct_list[0]
            mapped_instinct = self._map_to_q_action(primary_instinct)
            if mapped_instinct not in self.actions: mapped_instinct = "ATTACK"
            
            instinct_idx = self.actions.index(mapped_instinct)

            # -----------------------------------------------------------------
            # 3. LÓGICA DE SOBREPOSIÇÃO: INSTINTO VS CÉREBRO
            # -----------------------------------------------------------------
            q_value_instinct = self.q_table[state][instinct_idx]

            if q_value_instinct >= 0:
                # O instinto provou ser bom (ou é inédito = 0). USE-O.
                action_idx = instinct_idx
            else:
                # O instinto tem saldo NEGATIVO. O cérebro vetou a jogada.
                # Excluímos a jogada ruim da lista de opções:
                available_indices = [i for i in range(len(self.actions)) if i != instinct_idx]
                
                # JOGADA ALEATÓRIA: Tentando aprender a melhor jogada.
                if random.random() < self.epsilon:
                    # Exploração: Joga algo completamente aleatório para descobrir
                    action_idx = random.choice(available_indices)
                else:
                    # Explotação: Usa a alternativa que as jogadas aleatórias passadas 
                    # já provaram ser a "melhor jogada possível" para substituir o instinto.
                    action_idx = max(available_indices, key=lambda idx: self.q_table[state][idx])

            # 4. Atualiza Memória e Decai Epsilon
            self.last_state = state
            self.last_action_idx = action_idx
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # 5. Roteia a decisão de volta para o motor de execução base
            chosen_action_str = self.actions[action_idx]
            execution_list = [chosen_action_str, "ATTACK", "SWITCH"]
            
            return self._execute_action(execution_list, battle)

        except Exception as e:
            traceback.print_exc()
            return self.choose_random_move(battle)

    # =========================================================================
    # 4. MEMÓRIA E SALVAMENTO
    # =========================================================================

    def _get_root_path(self, filename):
        """Garante que o cérebro seja salvo sempre na raiz do projeto."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, filename)

    def update_q_table(self, state, action_idx, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        
        old_val = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])
        
        new_val = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action_idx] = new_val

    def save_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        data = {"q_table": self.q_table, "epsilon": self.epsilon}
        try:
            with open(filepath, "wb") as f: pickle.dump(data, f)
        except Exception as e:
            print(f"Erro ao salvar: {e}")

    def load_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    self.q_table = data.get("q_table", {})
                    self.epsilon = data.get("epsilon", self.epsilon)
                print(f"[BLUE BOT] Cérebro carregado: {len(self.q_table)} estados aprendidos.")
                return True
            except Exception as e:
                print(f"Erro ao carregar: {e}")
        print("[BLUE BOT] Iniciando aprendizado do zero.")
        return False