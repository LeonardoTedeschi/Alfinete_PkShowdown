import numpy as np
import pickle
import os
import random

class BlueBrain:
    def __init__(self, alpha=0.1, gamma=0.8, epsilon=0.3, min_epsilon=0.01, decay=0.99995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        
        self.last_state = None
        self.last_action_idx = None
        self.previous_my_fainted = 0
        self.previous_opp_fainted = 0
        
        # EXATAMENTE AS 5 AÇÕES DA MATRIZ
        self.actions = ["ATTACK", "SWITCH", "SETUP", "HAZARD", "HEAL"]

        self.load_model("blue_brain.pkl")

    def _map_to_q_action(self, intent):
        """Força as intenções do Core a se enquadrarem nas 5 ações limitadas da Q-Table."""
        mapping = {
            "HEAL_50": "HEAL",
            "TEAM_CURE": "HEAL",
            "CLEAN": "HAZARD",
            "STAT_CLEAN": "HAZARD",
            "STATUS": "SETUP",
            "DEBUFF": "SETUP",
            "PROTECT": "SETUP",
            "BUFF": "SETUP"
        }
        return mapping.get(intent, intent)

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
        
        if self.last_action_idx is not None and self.actions[self.last_action_idx] == "SWITCH":
            reward += R_SWITCH_PENALTY

        self.previous_my_fainted = current_my_fainted
        self.previous_opp_fainted = current_opp_fainted
        return reward

    def update_feedback(self, current_state, battle):
        """Calcula o resultado do turno anterior e atualiza a matriz."""
        reward = self.calculate_reward(battle)
        
        if self.last_state is not None and self.last_action_idx is not None:
            if current_state not in self.q_table:
                self.q_table[current_state] = np.zeros(len(self.actions))
            
            old_val = self.q_table[self.last_state][self.last_action_idx]
            next_max = np.max(self.q_table[current_state])
            
            new_val = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * next_max)
            self.q_table[self.last_state][self.last_action_idx] = new_val

    def decide_action(self, state, instinct_intent):
        """Avalia a intenção com base no histórico da Q-Table e decide a ação final."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        mapped_intent = self._map_to_q_action(instinct_intent)
        if mapped_intent not in self.actions:
            mapped_intent = "ATTACK"
        
        instinct_idx = self.actions.index(mapped_intent)
        q_value_instinct = self.q_table[state][instinct_idx]

        # 1. Instinto >= 0 (Prioridade absoluta)
        if q_value_instinct >= 0:
            action_idx = instinct_idx
        else:
            # Filtra todas as outras opções disponíveis
            available_indices = [i for i in range(len(self.actions)) if i != instinct_idx]
            
            # 2. Busca alternativas no mesmo estado que sejam > 0
            positive_options = [i for i in available_indices if self.q_table[state][i] > 0]
            
            if positive_options:
                action_idx = max(positive_options, key=lambda idx: self.q_table[state][idx])
            else:
                # 3. Nenhuma alternativa > 0, decisão puramente aleatória
                action_idx = random.choice(available_indices)

        self.last_state = state
        self.last_action_idx = action_idx
        
        # Atualiza a taxa de exploração matemática, mesmo que a decisão acima não a utilize mais
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return self.actions[action_idx]

    def _get_root_path(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, filename)

    def save_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        data = {"q_table": self.q_table, "epsilon": self.epsilon}
        try:
            with open(filepath, "wb") as f: pickle.dump(data, f)
        except: pass

    def load_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    self.q_table = data.get("q_table", {})
                    self.epsilon = data.get("epsilon", self.epsilon)
                return True
            except: pass
        return False