import numpy as np
import pickle
import os
import random

class BlueBrain:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.4, min_epsilon=0.01, decay=0.99995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        
        # As 12 Ações Exatas do Instinto (Sem agrupamentos)
        self.actions = [
            "ATTACK", "SWITCH", "BUFF", "DEBUFF", "STATUS", "PROTECT", 
            "HAZARD", "HEAL", "HEAL_50", "TEAM_CURE", "CLEAN", "STAT_CLEAN"
        ]

        self.load_model("blue_brain.pkl")

    def calculate_reward(self, battle, history):
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
        
        prev_my_fainted = history.get('my_fainted', 0)
        prev_opp_fainted = history.get('opp_fainted', 0)
        last_action = history.get('last_action', None)
        
        new_kills = current_opp_fainted - prev_opp_fainted
        new_deaths = current_my_fainted - prev_my_fainted
        
        if new_kills > 0: reward += (R_KILL * new_kills)
        if new_deaths > 0: reward += (R_DEATH * new_deaths)
        
        if last_action == "SWITCH": reward += R_SWITCH_PENALTY

        return reward

    def update_feedback(self, current_state, last_state, last_action, reward):
        if last_state is not None and last_action is not None:
            if current_state not in self.q_table:
                self.q_table[current_state] = np.zeros(len(self.actions))
            if last_state not in self.q_table:
                self.q_table[last_state] = np.zeros(len(self.actions))
            
            action_idx = self.actions.index(last_action)
            old_val = self.q_table[last_state][action_idx]
            next_max = np.max(self.q_table[current_state])
            
            new_val = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * next_max)
            self.q_table[last_state][action_idx] = new_val

    def decide_action(self, state, instinct_intent):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        if instinct_intent not in self.actions:
            instinct_intent = "ATTACK"
        
        instinct_idx = self.actions.index(instinct_intent)

        # Exploração aleatória controlada pelo Epsilon
        if random.random() < self.epsilon:
            action_idx = random.choice(range(len(self.actions)))
        else:
            q_instinct = self.q_table[state][instinct_idx]
            best_action_idx = np.argmax(self.q_table[state])
            best_q_value = self.q_table[state][best_action_idx]

            # Regra 1: Se o instinto for positivo ou neutro (0.0)
            if q_instinct >= 0:
                # Checa as demais opções. Se houver uma mais positiva, escolhe ela.
                if best_q_value > q_instinct:
                    action_idx = best_action_idx
                # Se não houver (ou houver empate), usamos a do instinto.
                else:
                    action_idx = instinct_idx
            
            # Regra 2: Se o instinto for negativo
            else:
                # Buscamos outra opção com recompensa mais alta (que não seja negativa)
                if best_q_value >= 0:
                    action_idx = best_action_idx
                # Se não houver nenhuma (todas são negativas), fazemos uma jogada aleatória
                else:
                    available_indices = [i for i in range(len(self.actions)) if i != instinct_idx]
                    action_idx = random.choice(available_indices) if available_indices else instinct_idx

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