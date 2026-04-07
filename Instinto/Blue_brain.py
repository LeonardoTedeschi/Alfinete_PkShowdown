import numpy as np
import pickle
import os
import random

class BlueBrain:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.0, min_epsilon=0.0, decay=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        
        # As 13 Ações Definitivas do Competitivo
        self.actions = [
            "ATTACK", "SWITCH", "BUFF", "DEBUFF", "STATUS", "PROTECT", 
            "HAZARD", "CLEAN_HAZARD", "HEAL", "HEAL_STATUS", "STAT_CLEAN",
            "PHAZE", "FIELD_CONTROL"
        ]

        self.load_model("blue_brain.pkl")

    def calculate_reward(self, battle, history):
        R_WIN = 2000.0   
        R_LOSE = -4000.0 
        R_KILL = 200.0   
        R_DEATH = -200.0
        
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

    def decide_action(self, state, instinct_intent, available_actions):
        # Proteção: Garante que o Cérebro sempre escolha algo que está na lista disponível
        if instinct_intent not in available_actions:
            instinct_intent = available_actions[0] if available_actions else "ATTACK"

        # Se o estado for inédito, segue o instinto
        if state not in self.q_table:
            return instinct_intent

        # Converte a lista de strings para os índices reais da Q-Table
        available_indices = [self.actions.index(act) for act in available_actions if act in self.actions]
        if not available_indices: return instinct_intent # Fallback
        
        instinct_idx = self.actions.index(instinct_intent)
        
        # Filtra a Q-Table para enxergar APENAS as ações permitidas neste turno
        valid_q_values = {idx: self.q_table[state][idx] for idx in available_indices}
        
        q_instinct = valid_q_values.get(instinct_idx, 0.0)
        
        # Acha a melhor opção DENTRO das disponíveis
        best_action_idx = max(valid_q_values, key=valid_q_values.get)
        best_q_value = valid_q_values[best_action_idx]

        # Lógica de Tomada de Decisão baseada na Recompensa
        if q_instinct >= 0:
            # Se o instinto é positivo/neutro, só desobedece se a matriz tiver uma ideia melhor
            action_idx = best_action_idx if best_q_value > q_instinct else instinct_idx
        else:
            # Se o instinto é negativo, tenta a melhor ideia da matriz
            if best_q_value >= 0:
                action_idx = best_action_idx
            # Se a melhor ideia da matriz TAMBÉM é negativa, estamos em um beco sem saída. Exploração Reativa!
            else:
                options_to_explore = [i for i in available_indices if i != instinct_idx]
                action_idx = random.choice(options_to_explore) if options_to_explore else instinct_idx
        
        # Lógica de Epsilon mantida apenas para evitar quebra do arquivo .pkl
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