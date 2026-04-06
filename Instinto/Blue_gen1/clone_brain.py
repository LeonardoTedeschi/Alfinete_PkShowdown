import numpy as np
import pickle
import os

class CloneBrain:
    def __init__(self):
        self.q_table = {}
        self.actions = [
            "ATTACK", "SWITCH", "BUFF", "DEBUFF", "STATUS", "PROTECT", 
            "HAZARD", "HEAL", "HEAL_50", "TEAM_CURE", "CLEAN", "STAT_CLEAN"
        ]
        self.load_model("blue_brain_gen1.pkl")

    def decide_action(self, state, instinct_intent):
        if state not in self.q_table:
            return instinct_intent if instinct_intent in self.actions else "ATTACK"

        if instinct_intent not in self.actions:
            instinct_intent = "ATTACK"
        
        instinct_idx = self.actions.index(instinct_intent)
        q_instinct = self.q_table[state][instinct_idx]
        
        best_action_idx = np.argmax(self.q_table[state])
        best_q_value = self.q_table[state][best_action_idx]

        # O clone NUNCA explora. Ele sempre pega a opção >= 0
        if q_instinct >= 0:
            action_idx = best_action_idx if best_q_value > q_instinct else instinct_idx
        else:
            action_idx = best_action_idx if best_q_value >= 0 else instinct_idx
        
        return self.actions[action_idx]

    def load_model(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    self.q_table = data.get("q_table", {})
            except: pass