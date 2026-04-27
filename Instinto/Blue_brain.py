import numpy as np
import pickle
import os
import random
import threading

from instinct_core import MatchupState

class BlueBrain:
    # GAMMA REDUZIDO PARA 0.85: Foca mais no impacto tático presente do que no futuro estocástico distante.
    def __init__(self, alpha=0.2, gamma=0.85, epsilon=0.40, min_epsilon=0.05, decay=0.85):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        self.visit_counts = {}  
        
        self._qtable_lock = threading.Lock()
        
        self.base_actions = [
            "ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH", 
            "BUFF", "STATUS", "HEAL", "CLEAN_HAZARD", 
            "PROTECT", "DEBUFF", "STAT_CLEAN", "HEAL_STATUS", "PHAZE", 
            "FIELD_CONTROL", "HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"
        ]
        
        self.actions = []
        for act in self.base_actions:
            self.actions.append(act)
            if "SWITCH" not in act:
                self.actions.append(f"{act}_MEC")

        self.current_phase = "maxdamage"
        self.load_model("blue_brain.pkl")

    def enter_phase(self, phase_name):
        phase_config = {
            # --- ATUALIZADO: Epsilon mínimo e decay mais lento para manter exploração ---
            "maxdamage": {"epsilon_start": 0.40, "epsilon_min": 0.15, "decay": 0.92}, 
            "instinct":  {"epsilon_start": 0.30, "epsilon_min": 0.05, "decay": 0.90}, 
            "selfplay":  {"epsilon_start": 0.20, "epsilon_min": 0.02, "decay": 0.95}  
        }
        
        if phase_name in phase_config:
            cfg = phase_config[phase_name]
            self.epsilon = cfg["epsilon_start"]
            self.min_epsilon = cfg["epsilon_min"]
            self.epsilon_decay = cfg["decay"]
            self.current_phase = phase_name
            print(f"[CÉREBRO] Entrando na Fase de Aprendizado: {phase_name.upper()} (Eps: {self.epsilon:.2f} -> {self.min_epsilon:.2f})")

    def calculate_reward(self, battle, history):
        reward = 0.0

        if battle.won: return 1000.0
        if battle.lost: return -1000.0

        # --- RECOMPENSAS MACRO ESCALONADAS PARA BAIXO ---
        current_my_fainted = len([m for m in battle.team.values() if m.fainted])
        current_opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        
        if current_my_fainted > history.get('my_fainted', 0): reward -= 100.0
        if current_opp_fainted > history.get('opp_fainted', 0): reward += 100.0
        if history.get('opp_switched', False): reward += 5.0  

        # --- MICRO-RECOMPENSAS DE SOBREVIVÊNCIA (HP DELTA) ---
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_hp_prev = history.get('my_hp_prev', 0.0)
        opp_hp_prev = history.get('opp_hp_prev', 0.0)
        my_hp_curr = active.current_hp_fraction if active else 0.0
        opp_hp_curr = opp.current_hp_fraction if opp else 0.0

        my_hp_delta = my_hp_curr - my_hp_prev
        opp_hp_delta = opp_hp_curr - opp_hp_prev

        # --- ATUALIZADO: Recompensas de HP Reduzidas ---
        if opp_hp_delta < 0: reward += 10.0 * abs(opp_hp_delta)
        
        if my_hp_delta > 0: reward += 10.0 * my_hp_delta
        elif my_hp_delta < 0: reward -= 10.0 * abs(my_hp_delta)

        # --- AVALIAÇÃO POSICIONAL DENSA (CREDIT ASSIGNMENT) ---
        score = 0.0

        my_alive = len([m for m in battle.team.values() if not m.fainted])
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        
        # DELTA DE SOBREVIVENTES (Substitui a pontuação estática absoluta)
        prev_my_alive = history.get('my_alive', my_alive)
        prev_opp_alive = history.get('opp_alive', opp_alive)
        alive_delta = (my_alive - opp_alive) - (prev_my_alive - prev_opp_alive)
        score += alive_delta * 5.0  

        if active and active.status: score -= 5.0
        if opp and opp.status: score += 5.0

        if hasattr(self, 'core') and self.core:
            my_hazards = self.core.get_hazard_state(battle.side_conditions)
            opp_hazards = self.core.get_hazard_state(battle.opponent_side_conditions)
            if my_hazards == "SET": score -= 5.0
            if opp_hazards == "SET": score += 10.0

            # RESGATA O MATCHUP DO TURNO PASSADO (Quando a ação foi de fato tomada)
            matchup = history.get('matchup')
            if matchup:
                # Escala de Matchup esmagada para parear com micro-recompensas de HP
                matchup_score = {
                    MatchupState.DOMINANT: 10.0,
                    MatchupState.OFFENSIVE_ADV: 5.0,
                    MatchupState.VOLATILE: 0.0,
                    MatchupState.NEUTRAL: 0.0,
                    MatchupState.DEFENSIVE_ADV: -5.0,
                    MatchupState.STALEMATE: 0.0,
                    MatchupState.OFFENSIVE_DIS: -8.0,
                    MatchupState.DEFENSIVE_DIS: -10.0,
                    MatchupState.CRITICAL_DIS: -15.0,
                }
                score += matchup_score.get(matchup, 0.0)

        reward += score

        last_action_tuple = history.get('last_action')
        base_action = last_action_tuple[0] if last_action_tuple else None
        
        # --- ATUALIZADO: Recompensa de Troca Aumentada ---
        is_switch = base_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]
        if is_switch and active and opp:
            if hasattr(self, 'core') and self.core:
                matchup = self.core.get_matchup_state(active, opp)
                if matchup in [MatchupState.DOMINANT, MatchupState.OFFENSIVE_ADV]:
                    reward += 10.0   
                elif matchup in [MatchupState.CRITICAL_DIS, MatchupState.DEFENSIVE_DIS]:
                    reward -= 5.0   

        curr_opp_status = "AFFLICTED" if opp and opp.status else "CLEAN"
        if history.get('opp_status') == "CLEAN" and curr_opp_status == "AFFLICTED":
            reward += 10.0 

        if base_action == "HAZARD": reward += 10.0
        elif base_action == "CLEAN_HAZARD": reward += 15.0

        return reward

    def update_feedback(self, current_state, last_state, last_action_tuple, reward):
        if last_state is not None and last_action_tuple is not None:
            with self._qtable_lock:
                if last_state not in self.q_table:
                    self.q_table[last_state] = np.zeros(len(self.actions))
                    self.visit_counts[last_state] = 1
                    effective_alpha = min(0.5, self.alpha * 2.0)
                else:
                    self.visit_counts[last_state] = self.visit_counts.get(last_state, 1) + 1
                    visits = self.visit_counts[last_state]
                    effective_alpha = max(0.05, self.alpha / (1 + 0.05 * visits))
                
                base_action, mechanic = last_action_tuple
                action_str = f"{base_action}_MEC" if mechanic else base_action
                
                if action_str in self.actions:
                    action_idx = self.actions.index(action_str)
                    old_val = self.q_table[last_state][action_idx]
                    
                    if current_state in [("TERMINAL_WIN",), ("TERMINAL_LOSS",)]:
                        next_max = 0.0
                    else:
                        if current_state not in self.q_table:
                            self.q_table[current_state] = np.zeros(len(self.actions))
                            self.visit_counts[current_state] = 0
                        next_max = np.max(self.q_table[current_state])
                    
                    new_val = (1 - effective_alpha) * old_val + effective_alpha * (reward + self.gamma * next_max)
                    self.q_table[last_state][action_idx] = new_val

    def _add_action_indices(self, intent, target_list, valid_indices, is_mec_avail):
        """Adiciona índices da ação base e _MEC (se disponível) à lista, verificando duplicatas e validade."""
        if intent in self.actions:
            idx = self.actions.index(intent)
            if idx in valid_indices and idx not in target_list:
                target_list.append(idx)
        if is_mec_avail and "SWITCH" not in intent:
            mec = f"{intent}_MEC"
            if mec in self.actions:
                idx = self.actions.index(mec)
                if idx in valid_indices and idx not in target_list:
                    target_list.append(idx)

    def decide_action(self, state, instinct_profile):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
            self.visit_counts[state] = 0
            
        # --- ATUALIZADO: Herança _MEC para estados Novos E Existentes ---
        for i, act in enumerate(self.actions):
            if act.endswith("_MEC"):
                base = act.replace("_MEC", "")
                if base in self.actions:
                    base_idx = self.actions.index(base)
                    # Herda apenas se o _MEC for 0.0 e a ação base já tiver aprendido algo
                    if self.q_table[state][i] == 0.0 and self.q_table[state][base_idx] != 0.0:
                        self.q_table[state][i] = self.q_table[state][base_idx]

        primary_intent, _, ranking_list, candidate_mask = instinct_profile

        is_mec_avail = False
        if isinstance(state, tuple) and len(state) > 0:
            is_mec_avail = (state[-1] == "MEC_AVAIL")
        
        valid_indices = []
        for act in candidate_mask:
            if act in self.base_actions:
                valid_indices.append(self.actions.index(act)) 
                if is_mec_avail and "SWITCH" not in act:
                    valid_indices.append(self.actions.index(f"{act}_MEC"))

        if not valid_indices: 
            return (primary_intent, None)
            
        valid_q_values = {idx: self.q_table[state][idx] for idx in valid_indices}
        best_action_idx = max(valid_q_values, key=valid_q_values.get)
        worst_action_idx = min(valid_q_values, key=valid_q_values.get)
        
        best_q_value = valid_q_values[best_action_idx]
        worst_q_value = valid_q_values[worst_action_idx]

        if best_q_value == 0.0 and worst_q_value == 0.0:
            try:
                action_idx = self.actions.index(primary_intent)
            except ValueError:
                action_idx = random.choice(valid_indices)
        else:
            # --- EPSILON ADAPTATIVO POR VISITAS E INCERTEZA ---
            visits = self.visit_counts.get(state, 0)
            
            if visits < 5:
                state_epsilon = min(0.6, self.epsilon * 2.0)  # Exploração dobrada para estados super recentes
            elif visits < 20:
                state_epsilon = min(0.6, self.epsilon * 1.2)  # Exploração moderada
            else:
                # Pânico Escalonado e Incerteza Matemática para estados já maduros
                spread = best_q_value - worst_q_value
                if best_q_value < 0 and spread < 10.0:
                    uncertainty_factor = 1.0  
                elif best_q_value < 0:
                    uncertainty_factor = 0.5  
                else:
                    uncertainty_factor = np.exp(-spread / 20.0) 
                    
                state_epsilon = min(0.6, self.epsilon * uncertainty_factor)

            if random.random() < state_epsilon:
                # === EXPLORAÇÃO POR CAMADAS DE RISCO ===
                
                safe_indices = []
                for intent in ranking_list[:3]:
                    self._add_action_indices(intent, safe_indices, valid_indices, is_mec_avail)
                
                risky_indices = []
                for intent in ranking_list[3:]:
                    self._add_action_indices(intent, risky_indices, valid_indices, is_mec_avail)
                
                off_radar = [idx for idx in valid_indices 
                             if idx not in safe_indices and idx not in risky_indices]
                
                layer_roll = random.random()
                
                if layer_roll < 0.50:
                    pool = safe_indices
                elif layer_roll < 0.85:
                    pool = risky_indices if risky_indices else safe_indices
                else:
                    pool = off_radar if off_radar else (risky_indices if risky_indices else safe_indices)
                
                if not pool:
                    pool = valid_indices
                
                action_idx = random.choice(pool)
            else:
                action_idx = best_action_idx

        chosen_action_str = self.actions[action_idx]
        
        if chosen_action_str.endswith("_MEC"):
            base_act = chosen_action_str.replace("_MEC", "")
            return (base_act, "ACTIVATE")
        else:
            return (chosen_action_str, None)

    def decay_epsilon(self, battle_count=None):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _get_root_path(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, filename)

    def save_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        data = {
            "q_table": self.q_table,
            "visit_counts": getattr(self, 'visit_counts', {}),
            "epsilon": self.epsilon,
            "current_phase": getattr(self, 'current_phase', 'maxdamage')
        }
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
                    self.visit_counts = data.get("visit_counts", {}) 
                    self.epsilon = data.get("epsilon", self.epsilon)
                    self.current_phase = data.get("current_phase", "maxdamage")
                    print(f"[CÉREBRO] Brain Carregado. Estados: {len(self.q_table)} | Fase: {self.current_phase.upper()}")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar modelo: {e}")