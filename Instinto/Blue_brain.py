import numpy as np
import pickle
import os
import random
import threading
from collections import deque

from instinct_core import MatchupState

class BlueBrain:
    def __init__(self, alpha=0.2, gamma=0.90, epsilon=0.50, min_epsilon=0.10, decay=0.985):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        self.visit_counts = {}  
        
        self._qtable_lock = threading.Lock()
        
        # --- NOVO: EXPERIENCE REPLAY (DYNA-Q) ---
        self.memory = deque(maxlen=50000) # Guarda as últimas 50 mil transições na RAM
        self.batch_size = 128             # Quantidade de memórias processadas em cada "sonho"
        
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
            # Decay Base: 0.005 por bloco. (-0.30 em 60 blocos = 30.000 batalhas exatas)
            "maxdamage": {"epsilon_start": 0.40, "epsilon_min": 0.10, "decay": 0.005}, 
            "instinct":  {"epsilon_start": 0.40, "epsilon_min": 0.10, "decay": 0.005}, 
            # Selfplay: cai de 0.20 para 0.02 (-0.18) em 60 blocos = 0.003
            "selfplay":  {"epsilon_start": 0.20, "epsilon_min": 0.02, "decay": 0.003}  
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

        # --- 1. RECOMPENSAS MACRO (APENAS NA TRANSIÇÃO DE MORTE) ---
        current_my_fainted = len([m for m in battle.team.values() if m.fainted])
        current_opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        
        my_fainted_prev = history.get('my_fainted', 0)
        
        if current_my_fainted > my_fainted_prev: reward -= 100.0
        if current_opp_fainted > history.get('opp_fainted', 0): reward += 100.0

        # --- 2. MICRO-RECOMPENSAS DE HP E TROCAS ---
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_hp_prev = history.get('my_hp_prev', 0.0)
        opp_hp_prev = history.get('opp_hp_prev', 0.0)
        my_hp_curr = active.current_hp_fraction if active else 0.0
        opp_hp_curr = opp.current_hp_fraction if opp else 0.0

        my_hp_delta = my_hp_curr - my_hp_prev
        opp_hp_delta = opp_hp_curr - opp_hp_prev

        prev_my_species = history.get('my_species')
        curr_my_species = active.species if active else None

        # --- PUNIÇÃO E RECOMPENSA DE POSICIONAMENTO (TROCAS) ---
        if prev_my_species and curr_my_species and prev_my_species != curr_my_species:
            if current_my_fainted == my_fainted_prev:
                
                # Punição por jogar Buff fora
                my_boosts_prev = history.get('my_boosts', {})
                positive_boosts = sum(v for v in my_boosts_prev.values() if v > 0)
                if positive_boosts > 0:
                    reward -= 15.0 * positive_boosts
                    
                # Recompensa de Matchup (Compensa o dano tomado na entrada)
                prev_matchup = history.get('matchup')
                curr_matchup = getattr(self, 'core', None)
                if curr_matchup and prev_matchup:
                    try:
                        curr_matchup_val = self.core.get_matchup_state(active, opp) if active and opp else None
                        if curr_matchup_val and prev_matchup != curr_matchup_val:
                            matchup_weights = {
                                MatchupState.DOMINANT: 4.0, MatchupState.OFFENSIVE_ADV: 2.0,
                                MatchupState.DEFENSIVE_ADV: 1.0, MatchupState.VOLATILE: 0.0,
                                MatchupState.NEUTRAL: 0.0, MatchupState.STALEMATE: 0.0,
                                MatchupState.OFFENSIVE_DIS: -2.0, MatchupState.DEFENSIVE_DIS: -3.0,
                                MatchupState.CRITICAL_DIS: -4.0
                            }
                            weight_delta = matchup_weights.get(curr_matchup_val, 0.0) - matchup_weights.get(prev_matchup, 0.0)
                            
                            if weight_delta > 0:
                                reward += weight_delta * 10.0  
                            elif weight_delta < 0:
                                reward += weight_delta * 10.0  
                    except: pass

        # --- AJUSTE DE ESCALA: Máximo de 25 pontos por barra de vida ---
        if prev_my_species and curr_my_species and prev_my_species == curr_my_species:
            if my_hp_delta < 0: 
                reward -= 25.0 * abs(my_hp_delta)
            elif my_hp_delta > 0: 
                # Cura mantida estritamente menor para coibir loops de farm
                reward += 10.0 * my_hp_delta

        prev_opp_species = history.get('opp_species')
        curr_opp_species = opp.species if opp else None

        if prev_opp_species and curr_opp_species and prev_opp_species == curr_opp_species:
            if opp_hp_delta < 0: 
                reward += 25.0 * abs(opp_hp_delta)

        # --- 3. RECOMPENSAS ÚNICAS DE AÇÃO (TRANSIÇÕES ATIVAS) ---
        last_action_tuple = history.get('last_action')
        base_action = last_action_tuple[0] if last_action_tuple else None

        curr_opp_status = "AFFLICTED" if opp and opp.status else "CLEAN"
        if history.get('opp_status') == "CLEAN" and curr_opp_status == "AFFLICTED":
            reward += 15.0 

        if base_action == "HAZARD":
            reward += 15.0
        elif base_action == "CLEAN_HAZARD":
            reward += 15.0

        return reward

    def _get_abstract_state(self, state):
        """Remove a dimensão de Mecânica (MEC) da chave de aprendizado para generalizar a tabela."""
        if state in [("TERMINAL_WIN",), ("TERMINAL_LOSS",)]:
            return state
        if isinstance(state, tuple) and len(state) == 14:
            # Substitui o MEC_AVAIL ou MEC_USED por MEC_ANY
            return state[:-1] + ("MEC_ANY",)
        return state

    def update_feedback(self, current_state, last_state, last_action_tuple, reward):
        """Recebe o feedback do Agente, salva na memória e faz o update em tempo real."""
        if last_state is not None and last_action_tuple is not None:
            base_action, mechanic = last_action_tuple
            action_str = f"{base_action}_MEC" if mechanic else base_action
            
            if action_str in self.actions:
                self.memory.append((last_state, action_str, reward, current_state))
                # Passamos is_replay=False porque isso aconteceu no mundo real
                self._apply_q_update(last_state, action_str, reward, current_state, is_replay=False)

    def _apply_q_update(self, state, action_str, reward, next_state, is_replay=False):
        """Núcleo matemático do Q-Learning isolado."""
        abs_state = self._get_abstract_state(state)
        abs_next = self._get_abstract_state(next_state)

        with self._qtable_lock:
            if abs_state not in self.q_table:
                self.q_table[abs_state] = np.zeros(len(self.actions))
                self.visit_counts[abs_state] = 0

            # --- CORREÇÃO DO TEMPO: O Sonho não envelhece o estado ---
            if not is_replay:
                self.visit_counts[abs_state] = self.visit_counts.get(abs_state, 0) + 1

            visits = self.visit_counts.get(abs_state, 1)
            
            # Alpha dinâmico baseado APENAS em visitas reais
            if visits <= 1:
                effective_alpha = min(0.5, self.alpha * 2.0)
            else:
                effective_alpha = max(0.05, self.alpha / (1 + 0.05 * visits))
            
            action_idx = self.actions.index(action_str)
            old_val = self.q_table[abs_state][action_idx]
            
            if abs_next in [("TERMINAL_WIN",), ("TERMINAL_LOSS",)]:
                next_max = 0.0
            else:
                if abs_next not in self.q_table:
                    self.q_table[abs_next] = np.zeros(len(self.actions))
                    self.visit_counts[abs_next] = 0
                next_max = np.max(self.q_table[abs_next])
            
            new_val = (1 - effective_alpha) * old_val + effective_alpha * (reward + self.gamma * next_max)
            self.q_table[abs_state][action_idx] = new_val

    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return 
        
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action_str, reward, next_state in minibatch:
            # Passamos is_replay=True para a IA saber que é apenas um sonho
            self._apply_q_update(state, action_str, reward, next_state, is_replay=True)

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
        
        # 1. Verifica no estado REAL se a mecânica está disponível
        is_mec_avail = False
        if isinstance(state, tuple) and len(state) > 0:
            is_mec_avail = (state[-1] == "MEC_AVAIL")

        # 2. Converte para o estado ABSTRATO para consultar a Q-Table
        abs_state = self._get_abstract_state(state)

        if abs_state not in self.q_table:
            self.q_table[abs_state] = np.zeros(len(self.actions))
            self.visit_counts[abs_state] = 0
            
        # --- Herança _MEC para estados Novos E Existentes (Usando abs_state) ---
        for i, act in enumerate(self.actions):
            if act.endswith("_MEC"):
                base = act.replace("_MEC", "")
                if base in self.actions:
                    base_idx = self.actions.index(base)
                    if self.q_table[abs_state][i] == 0.0 and self.q_table[abs_state][base_idx] != 0.0:
                        self.q_table[abs_state][i] = self.q_table[abs_state][base_idx]

        primary_intent, _, ranking_list, candidate_mask = instinct_profile
        
        valid_indices = []
        for act in candidate_mask:
            if act in self.base_actions:
                valid_indices.append(self.actions.index(act)) 
                # Usa a flag do estado REAL para liberar ou não a ação _MEC
                if is_mec_avail and "SWITCH" not in act:
                    valid_indices.append(self.actions.index(f"{act}_MEC"))

        if not valid_indices: 
            return (primary_intent, None)
            
        # Busca os Q-Values usando o abs_state
        valid_q_values = {idx: self.q_table[abs_state][idx] for idx in valid_indices}
        best_action_idx = max(valid_q_values, key=valid_q_values.get)
        worst_action_idx = min(valid_q_values, key=valid_q_values.get)
        
        best_q_value = valid_q_values[best_action_idx]
        worst_q_value = valid_q_values[worst_action_idx]

        visits = self.visit_counts.get(abs_state, 0)
        
        if visits == 0 or (best_q_value == 0.0 and worst_q_value == 0.0):
            try:
                action_idx = self.actions.index(primary_intent)
                # Garante que a ação pedida pelo Mestre é viável neste turno
                if action_idx not in valid_indices:
                    # Se o mestre pediu "ATTACK_STRONG" mas o jogo exige a Mecânica, faz o ajuste fino
                    mec_str = f"{primary_intent}_MEC"
                    if mec_str in self.actions and self.actions.index(mec_str) in valid_indices:
                        action_idx = self.actions.index(mec_str)
                    else:
                        action_idx = best_action_idx
            except ValueError:
                action_idx = best_action_idx
        else:
            # --- O ALUNO ASSUME: Agora que ele tem experiência, decide se vai explorar ---
            if visits < 5:
                # Teste cauteloso: O Aluno testa ideias novas, mas com Epsilon 20% menor que o global
                state_epsilon = min(0.30, self.epsilon * 0.8)
            elif visits < 20:
                # Fase de consolidação: Exploração normal ditada pelo bloco de treino (ex: 40%)
                state_epsilon = self.epsilon
            else:
                # Pânico Escalonado e Incerteza Matemática Normalizada para estados já maduros
                spread = best_q_value - worst_q_value
                
                # --- A ÚNICA CORREÇÃO FALTANDO: SPREAD NORMALIZADO ---
                max_abs_q = max(abs(best_q_value), abs(worst_q_value), 1.0)
                normalized_spread = spread / max_abs_q
                
                # Se a diferença entre o melhor e pior for menor que 10% (0.1), é pânico total.
                if best_q_value < 0 and normalized_spread < 0.1:
                    uncertainty_factor = 1.0  
                elif best_q_value < 0:
                    uncertainty_factor = 0.5  
                else:
                    # Agora a matemática funciona como porcentagem!
                    uncertainty_factor = np.exp(-normalized_spread * 10.0) 
                    # Garante que NUNCA caia a 0 absoluto (mantém curiosidade mínima)
                    uncertainty_factor = max(0.1, uncertainty_factor)
                    
                state_epsilon = min(0.5, self.epsilon * uncertainty_factor)

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
                
                if layer_roll < 0.55:
                    pool = safe_indices
                # 30% do tempo testa setups ou trocas arriscadas
                elif layer_roll < 0.90:
                    pool = risky_indices if risky_indices else safe_indices
                # Apenas 5% do tempo ele pode fazer algo completamente aleatório
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
        if not self.visit_counts:
            return

        # 1. Pega os 5% dos estados mais visitados (O "Core" do metagame)
        # Ignoramos a massa de estados com 1 visita para não distorcer a média
        counts = list(self.visit_counts.values())
        counts.sort(reverse=True)
        top_5_percent_idx = max(1, int(len(counts) * 0.05))
        top_counts = counts[:top_5_percent_idx]

        avg_top_visits = sum(top_counts) / len(top_counts)

        # 2. Multiplicador de Maturidade Orgânica (Varia de 0.5x a 1.5x)
        # Se a média core for 10 visitas, o multiplicador é 1.0 (Queda exata no alvo).
        maturity_multiplier = min(1.5, max(0.5, avg_top_visits / 10.0))

        # 3. Subtração Real
        actual_decay = self.epsilon_decay * maturity_multiplier
        self.epsilon = max(self.min_epsilon, self.epsilon - actual_decay)

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
                    
                    saved_phase = data.get("current_phase", "maxdamage")

                    if saved_phase == self.current_phase:
                        self.epsilon = data.get("epsilon", self.epsilon)
                    else:
                        # Mantém o 40% fresquinho que a enter_phase gerou.
                        print(f"[CÉREBRO] Nova fase detectada ({saved_phase} -> {self.current_phase})! Epsilon resetado.")
                        
                    print(f"[CÉREBRO] Brain Carregado. Estados: {len(self.q_table)} | Fase: {self.current_phase.upper()}")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar modelo: {e}")