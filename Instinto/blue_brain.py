import numpy as np
import pickle
import os
import random
import threading
from collections import deque

class BlueBrain:
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=0.40, min_epsilon=0.05, decay=0.005):
        self.initial_alpha = alpha
        self.min_alpha = 0.005
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = epsilon 
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        self.visit_counts = {}  
        self._qtable_lock = threading.Lock()

        self.memory = deque(maxlen=10000) 
        self.batch_size = 512
        
        # --- ESTRUTURA DE AÇÕES ATUALIZADA (V7) ---
        self.base_actions = [
            "ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH", 
            "BUFF", "STATUS", "HEAL", "CLEAN_HAZARD", 
            "PROTECT", "DEBUFF", "DISRUPTION", "STAT_CLEAN", "HEAL_STATUS", "PHAZE",
            "FIELD_CONTROL", "HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE",
            "BARRIER"
        ]
        
        # Constrói a lista expandida para a Q-Table (Ação + Ação_MEC)
        self.actions = []
        for act in self.base_actions:
            self.actions.append(act)
            if "SWITCH" not in act:
                self.actions.append(f"{act}_MEC")
        
        self.current_phase = "maxdamage"

    def inspect_brain(self):
        """Analisa a saúde e a convergência da Tabela Q."""
        total_states = len(self.q_table)
        if total_states == 0:
            return 0, 0.0, 0.0

        total_visits = 0
        confident_states = 0
        confidence_threshold = 3 

        for state, actions in self.q_table.items():
            # Conta as visitas da estrutura dinâmica da Tabela Q (array ou dict)
            if isinstance(actions, dict):
                state_visits = sum(action_data.get('visits', 0) for action_data in actions.values())
            else:
                # Se for lista do Q-Learning puro, puxamos do dicionário de visitas
                state_visits = self.visit_counts.get(state, 1)
                
            total_visits += state_visits
            if state_visits >= confidence_threshold:
                confident_states += 1

        avg_visits = total_visits / total_states
        confidence_rate = (confident_states / total_states) * 100.0

        return total_visits, avg_visits, confidence_rate

    def enter_phase(self, phase_name):
        phase_config = {
            # Fase 1: Aprendizado bruto (Alta exploração, Alta absorção)
            "maxdamage": {"epsilon_start": 0.40, "epsilon_min": 0.05, "decay": 0.005, "alpha_start": 0.15, "alpha_min": 0.005}, 
            # Fase 2: Adaptação estratégica (Exploração média, Absorção média)
            "instinct":  {"epsilon_start": 0.40, "epsilon_min": 0.03, "decay": 0.002, "alpha_start": 0.15, "alpha_min": 0.005}, 
            # Fase 3: Lapidação (Exploração baixa, Absorção baixa)
            "selfplay":  {"epsilon_start": 0.30, "epsilon_min": 0.01, "decay": 0.002, "alpha_start": 0.10, "alpha_min": 0.001}
        }
        
        if phase_name in phase_config:
            cfg = phase_config[phase_name]
            
            # Reseta o Epsilon
            self.initial_epsilon = cfg["epsilon_start"]
            self.epsilon = cfg["epsilon_start"]
            self.min_epsilon = cfg["epsilon_min"]
            self.epsilon_decay = cfg["decay"]
            
            # Reseta o Alpha
            self.initial_alpha = cfg["alpha_start"]
            self.min_alpha = cfg["alpha_min"]
            self.alpha = self.initial_alpha
            
            self.current_phase = phase_name
            self.memory.clear() 
            print(f"[CÉREBRO] Entrando na Fase: {phase_name.upper()} | Eps: {self.epsilon:.2f} -> {self.min_epsilon:.2f} | Alpha: {self.alpha:.3f}")

    def _calculate_potential(self, battle, state):
        """
        Calcula o Potencial Total (Phi) de um estado dado.
        Teoria: Phi = Phi_Guerra (Macro) + Phi_Batalha (Micro)
        """
        if not state or len(state) < 15: return 0.0
        
        phi_guerra = 0.0
        phi_batalha = 0.0
        
        # --- PILAR A: CONTEXTO E MATCHUP ---
        matchup_vals = {
            "DOMINANT": 40.0, "OFFENSIVE_ADV": 20.0, "DEFENSIVE_ADV": 10.0,
            "NEUTRAL": 0.0, "VOLATILE": -5.0, "STALEMATE": 0.0,
            "OFFENSIVE_DIS": -10.0, "DEFENSIVE_DIS": -20.0, "CRITICAL_DIS": -40.0
        }
        matchup = str(state[2]).upper()
        phi_batalha += matchup_vals.get(matchup, 0.0)
        
        macro_context = str(state[14]).upper()
        context_vals = {"DOMINATING": 30.0, "RECOVERING": -30.0}
        phi_guerra += context_vals.get(macro_context, 0.0)

        # --- PILAR B: VANTAGEM MATERIAL (HP GLOBAL) ---
        my_total_hp = sum(m.current_hp_fraction for m in battle.team.values())
        opp_total_hp = sum(m.current_hp_fraction for m in battle.opponent_team.values())
        # A diferença de HP (Guerra) multiplicada por um peso de escala
        phi_guerra += (my_total_hp - opp_total_hp) * 40.0 

        # --- PILAR C: CONTROLE DE CAMPO (HAZARDS E CLIMA) ---
        # Nosso Campo (Hazards são ruins para nós)
        if state[11] == "SET": phi_guerra -= 10.0
        # Campo Inimigo (Hazards são bons para nós)
        if state[12] == "SET": phi_guerra += 10.0
        
        weather_state = str(state[5]).upper()
        field_vals = {"FIELD_SWEEP": 20.0, "FIELD_POWER": 15.0, "FIELD_SPEED": 15.0, 
                      "FIELD_DEFENSE": 10.0, "FIELD_HOSTILE": -20.0}
        phi_guerra += field_vals.get(weather_state, 0.0)

        # --- INTEGRAÇÃO DE ROLES E SPEED (INICIATIVA) ---
        my_role = str(state[0]).upper()
        speed_tier = str(state[6]).upper()
        
        if speed_tier == "FASTER":
            phi_batalha += 20.0 if my_role == "SWEEPER" else 5.0
        else: # SLOWER
            phi_batalha -= 20.0 if my_role == "SWEEPER" else 0.0

        # --- BOOSTS, STATUS E MECÂNICAS ---
        my_boost = str(state[9]).upper()
        opp_boost = str(state[10]).upper()
        my_status = str(state[7]).upper()
        opp_status = str(state[8]).upper()
        mec_state = str(state[13]).upper()

        if "BUFFED" in my_boost: phi_batalha += 10.0
        if "DEBUFF" in my_boost: phi_batalha -= 10.0
        if my_status == "AFFLICTED": phi_batalha -= 15.0
        if opp_status == "AFFLICTED": phi_batalha += 15.0
        if "BUFFED" in opp_boost: phi_batalha -= 10.0
        if "DEBUFF" in opp_boost: phi_batalha += 10.0
        
        # Mecânica como recurso de pressão (Poder Guardado)
        if mec_state == "MEC_AVAIL": phi_batalha += 10.0

        return phi_guerra + phi_batalha


    def calculate_reward(self, battle, history, current_state=None):
        reward = 0.0

        # 1. RECOMPENSAS EXTERNAS (REALIDADE DO AMBIENTE)
        if battle.won: reward += 1500.0
        elif battle.lost: reward -= 1500.0

        current_my_fainted = len([m for m in battle.team.values() if m.fainted])
        current_opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        my_fainted_prev = history.get('my_fainted', 0)
        opp_fainted_prev = history.get('opp_fainted', 0)

        if current_my_fainted > my_fainted_prev: reward -= 100.0
        if current_opp_fainted > opp_fainted_prev: reward += 100.0

        # 2. POTENTIAL-BASED REWARD SHAPING (PBRS)
        # Calculamos a nota do estado atual
        phi_current = self._calculate_potential(battle, current_state)
        
        # Recuperamos a nota do estado anterior (salva pelo Agente no histórico)
        # Se for o primeiro turno, o potencial anterior é igual ao atual para dar 0
        phi_prev = history.get('last_phi', phi_current)
        
        # A Equação de Bellman para Shaping: F = (gamma * Phi_t+1) - Phi_t
        shaping_reward = (self.gamma * phi_current) - phi_prev
        
        # 3. INTEGRAÇÃO FINAL
        reward += shaping_reward
        
        # Salva o potencial atual para o próximo turno no histórico
        history['last_phi'] = phi_current

        return reward

    def _get_abstract_state(self, state):
        """A Tabela Q agora aprende exatamente QUANDO usar a mecânica."""
        if state in [("TERMINAL_WIN",), ("TERMINAL_LOSS",)]:
            return state
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
                effective_alpha = max(self.alpha, self.alpha / (1 + 0.05 * visits))

            if is_replay:
                effective_alpha *= 0.2
            
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

    # =====================================================================
    # PRIORITIZED EXPERIENCE REPLAY (PER) - Motor de Sonho Inteligente
    # =====================================================================        
    def replay_experience(self):
        # 1. Dobramos a capacidade de processamento por turno
        self.batch_size = 512
        
        if len(self.memory) < self.batch_size:
            return 
        
        memory_list = list(self.memory)
        
        # 2. Otimização de Performance: Amostragem da Memória Recente
        # Usa a memória atual de 10k turnos como termômetro da ignorância da IA.
        single_visit_count = 0
        
        bucket_1_visit = []
        bucket_2_to_4 = []
        bucket_high_reward = []
        
        for m in memory_list:
            state, action_str, reward, next_state = m
            abs_state = self._get_abstract_state(state)
            visits = self.visit_counts.get(abs_state, 0)
            
            # Distribuição mútua
            if visits <= 1:
                single_visit_count += 1
                bucket_1_visit.append(m)
            elif 2 <= visits <= 4:
                bucket_2_to_4.append(m)
                
            # Alto Impacto Tático (+80 de Potencial ou Dano Massivo)
            if abs(reward) >= 80.0:
                bucket_high_reward.append(m)

        # 3. O Termômetro da Ignorância (A Regra de 30%)
        single_visit_ratio = single_visit_count / len(memory_list)
        
        if single_visit_ratio > 0.30:
            # MODO FAXINA: Há muito desconhecimento na área
            target_1v = int(self.batch_size * 0.70)
            target_2to4 = int(self.batch_size * 0.20)
            target_high = self.batch_size - target_1v - target_2to4 # ~10%
        else:
            # MODO CONSOLIDAÇÃO: O terreno já é familiar
            target_1v = int(self.batch_size * 0.20)
            target_2to4 = int(self.batch_size * 0.60)
            target_high = self.batch_size - target_1v - target_2to4 # ~20%
            
        # 4. Amostragem de Segurança (Garante que o lote encha sem quebrar por falta de dados)
        batch = []
        
        def sample_bucket(bucket, target_size):
            available = len(bucket)
            if available == 0: return []
            take = min(target_size, available)
            return random.sample(bucket, take)
            
        batch.extend(sample_bucket(bucket_1_visit, target_1v))
        batch.extend(sample_bucket(bucket_2_to_4, target_2to4))
        batch.extend(sample_bucket(bucket_high_reward, target_high))
        
        # Se os baldes acima não preencherem os 256 slots, completamos com a memória geral
        missing = self.batch_size - len(batch)
        if missing > 0:
            batch.extend(random.sample(memory_list, min(missing, len(memory_list))))
            
        # Embaralha para que a Q-Table não crie viés da ordem de atualização
        random.shuffle(batch)
        
        # 5. Execução do Sonho
        for state, action_str, reward, next_state in batch:
            self._apply_q_update(state, action_str, reward, next_state, is_replay=True)

    def _add_action_indices(self, intent, target_list, valid_indices, is_mec_avail):
        #Adiciona índices da ação base e _MEC (se disponível) à lista, verificando duplicatas e validade.
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

    def decide_action(self, state, valid_actions, ranking_list):
        is_mec_avail = False
        if isinstance(state, tuple) and len(state) >= 14:
            is_mec_avail = (state[13] == "MEC_AVAIL")

        abs_state = self._get_abstract_state(state)
        
        if abs_state not in self.q_table:
            self.q_table[abs_state] = [0.0] * len(self.actions)
            
        # --- A BRILHANTE HERANÇA DE MECÂNICA (Restaurada da V1) ---
        for i, act in enumerate(self.actions):
            if act.endswith("_MEC"):
                base = act.replace("_MEC", "")
                if base in self.actions:
                    base_idx = self.actions.index(base)
                    if self.q_table[abs_state][i] == 0.0 and self.q_table[abs_state][base_idx] != 0.0:
                        self.q_table[abs_state][i] = self.q_table[abs_state][base_idx]

        valid_indices = [self.actions.index(a) for a in valid_actions]
        valid_q_values = {idx: self.q_table[abs_state][idx] for idx in valid_indices}
        
        best_action_idx = max(valid_q_values, key=valid_q_values.get)
        worst_action_idx = min(valid_q_values, key=valid_q_values.get)
        best_q_value = valid_q_values[best_action_idx]
        worst_q_value = valid_q_values[worst_action_idx]

        visits = self.visit_counts.get(abs_state, 0)

        # Expandimos o Ranking do Instinto para incluir as opções de Mecânica
        valid_ranked = []
        for intent in ranking_list:
            if intent in valid_actions and intent not in valid_ranked:
                valid_ranked.append(intent)
            mec_intent = f"{intent}_MEC"
            if mec_intent in valid_actions and mec_intent not in valid_ranked:
                valid_ranked.append(mec_intent)

        if not valid_ranked: valid_ranked = valid_actions

        # === 2. DINÂMICA MESTRE-ALUNO (ESTADO ZERO) ===
        if visits == 0 or (best_q_value == 0.0 and worst_q_value == 0.0):
            action_idx = self.actions.index(valid_ranked[0])
            
        else:
            # === 3. CÁLCULO DE INCERTEZA MATEMÁTICA (Restaurado da V1) ===
            if visits < 5:
                state_epsilon = min(0.30, self.epsilon * 0.8)
            elif visits < 20:
                state_epsilon = self.epsilon
            else:
                spread = best_q_value - worst_q_value
                max_abs_q = max(abs(best_q_value), abs(worst_q_value), 1.0)
                normalized_spread = spread / max_abs_q
                
                if best_q_value < 0 and normalized_spread < 0.1:
                    uncertainty_factor = 1.0  
                elif best_q_value < 0:
                    uncertainty_factor = 0.5  
                else:
                    uncertainty_factor = np.exp(-normalized_spread * 10.0) 
                    uncertainty_factor = max(0.1, uncertainty_factor)
                    
                state_epsilon = min(0.5, self.epsilon * uncertainty_factor)

            # === 4. EXPLORAÇÃO POR BUCKETS (A Genialidade da V2) ===
            if random.random() < state_epsilon:
                r = random.random()
                if r < 0.50 and len(valid_ranked) >= 2:
                    chosen_action = valid_ranked[0] if random.random() < 0.60 else valid_ranked[1]
                elif r < 0.85 and len(valid_ranked) >= 4:
                    chosen_action = valid_ranked[2] if random.random() < 0.60 else valid_ranked[3]
                else:
                    remaining = valid_ranked[4:] if len(valid_ranked) >= 5 else valid_ranked
                    chosen_action = random.choice(remaining) if remaining else valid_ranked[0]
                
                action_idx = self.actions.index(chosen_action)
            else:
                action_idx = best_action_idx

        # === 5. RETORNO COMPATÍVEL COM O AGENTE (Restaurado da V1) ===
        chosen_action_str = self.actions[action_idx]
        if chosen_action_str.endswith("_MEC"):
            base_act = chosen_action_str.replace("_MEC", "")
            return (base_act, "ACTIVATE")
        else:
            return (chosen_action_str, None)

    def decay_epsilon(self, new_states=0, battles_in_block=500):
        if not self.visit_counts:
            return

        # Taxa de descoberta: Quantos estados novos foram achados por batalha neste bloco
        discovery_rate = new_states / max(1, battles_in_block)

        # 1. CURVA ADAPTATIVA DE EXPLORAÇÃO
        decay_multiplier = 3.0 / max(discovery_rate, 0.1)
        
        # Limitamos a aceleração máxima a 3x a velocidade normal para que, mesmo no fim, não haja uma queda de penhasco.
        decay_multiplier = min(3.0, decay_multiplier)
        
        # 2. CÁLCULO E APLICAÇÃO DO DECAIMENTO
        actual_decay = self.epsilon_decay * decay_multiplier
        self.epsilon = max(self.min_epsilon, self.epsilon - actual_decay)

        # 3. DECAIMENTO DO ALPHA (Taxa de Aprendizado)
        initial_eps = getattr(self, 'initial_epsilon', 0.40)
        
        if self.epsilon > self.min_epsilon:
            progress = (self.epsilon - self.min_epsilon) / max(0.01, (initial_eps - self.min_epsilon))
        else:
            progress = 0.0
        
        self.alpha = max(self.min_alpha, self.min_alpha + (self.initial_alpha - self.min_alpha) * progress)

    def _get_root_path(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, filename)

    def save_model(self, filename="blue_brain.pkl"):
        filepath = self._get_root_path(filename)
        data = {
            "q_table": self.q_table,
            "visit_counts": getattr(self, 'visit_counts', {}),
            "epsilon": self.epsilon,
            "alpha": getattr(self, 'alpha', self.initial_alpha), # Salva o Alpha atual
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

                    # Só recuperamos os decaimentos se for a MESMA fase. 
                    # Se for diferente, deixamos os valores virgens carregados pelo enter_phase()
                    if saved_phase == self.current_phase:
                        self.epsilon = data.get("epsilon", self.epsilon)
                        self.alpha = data.get("alpha", getattr(self, 'alpha', self.initial_alpha))
                    else:
                        print(f"[CÉREBRO] Nova fase detectada ({saved_phase} -> {self.current_phase})! Epsilon e Alpha resetados com sucesso.")
                        
                    print(f"[CÉREBRO] Brain Carregado. Estados: {len(self.q_table)} | Fase: {self.current_phase.upper()}")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar modelo: {e}")