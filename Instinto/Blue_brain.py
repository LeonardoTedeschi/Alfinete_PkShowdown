import numpy as np
import pickle
import os
import random
import threading
from collections import deque

class BlueBrain:
    def __init__(self, alpha=0.2, gamma=0.90, epsilon=0.40, min_epsilon=0.05, decay=0.005):
        self.initial_alpha = alpha
        self.min_alpha = 0.005
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        self.visit_counts = {}  
        self._qtable_lock = threading.Lock()

        self.memory = deque(maxlen=10000) 
        self.batch_size = 128
        
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


    def calculate_reward(self, battle, history, current_state=None):
        reward = 0.0

        # --- 1. RECOMPENSAS MACRO (ABATES, MORTES E SACRIFÍCIOS) ---
        current_my_fainted = len([m for m in battle.team.values() if m.fainted])
        current_opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        
        my_fainted_prev = history.get('my_fainted', 0)
        opp_fainted_prev = history.get('opp_fainted', 0)
        
        # Recupera a memória de 2 turnos atrás
        my_fainted_prev_prev = history.get('my_fainted_prev_turn', 0)
        
        # A. Punição normal por morte
        if current_my_fainted > my_fainted_prev: 
            reward -= 100.0

        # B. Recompensa por Abate e A Arte do Sacrifício
        if current_opp_fainted > opp_fainted_prev: 
            reward += 100.0 
            
            # C. BÔNUS DE REVENGE KILL ("Boi de Piranha")
            if my_fainted_prev > my_fainted_prev_prev:
                reward += 50.0 # Bônus brutal compensa o sacrifício e ensina o bot a preparar o terreno
                
            # D. BÔNUS DE KAMIKAZE (Double KO estratégico)
            elif current_my_fainted > my_fainted_prev:
                reward += 30.0 # Recompensa por se sacrificar para levar uma ameaça junto

        # --- 2. MICRO-RECOMPENSAS (SOMA GLOBAL DE HP) ---
        
        # 2A. Delta da Nossa Equipe (Dano Sofrido)
        team_hp_history = history.get('team_hp', {})
        my_current_team_hp = sum(m.current_hp_fraction for m in battle.team.values())
        my_prev_team_hp = sum(team_hp_history.values()) if team_hp_history else my_current_team_hp
        my_team_delta = my_current_team_hp - my_prev_team_hp
        
        if current_my_fainted == my_fainted_prev:
            if my_team_delta < 0:
                reward -= 3.0 * abs(my_team_delta)

        # 2B. Delta da Equipe Adversária (Dano Causado)
        opp_team_hp_history = history.get('opp_team_hp', {})
        opp_current_team_hp = sum(m.current_hp_fraction for m in battle.opponent_team.values())
        opp_prev_team_hp = sum(opp_team_hp_history.values()) if opp_team_hp_history else opp_current_team_hp
        opp_team_delta = opp_current_team_hp - opp_prev_team_hp

        if current_opp_fainted == opp_fainted_prev:
            if opp_team_delta < 0: 
                reward += 5.0 * abs(opp_team_delta) 

        # Recupera a ação e o estado anterior para as análises abaixo
        last_action_tuple = history.get('last_action')
        base_action = last_action_tuple[0] if last_action_tuple else None
        last_state = history.get('state', [])

        # Recupera a ação e o estado anterior para as análises abaixo
        last_action_tuple = history.get('last_action')
        base_action = last_action_tuple[0] if last_action_tuple else None
        last_state = history.get('state', [])

        # --- CORREÇÃO: EXTRAINDO O MACRO CONTEXT DO ESTADO ---
        macro_context = "BRAWL" # Valor padrão de segurança
        if last_state and len(last_state) >= 15:
            macro_context = last_state[14]

       # --- 3. PEDÁGIO DE TROCA E DESPERDÍCIO ---
        active = battle.active_pokemon
        curr_my_species = active.species if active else None
        prev_my_species = history.get('my_species')

        if prev_my_species and curr_my_species and prev_my_species != curr_my_species:
            if current_my_fainted == my_fainted_prev: 
                # 1. Pedágio Fixo por qualquer troca voluntária
                if base_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"]:
                    reward -= 5.0
                    
                    # 2. Punição por trocas consecutivas (Fadiga/Loop)
                    prev_action = history.get('prev_action')
                    if prev_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]:
                        reward -= 10.0
                    
                # 3. Punição por desperdício de status positivo (Buff) na troca
                if last_state and len(last_state) >= 10:
                    my_last_boost_state = str(last_state[9]).upper()
                    if "BUFFED" in my_last_boost_state and "DEBUFF" not in my_last_boost_state:
                        reward -= 15.0

        # --- 4. A GUILHOTINA TÁTICA (Punições por Redundância) ---
        opp = battle.opponent_active_pokemon
        
        if base_action == "HAZARD":
            # Puxa o estado atual dos Hazards no campo do oponente
            opp_conds = [str(k).upper() for k in battle.opponent_side_conditions.keys()]
            already_set = any(h in opp_conds for h in ['STEALTH_ROCK', 'SPIKES', 'TOXIC_SPIKES', 'STICKY_WEB'])
            
            if already_set:
                reward -= 10.0
            else:
                if macro_context == "OPENING": reward += 5.0
                elif macro_context == "BRAWL": reward += 2.0

        elif base_action == "STATUS":
            # Se o oponente atual JÁ ESTÁ com status e tentamos aplicar novamente
            curr_opp_status = "AFFLICTED" if opp and opp.status else "CLEAN"
            prev_opp_status = history.get('opp_status', "CLEAN")
            
            if prev_opp_status == "AFFLICTED":
                reward -= 10.0 # Reduzido: Ação de status redundante
            elif curr_opp_status == "AFFLICTED":
                if macro_context == "OPENING": reward += 5.0
                elif macro_context == "BRAWL": reward += 3.0
                else: reward += 1.0

        # --- 5. RECOMPENSAS TEMPORAIS E DE MACRO-ESTRATÉGIA ---
        elif base_action == "CLEAN_HAZARD":
            if macro_context in ["OPENING", "BRAWL"]: reward += 4.0
            else: reward += 1.0

        elif base_action == "PHAZE":
            if macro_context in ["OPENING", "BRAWL", "RECOVERING"]: reward += 3.0
            elif macro_context in ["CLUTCH", "DOMINATING"]: reward -= 5.0 

        elif base_action == "BUFF":
            if macro_context in ["CLUTCH", "DOMINATING", "RECOVERING"]: 
                reward += 5.0 
                
        elif base_action == "DISRUPTION":
            prev_action = history.get('prev_action', "")
            str_prev = str(prev_action[0]) if isinstance(prev_action, tuple) else str(prev_action)
            
            if "DISRUPTION" in str_prev:
                reward -= 15.0 # Punição severa: Desperdiçou o turno tentando renovar algo que já está ativo
            else:
                # Bônus Originais de Execução Perfeita
                opp_role = str(last_state[1]) if len(last_state) >= 2 else ""
                if opp_role == "TANK":
                    reward += 10.0 # Destruiu a função da Wall inimiga!
                if macro_context == "OPENING":
                    reward += 6.0  # Excelente para impedir Hazards e Setups iniciais
                
        elif base_action == "DEBUFF":
            # Screech, Charm, etc.
            reward += 2.0 # Recompensa base pequena por reduzir status
                
        elif base_action == "FIELD_CONTROL":
            # A IA SÓ PODE SER JULGADA PELO QUE ELA ENXERGA (last_state vs current_state)
            prev_field = str(last_state[5]).upper() if last_state and len(last_state) >= 6 else "NORMAL"
            
            # Pega a nova abstração de campo (Se o bot morreu no turno, assume NORMAL)
            curr_field = "NORMAL"
            if current_state and len(current_state) >= 6:
                curr_field = str(current_state[5]).upper()
                
            positive_fields = ["FIELD_POWER", "FIELD_SPEED", "FIELD_DEFENSE", "FIELD_SWEEP"]
            
            # O Julgamento Matemático Purista
            if curr_field in positive_fields:
                if prev_field == "FIELD_HOSTILE":
                    reward += 15.0 # Mestre: Reverteu o clima inimigo e ganhou vantagem tática
                else:
                    reward += 10.0 # Ótimo: Criou vantagem do zero
                    
            elif curr_field == "FIELD_NEUTRAL":
                if prev_field == "FIELD_HOSTILE":
                    reward += 5.0  # Tático de Defesa: Limpou o clima hostil, mesmo que não ganhe bônus direto
                else:
                    reward -= 2.0  # Ruído: Alterou o campo, mas o próprio Cérebro não sabe como usar isso
                    
            elif curr_field == "FIELD_HOSTILE":
                reward -= 10.0     # Suicídio Tático: O Cérebro usou um golpe que piorou a própria situação!
                
            else:
                reward -= 8.0      # Falha: O oponente impediu (Taunt), errou, ou sobrescreveu no mesmo turno

        elif base_action == "PROTECT":
            prev_action = history.get('prev_action', "")
            str_prev = str(prev_action[0]) if isinstance(prev_action, tuple) else str(prev_action)
            
            if "PROTECT" in str_prev:
                reward -= 15.0 
            else:
                # Recompensa moderada por usar o Protect de forma inteligente (Scout)
                reward += 3.0
                
        elif base_action == "HEAL_STATUS":
            # Punição por desperdiçar turno se o time não estiver curado
            reward += 8.0 # Recompensa padrão por curar o time
                
        elif base_action in ["HEAL"]:
            # NOVA PUNIÇÃO: Cura desnecessária (Overheal) com a vida no bucket FULL
            if base_action == "HEAL" and history.get('my_hp_bucket') == "FULL":
                reward -= 3.0
            else:
                if macro_context in ["BRAWL", "RECOVERING", "CLUTCH"]: reward += 1.5
                else: reward += 0.5

        had_lethal = history.get('has_lethal', False)
        if had_lethal and base_action not in ["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_TECH", "ATTACK_PIVOT"]:
            reward -= 30.0

        elif base_action == "STAT_CLEAN":
            my_boost = str(last_state[9]).upper() if len(last_state) >= 10 else "NEUTRAL"
            opp_boost = str(last_state[10]).upper() if len(last_state) >= 11 else "NEUTRAL"
            
            if "DEBUFF" not in my_boost and "BUFFED" not in opp_boost:
                reward -= 8.0 # Punição: Usou Haze num cenário neutro
            else:
                reward += 5.0  # Genial: Limpou um Sweeper inimigo ou curou nossos drops

        # --- 5.5 RECOMPENSA DE MOMENTUM ---
        prev_action = history.get('prev_action')
        if prev_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE", "ATTACK_PIVOT"]:
            if base_action in ["ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_TECH", "ATTACK_PIVOT", "STATUS", "HAZARD", "BUFF"]:
                reward += 8.0
            elif base_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"]:
                reward -= 10.0 

        # --- 5.6 SINERGIA DE CAMPO (Aproveitando a Vantagem) ---
        if last_state and len(last_state) >= 6:
            field_context = str(last_state[5]).upper()
            
            if field_context == "FIELD_POWER" and base_action in ["ATTACK_STRONG", "ATTACK_PREDICTIVE"]:
                reward += 4.0 # Sinergia: Usou o bônus matemático de dano para esmagar o oponente
                
            elif field_context == "FIELD_SPEED" and base_action not in ["SWITCH_DEFENSIVE", "PROTECT"]:
                reward += 3.0 # Sinergia: Usou a vantagem de turno garantido para agir ativamente
                
            elif field_context == "FIELD_DEFENSE" and base_action in ["BUFF", "HAZARD", "HEAL", "STATUS"]:
                reward += 4.0 # Sinergia: Usou a Evasão/Cura do campo para fazer setup com segurança absoluta

        # --- 6. O XEQUE-MATE ---
        if battle.won: 
            reward += 1500.0
        elif battle.lost: 
            reward -= 1500.0

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

    def replay_experience(self):
        if len(self.memory) < self.batch_size:
            return 
        
        memory_list = list(self.memory)
        
        # --- VIÉS DE RECÊNCIA (RECENT-BIASED SAMPLING) ---
        recent_cutoff = int(len(memory_list) * 0.66)
        recent_pool = memory_list[recent_cutoff:]
        old_pool = memory_list[:recent_cutoff]
        
        recent_batch_size = int(self.batch_size * 0.75)
        old_batch_size = self.batch_size - recent_batch_size
        
        if len(recent_pool) >= recent_batch_size and len(old_pool) >= old_batch_size:
            minibatch = random.sample(recent_pool, recent_batch_size) + random.sample(old_pool, old_batch_size)
        else:
            minibatch = random.sample(memory_list, self.batch_size)
            
        for state, action_str, reward, next_state in minibatch:
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

    def decay_epsilon(self, battle_count=None):
        if not self.visit_counts:
            return

        # 1. Pega os 5% dos estados mais visitados (O "Core" do metagame)
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

        # 4. DECAIMENTO DO ALPHA
        initial_eps = 0.40 
        if self.epsilon > self.min_epsilon:
            progress = (self.epsilon - self.min_epsilon) / (initial_eps - self.min_epsilon)
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