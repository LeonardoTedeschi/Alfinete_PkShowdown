import numpy as np
import pickle
import os
import random

class BlueBrain:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.40, min_epsilon=0.05, decay=0.99998):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = decay
        
        self.q_table = {}
        
        # V_6: Lista expandida para 18 ações (Ações táticas ramificadas)
        self.actions = [
            "ATTACK_STRONG", "ATTACK_PREDICTIVE", "ATTACK_PIVOT", "ATTACK_TECH", 
            "ATTACK_MEC", "BUFF", "STATUS", "HEAL", "CLEAN_HAZARD", 
            "PROTECT", "DEBUFF", "STAT_CLEAN", "HEAL_STATUS", "PHAZE", 
            "FIELD_CONTROL", "HAZARD", "SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"
        ]

        self.load_model("blue_brain.pkl")

    def calculate_reward(self, battle, history):
        reward = 0.0

        # 1. OBJETIVO MACRO (Vitoria/Derrota)
        if battle.won: return 2000.0
        if battle.lost: return -3000.0

        # 2. VANTAGEM MATERIAL (Abates e Perdas - Valor Moderado)
        current_my_fainted = len([m for m in battle.team.values() if m.fainted])
        current_opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        
        # Conforme sua sugestão, 200 pontos para não ofuscar o resultado final
        if current_my_fainted > history.get('my_fainted', 0): reward -= 200.0
        if current_opp_fainted > history.get('opp_fainted', 0): reward += 200.0

        last_action = history.get('last_action')
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        core = self.load_instinct_core_temp()

        if last_action in ["SWITCH_DEFENSIVE", "SWITCH_OFFENSIVE"] and active:
            reward -= 15.0  # Pedágio: Custou um turno
        

        # 4. MICRO-RECOMPENSAS ESTRATÉGICAS (Guia de aprendizado)
        # Aplicar Status
        curr_opp_status = "AFFLICTED" if opp and opp.status else "CLEAN"
        if history.get('opp_status') == "CLEAN" and curr_opp_status == "AFFLICTED":
            reward += 20.0 

        # Colocar Hazards
        if last_action == "HAZARD":
            reward += 20.0

        # Limpar Hazards do próprio campo
        curr_my_hazards = core.get_hazard_state(battle.side_conditions)
        if history.get('my_hazards') == "SET" and curr_my_hazards == "CLEAR":
            reward += 30.0

        # 5. PUNIÇÃO POR DANO CRÍTICO (Fobia de morrer)
        curr_my_hp_b = core.get_hp_bucket(active)
        if history.get('my_hp_bucket') == "FULL" and curr_my_hp_b in ["LOW", "CRIT"]:
            reward -= 30.0

        return reward

    def load_instinct_core_temp(self):
        from instinct_core import InstinctCore
        return InstinctCore()

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
        if state not in self.q_table:
            return instinct_intent

        available_indices = [self.actions.index(act) for act in available_actions if act in self.actions]
        if not available_indices: return instinct_intent 
        
        instinct_idx = self.actions.index(instinct_intent)
        valid_q_values = {idx: self.q_table[state][idx] for idx in available_indices}
        
        best_action_idx = max(valid_q_values, key=valid_q_values.get)
        best_q_value = valid_q_values[best_action_idx]
        q_instinct = valid_q_values.get(instinct_idx, 0.0)

        # =========================================================
        # ARQUITETURA MESTRE-ALUNO
        # =========================================================
        
        if best_q_value == 0.0:
            # FASE 1: PREENCHIMENTO RÁPIDO (Bootstrapping)
            # Estado novo. Sem testes cegos, sem derrotas burras. Segue o Mestre (Instinto).
            action_idx = instinct_idx
            
        else: 
            # Ajuste de Exploração Dinâmica: 
            # - Se a nota é negativa (Modo Desespero): Dobra o Epsilon.
            # - Se a nota é positiva (Zona de Conforto): Corta o Epsilon pela metade (Curiosidade cirúrgica).
            current_epsilon = min(0.99, self.epsilon * 2.0) if best_q_value < 0 else (self.epsilon * 0.5)

            if random.random() < current_epsilon:
                # CURIOSIDADE: Tenta algo novo que não seja a escolha óbvia.
                # Como a Matriz 4D já filtra lixo (via available_actions), qualquer teste aqui é "possível" de dar certo.
                options_to_explore = [i for i in available_indices if i != best_action_idx]
                action_idx = random.choice(options_to_explore) if options_to_explore else best_action_idx
            else:
                # SUPERAÇÃO ABSOLUTA: 
                # Confia na matemática. Se a melhor nota da tabela for maior que a do Instinto,
                # o Cérebro toma o controle. Se houver empate, respeita o Instinto.
                if best_q_value > q_instinct:
                    action_idx = best_action_idx
                else:
                    action_idx = instinct_idx

        # Decaimento progressivo
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
                    old_q_table = data.get("q_table", {})
                    
                    new_q_table = {}
                    for old_state, old_values in old_q_table.items():
                        # 1. ATUALIZA A TUPLA (13 para 14)
                        if len(old_state) == 13:
                            state_list = list(old_state)
                            if state_list[0] == "SPEED_SWEEPER": state_list[0] = "SWEEPER"
                            elif state_list[0] == "TANK_BULK": state_list[0] = "TANK"
                            if state_list[1] == "SPEED_SWEEPER": state_list[1] = "SWEEPER"
                            elif state_list[1] == "TANK_BULK": state_list[1] = "TANK"
                            
                            state_list.append("MEC_USED") 
                            new_state = tuple(state_list)
                        else:
                            new_state = old_state
                            
                        # 2. ATUALIZA A LISTA DE NOTAS (De 14 para 18 Ações - MUDANÇA V_6)
                        if len(old_values) == 14:
                            new_values = [0.0] * 18
                            
                            # Ramificação dos Ataques: herdando a experiência antiga de ATTACK
                            new_values[0] = old_values[0]  # ATTACK_STRONG
                            new_values[1] = old_values[0]  # ATTACK_PREDICTIVE
                            new_values[2] = old_values[0]  # ATTACK_PIVOT
                            new_values[3] = old_values[0]  # ATTACK_TECH
                            
                            new_values[4] = old_values[1]  # ATTACK_MEC
                            new_values[5] = old_values[2]  # BUFF
                            new_values[6] = old_values[3]  # STATUS
                            new_values[7] = old_values[4]  # HEAL
                            new_values[8] = old_values[5]  # CLEAN_HAZARD
                            new_values[9] = old_values[6]  # PROTECT
                            new_values[10] = old_values[7] # DEBUFF
                            new_values[11] = old_values[8] # STAT_CLEAN
                            new_values[12] = old_values[9] # HEAL_STATUS
                            new_values[13] = old_values[10] # PHAZE
                            new_values[14] = old_values[11] # FIELD_CONTROL
                            new_values[15] = old_values[12] # HAZARD
                            
                            # Ramificação das Trocas: herdando a experiência antiga de SWITCH
                            new_values[16] = old_values[13] # SWITCH_DEFENSIVE
                            new_values[17] = old_values[13] # SWITCH_OFFENSIVE
                            
                            new_q_table[new_state] = new_values
                            
                        # Fallback caso encontre uma tabela ultra-antiga de 10 ações (V_4)
                        elif len(old_values) == 10:
                            new_values = [0.0] * 18
                            new_values[0] = old_values[0]; new_values[1] = old_values[0]; new_values[2] = old_values[0]; new_values[3] = old_values[0]
                            new_values[5] = old_values[1]
                            new_values[6] = old_values[2]
                            new_values[7] = max(old_values[3], old_values[9])
                            new_values[8] = old_values[4]
                            new_values[9] = old_values[5]
                            new_values[10] = old_values[6]
                            new_values[12] = old_values[7]
                            new_values[16] = old_values[8]; new_values[17] = old_values[8]
                            new_q_table[new_state] = new_values
                            
                        else:
                            new_q_table[new_state] = old_values

                    self.q_table = new_q_table
                    print(f"[CÉREBRO V_6] Arquivo migrado e carregado com sucesso. Estados preservados: {len(self.q_table)}")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar e migrar modelo: {e}")