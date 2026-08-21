import numpy as np
import random
import pickle
import os
import math
import json
from poke_env.data import GenData

class RLBrain:
    def __init__(self, alpha=0.1, gamma=0.8, epsilon=0.3, min_epsilon=0.01, decay=0.995):
        self.alpha = alpha     
        self.gamma = gamma     
        self.epsilon = epsilon         
        self.min_epsilon = min_epsilon 
        self.epsilon_decay = decay     
        self.q_table = {}
        
        self.type_chart = {}
        try:
            with open('tabela_tipos_dump.json', 'r') as f:
                self.type_chart = json.load(f)
        except:
            print("Aviso: tabela_tipos_dump.json não encontrado.")

        self.STATUS_MAP = {
            'PSN': 1, 'TOX': 2, 'BRN': 3,
            'PAR': 4, 'SLP': 5, 'FRZ': 6,
            'NONE': 0
        }

    # --- IO ---
    def save_model(self, filename="red_brain.pkl"):
        data = {"q_table": self.q_table.copy(), "epsilon": self.epsilon}
        try:
            with open(filename, "wb") as f: pickle.dump(data, f)
        except Exception: pass

    def load_model(self, filename="red_brain.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    self.q_table = data.get("q_table", {})
                    self.epsilon = data.get("epsilon", self.epsilon)
                return True
            except: return False
        return False

    # --- HELPERS ---
    def _name(self, obj):
        try:
            if hasattr(obj, 'name'): return str(obj.name).upper()
            if isinstance(obj, dict): return str(obj.get('name', 'NORMAL')).upper()
            if isinstance(obj, str): return obj.upper()
        except: pass
        return "NORMAL"

    def _get_stat_mult(self, boost):
        return (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)

    def get_type_multiplier(self, atk_type, defender_types):
        if not atk_type: return 1.0
        final_multiplier = 1.0
        
        def_types_list = []
        if isinstance(defender_types, list):
            def_types_list = [self._name(t) for t in defender_types if t]
        elif defender_types:
            def_types_list = [self._name(defender_types)]
            
        atk_name = self._name(atk_type)
        
        for dt in def_types_list:
            if dt not in self.type_chart:
                continue 
            
            defender_data = self.type_chart[dt]
            
            val = defender_data.get(atk_name, 1.0)
            
            final_multiplier *= float(val)
            
            if final_multiplier == 0.0: return 0.0
            
        return final_multiplier

    def _get_weather_details(self, weather, battle_fields):
        # Retorna ID técnico para cálculo de dano
        if 'TRICKROOM' in [str(k).upper() for k in battle_fields.keys()]: return 4
        if not weather: return 0
        w = self._name(weather)
        if 'SUN' in w or 'DROUGHT' in w: return 1
        if 'RAIN' in w or 'DRIZZLE' in w: return 2
        if 'SAND' in w or 'SNOW' in w or 'HAIL' in w: return 3
        return 0

    def _get_weather_multiplier(self, type_name, weather_id):
        # Usa o ID técnico (1=Sol, 2=Chuva) para calcular o buff correto
        t = self._name(type_name)
        if weather_id == 1: # Sol
            if t == 'FIRE': return 1.5
            if t == 'WATER': return 0.5
        elif weather_id == 2: # Chuva
            if t == 'WATER': return 1.5
            if t == 'FIRE': return 0.5
        return 1.0

    def _get_threat_level(self, active, opponent, weather_id):
        # (-1, 0, 1, 2)
        opp_atk = opponent.base_stats.get('atk', 100) * self._get_stat_mult(opponent.boosts.get('atk', 0))
        opp_spa = opponent.base_stats.get('spa', 100) * self._get_stat_mult(opponent.boosts.get('spa', 0))
        my_def = active.base_stats.get('def', 100) * self._get_stat_mult(active.boosts.get('def', 0))
        my_spd = active.base_stats.get('spd', 100) * self._get_stat_mult(active.boosts.get('spd', 0))

        opp_types = [self._name(t) for t in opponent.types if t]
        my_types = [self._name(t) for t in active.types if t] 
        max_score = 0.0
        
        for t in opp_types:
            power = max(opp_atk, opp_spa)
            defense = min(my_def, my_spd)
            type_mult = self.get_type_multiplier(t, my_types)
            # Aqui ele sabe se é Sol ou Chuva para aplicar corretamente
            weather_mult = self._get_weather_multiplier(t, weather_id)
            stab = 1.5
            
            score = (power / max(1, defense)) * type_mult * weather_mult * stab
            if score > max_score: max_score = score
            
        if max_score >= 2.5: return 2
        if max_score >= 1.2: return 1
        if max_score < 0.8: return -1
        return 0

    def _get_offense_potential(self, active, opponent, weather_id):
        # (-1, 0, 1, 2)
        my_atk = active.base_stats.get('atk', 100) * self._get_stat_mult(active.boosts.get('atk', 0))
        my_spa = active.base_stats.get('spa', 100) * self._get_stat_mult(active.boosts.get('spa', 0))
        opp_def = opponent.base_stats.get('def', 100) * self._get_stat_mult(opponent.boosts.get('def', 0))
        opp_spd = opponent.base_stats.get('spd', 100) * self._get_stat_mult(opponent.boosts.get('spd', 0))

        my_types = [self._name(t) for t in active.types if t]
        opp_types = [self._name(t) for t in opponent.types if t]
        max_score = 0.0
        
        for t in my_types:
            power = max(my_atk, my_spa)
            defense = min(opp_def, opp_spd)
            type_mult = self.get_type_multiplier(t, opp_types)
            weather_mult = self._get_weather_multiplier(t, weather_id)
            stab = 1.5
            
            score = (power / max(1, defense)) * type_mult * weather_mult * stab
            if score > max_score: max_score = score

        if max_score >= 2.5: return 2
        if max_score >= 1.2: return 1
        if max_score < 0.8: return -1
        return 0

    def _get_hp_bucket(self, hp_fraction):
        if hp_fraction >= 0.80: return 3
        if hp_fraction >= 0.50: return 2
        if hp_fraction >= 0.15: return 1
        return 0

    def _get_boost_level(self, pokemon):
        total = pokemon.boosts.get('atk', 0) + pokemon.boosts.get('spa', 0)
        if total > 0: return 1
        if total < 0: return -1
        return 0

    def _get_team_score(self, battle):
        n_my = len([p for p in battle.team.values() if not p.fainted])
        n_opp = 6 - len([p for p in battle.opponent_team.values() if p.fainted])
        diff = n_my - n_opp
        if diff > 0: return 1
        if diff < 0: return -1
        return 0

    def get_state_key(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not active or not opponent: return (0,0,0,0,0,0,0,0,0,0,0)

        # Obtém o ID técnico (1=Sol, 2=Chuva, 3=Areia, 4=TR)
        weather_tech_id = self._get_weather_details(battle.weather, battle.fields)
        
        # 1. Ameaça e Ofensiva usam o ID técnico para calcular dano real
        threat = self._get_threat_level(active, opponent, weather_tech_id)
        offense = self._get_offense_potential(active, opponent, weather_tech_id)
        
        hp_my = self._get_hp_bucket(active.current_hp_fraction)
        hp_opp = self._get_hp_bucket(opponent.current_hp_fraction)
        
        st_my_str = str(active.status).split('.')[-1] if active.status else 'NONE'
        st_my = 0
        for k, v in self.STATUS_MAP.items():
            if k in st_my_str.upper(): st_my = v
            
        st_opp = 1 if opponent.status else 0
        
        my_spe = active.base_stats['spe'] * self._get_stat_mult(active.boosts.get('spe', 0))
        opp_spe = opponent.base_stats['spe'] * self._get_stat_mult(opponent.boosts.get('spe', 0))
        if active.status and 'PAR' in str(active.status): my_spe *= 0.5
        if opponent.status and 'PAR' in str(opponent.status): opp_spe *= 0.5
        
        if weather_tech_id == 4: # TrickRoom
            am_i_faster = 1 if my_spe < opp_spe else 0
        else:
            am_i_faster = 1 if my_spe > opp_spe else 0 
            
        bst_my = self._get_boost_level(active)
        bst_opp = self._get_boost_level(opponent)
        score = self._get_team_score(battle)
        
        # --- SIMPLIFICAÇÃO FINAL DO ESTADO (Para a Q-Table) ---
        # Aqui fundimos Sol(1) e Chuva(2) no Estado de Clima 1 (Amplificador)
        # 0: Nenhum
        # 1: Amplificador (Sol ou Chuva)
        # 2: Desgastante (Areia ou Neve) - era o 3 técnico
        # 3: Trick Room - era o 4 técnico
        
        final_weather_state = 0
        if weather_tech_id == 1 or weather_tech_id == 2:
            final_weather_state = 1
        elif weather_tech_id == 3:
            final_weather_state = 2
        elif weather_tech_id == 4:
            final_weather_state = 3

        return (threat, offense, hp_my, hp_opp, st_my, st_opp, am_i_faster, bst_my, bst_opp, score, final_weather_state)

    # --- RECOMPENSAS ---
    def calculate_reward(self, battle, history):
        R_WIN = 1500.0   
        R_LOSE = -1000.0 
        R_KILL = 175.0   
        R_DEATH = -50.0 
        R_SWITCH_COST = -30.0 
        
        R_WASTE_TURN = -30.0  
        R_HELP_ENEMY = -50.0  
        R_HEAL_FULL = -40.0   
        R_SUB_ERROR = -40.0   

        R_SUPER_EFF = 50.0    
        R_RESISTED  = -30.0   
        R_IMMUNE    = -60.0   
        
        R_GOOD_PLAY  = 30.0   
        R_HAZARD_BONUS = 30.0
        R_STATUS_BONUS = 30.0
        R_HEAL_BONUS = 20.0
        R_BUFF_BONUS = 10.0

        if battle.finished:
            return R_WIN if battle.won else R_LOSE

        reward = 0.0
        move_used = history.get('move_obj')
        was_switch = history.get('was_switch')
        
        if was_switch:
            reward += R_SWITCH_COST

        h_my_hp = history.get('my_hp', 1.0)
        h_opp_hp = history.get('opp_hp', 1.0)
        h_opp_hazards = set(history.get('opp_hazards', []))
        h_opp_status = history.get('opp_status', "None")
        h_opp_boosts = history.get('opp_boosts', {}) 

        opp_active = battle.opponent_active_pokemon
        active = battle.active_pokemon
        curr_opp_hp = opp_active.current_hp_fraction if opp_active else 0
        curr_my_hp = active.current_hp_fraction if active else 0

        if move_used and not was_switch:
            move_id = getattr(move_used, 'id', '')
            move_heal = getattr(move_used, 'heal', 0)
            
            if move_id in ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']:
                if move_id in h_opp_hazards: reward += R_WASTE_TURN 

            causes_status = False
            if move_id in ['toxic', 'willowisp', 'thunderwave', 'spore', 'hypnosis']: causes_status = True
            if causes_status and h_opp_status != 'None': reward += R_WASTE_TURN 

            if move_id == 'substitute':
                active_effects = [str(e).lower() for e in active.effects.keys()] if active else []
                if 'substitute' in active_effects: reward += R_SUB_ERROR

            curr_opp_hazards = set(battle.opponent_side_conditions.keys())
            if len(curr_opp_hazards) < len(h_opp_hazards) and move_id in ['defog', 'courtchange']: 
                reward += R_HELP_ENEMY 

            sum_old_boosts = sum(h_opp_boosts.values())
            if move_id in ['haze', 'clearsmog']:
                if sum_old_boosts < 0: reward += R_HELP_ENEMY 
                elif sum_old_boosts > 0: reward += R_GOOD_PLAY 

            if hasattr(move_used, 'type') and opp_active:
                opp_types = [t.name for t in opp_active.types if t]
                eff = self.get_type_multiplier(move_used.type, opp_types)
                if move_used.category.name in ['PHYSICAL', 'SPECIAL']:
                    if eff > 1.1: reward += R_SUPER_EFF 
                    elif eff < 0.1: reward += R_IMMUNE    
                    elif eff < 0.9: reward += R_RESISTED  

            if len(curr_opp_hazards) > len(h_opp_hazards): reward += R_HAZARD_BONUS
            
            curr_opp_status = str(opp_active.status) if opp_active else "None"
            if h_opp_status == "None" and curr_opp_status != "None": reward += R_STATUS_BONUS

            if move_heal > 0:
                if h_my_hp > 0.8: reward += R_HEAL_FULL
                elif curr_my_hp > h_my_hp: reward += R_HEAL_BONUS

            curr_my_boosts = active.boosts if active else {}
            h_my_boosts = history.get('my_boosts', {})
            if sum(curr_my_boosts.values()) > sum(h_my_boosts.values()):
                if curr_my_hp > 0.6: reward += R_BUFF_BONUS

        if curr_opp_hp <= 0: reward += R_KILL 
        if curr_my_hp <= 0 and h_my_hp > 0: reward += R_DEATH

        return reward

    def update_knowledge(self, prev, act, reward, curr):
        old = self.q_table.get((prev, act), 0.0)
        qs = [self.q_table.get((curr, a), 0.0) for a in range(10)]
        fut = max(qs) if qs else 0.0
        new_q = old + self.alpha * (reward + self.gamma * fut - old)
        self.q_table[(prev, act)] = new_q

    def _is_move_useless(self, move, opp_pokemon):
        if not opp_pokemon: return False
        opp_types = [self._name(t) for t in opp_pokemon.types if t]
        move_type = self._name(move.type)
        move_id = move.id
        if move.category.name in ['PHYSICAL', 'SPECIAL']:
            mult = self.get_type_multiplier(move_type, opp_types)
            if mult == 0.0: return True
        if move.category.name == 'STATUS':
            if move_id in ['toxic', 'poisonpowder', 'poisonjob']:
                if 'STEEL' in opp_types or 'POISON' in opp_types: return True
            if move_id == 'thunderwave':
                if 'GROUND' in opp_types or 'ELECTRIC' in opp_types: return True
            if move_id == 'willowisp':
                if 'FIRE' in opp_types: return True
            if move_id in ['leechseed', 'spore', 'sleeppowder', 'stunspore']:
                if 'GRASS' in opp_types: return True
        return False

    def choose_action(self, state, moves, switches, battle=None): 
        threat_level = state[0]  
        offense_potential = state[1] 
        am_i_faster = state[6] 

        atk_indices = []
        switch_indices = []
        index_to_obj = {}
        opp_pokemon = battle.opponent_active_pokemon if battle else None

        for i, m in enumerate(moves):
            if i < 4:
                if opp_pokemon and self._is_move_useless(m, opp_pokemon): continue
                atk_indices.append(i)
                index_to_obj[i] = m
                
        for i, s in enumerate(switches):
            idx = 4 + i; 
            if idx < 10: switch_indices.append(idx); index_to_obj[idx] = s

        if not atk_indices and not switch_indices: 
            for i, m in enumerate(moves):
                 if i < 4: atk_indices.append(i); index_to_obj[i] = m

        if not atk_indices and not switch_indices: return None, -1, "none"

        # DECISÃO PURA (Sem Instinto Hardcoded)
        # Deixa a Q-Table decidir se deve atacar ou trocar com base no aprendizado
        final_indices = atk_indices + switch_indices

        if not final_indices: 
            # Fallback de segurança se nenhuma lista tiver opções (muito raro)
            final_indices = atk_indices if atk_indices else switch_indices

        if random.random() < self.epsilon:
            chosen_idx = random.choice(final_indices)
            decision = "rand"
        else:
            best_q = -float('inf')
            best_idx = final_indices[0]
            random.shuffle(final_indices)
            for idx in final_indices:
                q_val = self.q_table.get((state, idx), 0.0)
                if q_val > best_q:
                    best_q = q_val
                    best_idx = idx
            chosen_idx = best_idx
            decision = "smart"

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return index_to_obj[chosen_idx], chosen_idx, decision