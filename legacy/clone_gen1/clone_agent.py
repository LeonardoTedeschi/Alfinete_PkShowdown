from poke_env.player import Player
from .clone_core import InstinctCore
from .clone_brain import CloneBrain

class BlueClone(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = InstinctCore()
        self.brain = CloneBrain()

    def teampreview(self, battle):
        try:
            return self.core.get_best_lead(battle)
        except Exception:
            return "/team 123456"

    def choose_move(self, battle):
        # Limpeza silenciosa de memória
        battles_snapshot = list(self.battles.items())
        for b_id, b in battles_snapshot:
            if b.finished and b_id in self.battles:
                del self.battles[b_id]

        try:
            switch_forced = False
            if isinstance(battle.force_switch, list): switch_forced = any(battle.force_switch)
            else: switch_forced = bool(battle.force_switch)

            if switch_forced or (battle.active_pokemon and battle.active_pokemon.fainted):
                best_switch = self.core.get_best_switch(battle)
                if best_switch: return self.create_order(best_switch)
                return self.choose_random_move(battle)

            # Execução cega baseada na Gen1
            current_state = self.core.get_state(battle)
            instinct_intent = self.core.get_intent(battle)
            final_decision = self.brain.decide_action(current_state, instinct_intent)

            execution_list = [final_decision, instinct_intent, "ATTACK", "SWITCH"]
            best_object = self.core.get_best_execution_object(execution_list, battle)

            if best_object:
                return self.create_order(best_object)
            else:
                return self.choose_random_move(battle)

        except Exception:
            return self.choose_random_move(battle)