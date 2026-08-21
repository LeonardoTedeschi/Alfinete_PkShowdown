"""
InstinctBot — Agente 2 do projeto ALFINETE.

Joga usando APENAS o conhecimento de domínio (o instinto), sem qualquer aprendizado:
sem Q-table, sem epsilon, sem treino. A mesma situação produz sempre a mesma decisão
(determinístico, a menos do RNG do próprio jogo).

Papel na pesquisa: é a RÉGUA de avaliação. Todos os outros agentes (Green/Q-puro,
Blue/Híbrido, Red/DQN) treinam e são medidos contra ele. Por ser determinístico,
qualquer diferença de Win Rate entre agentes vem DELES, não de flutuação do oponente
— é o que torna a comparação justa e reproduzível.

Este ficheiro é uma CASCA FINA: toda a lógica tática vive nos componentes do pacote
`instinct`. O bot apenas liga o fluxo (troca forçada -> decidir -> executar).
"""

from poke_env.player import Player

# Ao instalar no projeto: `from instinct import build_instinct`
from instinct import build_instinct


class InstinctBot(Player):
    """Agente que decide exclusivamente pelo instinto (conhecimento de domínio)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Monta os 6 componentes já ligados (physics, parser, masker, policy, executor).
        self.instinct = build_instinct()

    def teampreview(self, battle):
        """Escolhe a ordem de time inicial pela heurística de lead do instinto."""
        return self.instinct.executor.get_best_lead(battle)

    def choose_move(self, battle):
        try:
            # 1. Troca forçada (Pokémon ativo desmaiou ou foi forçado a sair).
            if battle.force_switch or (battle.active_pokemon and battle.active_pokemon.fainted):
                switch = self.instinct.executor.get_post_faint_switch(battle)
                return self.create_order(switch) if switch else self.choose_random_move(battle)

            # 2. Sem ativo/oponente definido (turno de transição) -> aleatório seguro.
            if not battle.active_pokemon or not battle.opponent_active_pokemon:
                return self.choose_random_move(battle)

            # 3. DECISÃO: a policy devolve o ranking de intenções (5 valores).
            #    Usamos o ranking_list (índice 2), não a tupla inteira.
            _primary, _conf, ranking_list, _mask, _has_lethal = \
                self.instinct.policy.get_instinct_profile(battle)

            # 4. EXECUÇÃO: percorre o ranking até o executor devolver um objeto válido.
            for intent in ranking_list:
                obj = self.instinct.executor.get_best_execution_object(intent, battle)
                if obj:
                    return self.create_order(obj)

            return self.choose_random_move(battle)

        except Exception:
            # Falha defensiva: nunca trava a batalha, joga algo legal.
            return self.choose_random_move(battle)
