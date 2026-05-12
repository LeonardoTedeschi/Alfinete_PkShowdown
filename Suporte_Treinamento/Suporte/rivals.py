from poke_env.player import Player

# Mapeamento simples para evitar erros de Z-Move
Z_CRYSTAL_MAP = {
    'wateriumz': 'water', 'normaliumz': 'normal', 'flyiniumz': 'flying',
    'rockiumz': 'rock', 'electriumz': 'electric', 'darkiniumz': 'dark',
    'iciumz': 'ice', 'grassiumz': 'grass', 'firiumz': 'fire',
    'poisoniumz': 'poison', 'ghostiumz': 'ghost', 'psychiumz': 'psychic',
    'steeliumz': 'steel', 'groundiumz': 'ground', 'dragoniumz': 'dragon',
    'bugiumz': 'bug', 'fightiniumz': 'fighting', 'fairiumz': 'fairy'
}

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # --- TRAVA DE SEGURANÇA PARA TROCAS FORÇADAS ---
        switch_forced = False
        if isinstance(battle.force_switch, list): 
            switch_forced = any(battle.force_switch)
        else: 
            switch_forced = bool(battle.force_switch)

        # Se a troca for obrigatória ou o Pokémon ativo desmaiou
        if (switch_forced or (battle.active_pokemon and battle.active_pokemon.fainted)) and battle.available_switches:
            # Escolhe o primeiro Pokémon disponível no banco (o MaxDamage não precisa pensar muito aqui)
            return self.create_order(battle.available_switches[0])
            
        # Proteção extra: se não há ataques disponíveis por algum motivo de bloqueio (Taunt/Encore/Disable)
        if not battle.available_moves and battle.available_switches:
            return self.create_order(battle.available_switches[0])
            
        if battle.available_moves:
            # 1. Escolhe o golpe mais forte
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            
            # 2. Configurações Iniciais
            do_mega = battle.can_mega_evolve
            do_tera = False
            do_z = False

            # 3. Verificação de Tera (Compatibilidade de versões)
            if hasattr(battle, "can_tera") and battle.can_tera:
                do_tera = True
            elif hasattr(battle, "can_terastallize") and battle.can_terastallize:
                do_tera = True
                
            # 4. Verificação SEGURA de Z-Move (A correção do erro)
            if battle.can_z_move and battle.active_pokemon.item:
                item_id = battle.active_pokemon.item
                # Só ativa Z-Move se o item for um cristal E o golpe for do tipo certo
                if item_id in Z_CRYSTAL_MAP:
                    required_type = Z_CRYSTAL_MAP[item_id]
                    if best_move.type and best_move.type.name.lower() == required_type:
                        do_z = True

            return self.create_order(
                best_move,
                mega=do_mega,
                z_move=do_z,
                terastallize=do_tera
            )
        else:
            return self.choose_random_move(battle)