import asyncio
import time
import sys
import os

# Importações dos Bots e Times
try:
    from instinct_bot import InstinctBot
    from rivals import MaxDamagePlayer
    from teams import RandomTeamFromPool, TEAMS_LIST
except ImportError as e:
    print(f"Erro crítico de importação: {e}")
    print("Verifique se instinct_bot.py, rivals.py e teams.py estão na mesma pasta.")
    sys.exit(1)

async def run_battles():
    # --- CONFIGURAÇÕES ---
    N_BATALHAS = 2000
    # O formato deve ser Gen 9 National Dex para aceitar os times do arquivo
    FORMATO = "gen9nationaldex" 
    
    # Prepara o construtor de times
    team_builder = RandomTeamFromPool(TEAMS_LIST)

    print(f"\n{'='*50}")
    print(f"   SIMULAÇÃO DE COMBATE: INSTINCT vs MAXDAMAGE")
    print(f"{'='*50}")
    print(f"Bot Principal: InstinctBot (Lógica Pura)")
    print(f"Oponente:      MaxDamagePlayer (Ataque Bruto)")
    print(f"Formato:       {FORMATO}")
    print(f"Times:         Pool de {len(team_builder.packed_teams)} times personalizados")
    print(f"Quantidade:    {N_BATALHAS} batalhas")
    print("-" * 50)
    print("Iniciando batalhas... (Aguarde o relatório final)")

    # 1. Inicializa os Bots com os Times Personalizados
    # Nota: Não usamos account_configuration/player_configuration para modo local
    try:
        hero = InstinctBot(
            battle_format=FORMATO,
            team=team_builder
        )

        villain = MaxDamagePlayer(
            battle_format=FORMATO,
            team=team_builder
        )
    except Exception as e:
        print(f"Erro ao criar os bots: {e}")
        return

    # 2. Execução
    start_time = time.time()

    try:
        # Executa as batalhas
        await hero.battle_against(villain, n_battles=N_BATALHAS)
    except Exception as e:
        print(f"\n[ERRO DURANTE AS BATALHAS]: {e}")
        print("Dica: Se o erro for 'challenge already exists', reinicie o servidor do Showdown.")
        return

    end_time = time.time()
    duration = end_time - start_time

    # 3. Coleta de Resultados
    total_matches = hero.n_finished_battles
    wins = hero.n_won_battles
    losses = total_matches - wins
    win_rate = (wins / total_matches * 100) if total_matches > 0 else 0.0

    # 4. Relatório Final
    print(f"\n{'='*40}")
    print(f"           RELATÓRIO FINAL")
    print(f"{'='*40}")
    print(f"Tempo Total:     {duration:.2f}s")
    if total_matches > 0:
        print(f"Tempo p/ Luta:   {duration/total_matches:.3f}s")
    print(f"{'-'*40}")
    print(f"Total Batalhas:  {total_matches}")
    print(f"Vitórias:        {wins}")
    print(f"Derrotas:        {losses}")
    print(f"{'-'*40}")
    print(f"TAXA DE VITÓRIA: {win_rate:.2f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    # Configuração específica para Windows para evitar erros de loop
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except AttributeError:
            pass # Ignora se a politica não existir nesta versão
        
    try:
        asyncio.run(run_battles())
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.")