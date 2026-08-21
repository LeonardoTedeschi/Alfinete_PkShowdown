from poke_env.data import GenData

def check_matchup():
    gen_data = GenData.from_gen(9)
    type_chart = gen_data.type_chart
    
    print("--- DIAGNÓSTICO DE TIPOS (GEN 9) ---")
    
    # Teste 1: Fogo vs Água (Deve ser 0.5)
    fire_vs_water = type_chart.get('FIRE', {}).get('WATER', 1.0)
    print(f"Fogo atacando Água: {fire_vs_water}x (Esperado: 0.5)")
    
    # Teste 2: Água vs Fogo (Deve ser 2.0)
    water_vs_fire = type_chart.get('WATER', {}).get('FIRE', 1.0)
    print(f"Água atacando Fogo: {water_vs_fire}x (Esperado: 2.0)")
    
    # Teste 3: Inseto vs Fogo (Deve ser 0.5)
    bug_vs_fire = type_chart.get('BUG', {}).get('FIRE', 1.0)
    print(f"Inseto atacando Fogo: {bug_vs_fire}x (Esperado: 0.5)")

    # Teste 4: Chuva
    print("\n--- LÓGICA DE CHUVA ---")
    # Simulação do nosso código brain.py
    weather_id = 2 # RAIN
    
    # Multiplicador de agua na chuva
    water_boost = 1.0
    if weather_id == 2: water_boost = 1.5
    print(f"Boost de Água na Chuva: {water_boost}x")
    
    # Dano Final Teórico
    total = water_vs_fire * water_boost
    print(f"Total (Água -> Fogo + Chuva): {total}x de Dano Base")

if __name__ == "__main__":
    check_matchup()