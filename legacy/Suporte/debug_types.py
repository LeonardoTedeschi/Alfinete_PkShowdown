import sys
import os

# Adiciona o diretório atual ao path para importar o brain
sys.path.append(os.getcwd())

try:
    from brain import RLBrain
    print("✅ Cérebro importado com sucesso!")
except ImportError:
    print("❌ Erro: Não foi possível importar 'brain.py'. Certifique-se que este arquivo está na mesma pasta.")
    sys.exit()

def testar_interacao(brain, atacante, defensor, esperado, nome_teste):
    """
    Testa uma interação específica e imprime o resultado detalhado.
    """
    print(f"\n--- TESTE: {nome_teste} ---")
    print(f"🗡️  Atacante: {atacante}")
    print(f"🛡️  Defensor: {defensor}")
    
    # 1. Espiar os dados CRUS (O Retorno do Showdown)
    # Tenta pegar os dados do defensor para ver o código
    defensor_data = brain.type_chart.get(defensor.title()) # Showdown usa Title Case (ex: "Fairy")
    codigo_bruto = "Não encontrado"
    
    if defensor_data and 'damageTaken' in defensor_data:
        # Tenta achar o atacante
        # Nota: As chaves dentro de damageTaken também podem variar, vamos procurar
        for key, val in defensor_data['damageTaken'].items():
            if key.upper() == atacante.upper():
                codigo_bruto = val
                break
    
    print(f"📊 Código Showdown (Cru): {codigo_bruto}")
    print(f"   (Legenda: 0=Neutro, 1=SuperEfetivo, 2=Resistido, 3=Imune)")
    
    # 2. Testar a Interpretação do Cérebro
    resultado = brain.get_type_multiplier(atacante, [defensor])
    
    print(f"🧠 Interpretação do Bot: {resultado}x")
    print(f"🎯 Esperado:             {esperado}x")
    
    if resultado == esperado:
        print("✅ APROVADO")
        return True
    else:
        print("❌ FALHOU")
        return False

# --- EXECUÇÃO ---
print("========================================")
print("DIAGNÓSTICO DE TIPOS - POKEMON BOT")
print("========================================")

# Inicializa o cérebro
print("Carregando tabelas do Poke-Env...")
brain = RLBrain()
print("Tabelas carregadas.\n")

erros = 0

# 1. O CASO DO CRIME: Dark vs Fairy (Deve ser Resistido/0.5x)
# Se o código antigo estivesse rodando, o "2" do showdown viraria "2.0x"
if not testar_interacao(brain, "Dark", "Fairy", 0.5, "Dark vs Fada (Resistido)"): erros += 1

# 2. Teste Super Efetivo: Water vs Fire (Deve ser 2.0x)
if not testar_interacao(brain, "Water", "Fire", 2.0, "Água vs Fogo (Super Efetivo)"): erros += 1

# 3. Teste Imunidade: Ground vs Flying (Deve ser 0.0x)
if not testar_interacao(brain, "Ground", "Flying", 0.0, "Terra vs Voador (Imune)"): erros += 1

# 4. Teste Neutro: Normal vs Water (Deve ser 1.0x)
if not testar_interacao(brain, "Normal", "Water", 1.0, "Normal vs Água (Neutro)"): erros += 1

# 5. Teste Duplo Tipo: Fire vs Grass/Bug (Deve ser 4.0x)
# Grass toma 2x, Bug toma 2x = 4x
print(f"\n--- TESTE: Fogo vs Grama/Inseto (4x) ---")
res_duplo = brain.get_type_multiplier("Fire", ["Grass", "Bug"])
print(f"Resultado: {res_duplo}x")
if res_duplo == 4.0:
    print("✅ APROVADO")
else:
    print(f"❌ FALHOU (Esperado 4.0)")
    erros += 1

print("\n========================================")
if erros == 0:
    print("🎉 SUCESSO TOTAL: O Tradutor está funcionando perfeitamente!")
    print("O bot agora sabe a diferença entre Resistência e Fraqueza.")
else:
    print(f"⚠️  ALERTA: {erros} testes falharam. Verifique o código.")
print("========================================")