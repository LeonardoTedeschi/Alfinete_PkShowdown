import subprocess
import time
import sys
import shutil
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# CONFIGURAÇÃO
SCRIPT_DO_BOT = os.path.join(current_dir, "train_vs_instinct.py")
QUANTIDADE_DE_SESSOES = 10    # Ex: 10 sessões de 10k = 100k batalhas
TEMPO_LIMITE_MINUTOS = 45     # Se demorar mais que isso, mata e reinicia (Anti-Travamento)

brain_path = os.path.join(root_dir, "red_brain.pkl")
backup_path = os.path.join(root_dir, "red_brain_backup.pkl")

def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [GERENTE] {msg}")

def run_night_cycle():
    log_msg("=== INICIANDO TREINAMENTO NOTURNO ===")
    
    for i in range(1, QUANTIDADE_DE_SESSOES + 1):
        log_msg(f"--- Iniciando Sessão {i}/{QUANTIDADE_DE_SESSOES} ---")
        try:
            shutil.copy(brain_path, backup_path)
        except FileNotFoundError:
            pass # Primeira vez, sem backup

        inicio = time.time()
        
        try:
            # Roda o bot com um cronômetro (timeout)
            # Se passar de 45 minutos (2700 segundos), ele levanta um erro
            subprocess.run(
                [sys.executable, SCRIPT_DO_BOT], 
                check=True, 
                timeout=TEMPO_LIMITE_MINUTOS * 60
            )
            
            tempo_gasto = (time.time() - inicio) / 60
            log_msg(f"Sessão {i} concluída com sucesso em {tempo_gasto:.1f} minutos.")

        except subprocess.TimeoutExpired:
            log_msg(f"ALERTA: A Sessão {i} TRAVOU (Timeout). O processo foi morto.")
            log_msg("Reiniciando o servidor local e partindo para a próxima...")
            # Aqui ele ignora o travamento e força o próximo loop
            
        except subprocess.CalledProcessError as e:
            log_msg(f"ERRO: O Bot quebrou na Sessão {i}. Código: {e.returncode}")
        
        except Exception as e:
            log_msg(f"Erro desconhecido: {e}")

        # Descanso para o Servidor Local respirar
        log_msg("Resfriando motor (30 segundos)...")
        time.sleep(30)

    log_msg("=== MADRUGADA ENCERRADA ===")

if __name__ == "__main__":
    run_night_cycle()