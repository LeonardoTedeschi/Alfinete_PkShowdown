import subprocess
import time
import shutil
import os
import sys
import json
import csv
from datetime import datetime

# --- CAMINHOS ---
INSTINTO_DIR = r"C:\Projetos Robotica Computacional\Projeto Showdown IA Pokemon\Bot-QV-Pokemon\Instinto"
SCRIPT_DO_BOT = os.path.join(INSTINTO_DIR, "blue_agent.py")
CONFIG_PATH = os.path.join(INSTINTO_DIR, "circuit_state.json")
SUMMARY_PATH = os.path.join(INSTINTO_DIR, "last_session_summary.json")
CSV_LOG_DIR = os.path.join(INSTINTO_DIR, "logs")
os.makedirs(CSV_LOG_DIR, exist_ok=True)

# Nome padronizado para o cérebro congelado do self-play
FROZEN_PATH = os.path.join(INSTINTO_DIR, "frozen_blue_brain.pkl")

# --- CONFIGURAÇÃO DO CIRCUITO ---
CIRCUIT = {
    "maxdamage": {
        "opponent": "maxdamage",
        "target_wr": 0.75,
        "patience": 2,
        "max_sessions": 6,
        "n_battles": 10000,
        "epsilon_start": 0.35,
        "brain_suffix": "maxdamage"
    },
    "instinct": {
        "opponent": "instinct",
        "target_wr": 0.60,
        "patience": 3,
        "max_sessions": 10,
        "n_battles": 10000,
        "epsilon_start": 0.30,
        "brain_suffix": "instinct"
    },
    "selfplay_v1": {
        "opponent": "selfplay_frozen",
        "target_wr": 0.52,
        "patience": 4,
        "max_sessions": 15,
        "n_battles": 10000,
        "epsilon_start": 0.20,
        "brain_suffix": "selfplay",
        "update_frozen_every": 3
    },
    "selfplay_v2": {
        "opponent": "selfplay_frozen",
        "target_wr": 0.52,
        "patience": 5,
        "max_sessions": 20,
        "n_battles": 10000,
        "epsilon_start": 0.10,
        "brain_suffix": "selfplay_final",
        "update_frozen_every": 3
    }
}

PHASE_ORDER = ["maxdamage", "instinct", "selfplay_v1", "selfplay_v2"]
TEMPO_LIMITE_MINUTOS = 240

def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [CIRCUITO] {msg}")

def read_last_summary():
    if not os.path.exists(SUMMARY_PATH):
        return None
    try:
        with open(SUMMARY_PATH, "r") as f:
            return json.load(f)
    except:
        return None

def backup_brain(source_name, suffix):
    src = os.path.join(INSTINTO_DIR, source_name)
    if not os.path.exists(src):
        return None
    backup_name = f"blue_brain_{suffix}.pkl"
    dst = os.path.join(INSTINTO_DIR, backup_name)
    shutil.copy2(src, dst)
    return dst

def create_or_update_frozen(brain_filename="blue_brain.pkl"):
    """Cria ou sobrescreve o frozen brain com o cérebro atual."""
    src = os.path.join(INSTINTO_DIR, brain_filename)
    if not os.path.exists(src):
        log_msg(f"AVISO: Cérebro fonte não encontrado: {src}")
        return False
    shutil.copy2(src, FROZEN_PATH)
    log_msg(f"Frozen brain atualizado: {FROZEN_PATH}")
    return True

def write_circuit_config(phase_key, session_num, frozen_brain=None):
    cfg = CIRCUIT[phase_key]

    config = {
        "phase": phase_key.replace("_v1", "").replace("_v2", ""),
        "opponent": cfg["opponent"],
        "n_battles": cfg["n_battles"],
        "session_number": session_num,
        "brain_filename": "blue_brain.pkl", # Nome limpo e padronizado
        "frozen_brain": frozen_brain,
        "target_wr": cfg["target_wr"],
        "patience": cfg["patience"],
        "epsilon_override": cfg.get("epsilon_start"),
        "block_size": 500
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    log_msg(f"Config escrita: fase={phase_key} | oponente={cfg['opponent']} | sessão={session_num}")

def run_session():
    inicio = time.time()
    try:
        subprocess.run(
            [sys.executable, SCRIPT_DO_BOT],
            cwd=INSTINTO_DIR,
            check=True,
            timeout=TEMPO_LIMITE_MINUTOS * 60
        )
        tempo = (time.time() - inicio) / 60
        return True, tempo
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        log_msg(f"ERRO/TIMEOUT na sessão: {e}")
        return False, 0.0

def main():
    log_msg("=== INICIANDO CIRCUITO DE TREINAMENTO AUTOMÁTICO ===")

    circuit_state_path = os.path.join(INSTINTO_DIR, "circuit_progress.json")
    if os.path.exists(circuit_state_path):
        with open(circuit_state_path, "r") as f:
            progress = json.load(f)
    else:
        progress = {
            "current_phase_idx": 0,
            "phase_sessions": 0,
            "stable_sessions": 0,
            "total_sessions": 0,
            "frozen_created": False
        }

    while progress["current_phase_idx"] < len(PHASE_ORDER):
        current_phase_key = PHASE_ORDER[progress["current_phase_idx"]]
        cfg = CIRCUIT[current_phase_key]

        if progress["phase_sessions"] >= cfg["max_sessions"]:
            log_msg(f"Hard cap atingido ({cfg['max_sessions']} sessões). Avançando.")
            progress["current_phase_idx"] += 1
            progress["phase_sessions"] = 0
            progress["stable_sessions"] = 0
            progress["frozen_created"] = False
            continue

        progress["total_sessions"] += 1
        progress["phase_sessions"] += 1
        session_num = progress["total_sessions"]

        # Removido: backup_brain("blue_brain.pkl", f"pre_sessao_{session_num}")
        # Lógica de backups de segurança movida apenas para o final de cada fase concluída.

        frozen_to_use = None
        if "selfplay" in current_phase_key:
            if not progress.get("frozen_created", False):
                if create_or_update_frozen("blue_brain.pkl"):
                    progress["frozen_created"] = True
            elif os.path.exists(FROZEN_PATH):
                update_every = cfg.get("update_frozen_every", 3)
                if progress["phase_sessions"] % update_every == 0:
                    create_or_update_frozen("blue_brain.pkl")
            
            if os.path.exists(FROZEN_PATH):
                frozen_to_use = FROZEN_PATH

        write_circuit_config(current_phase_key, session_num, frozen_brain=frozen_to_use)

        log_msg(f">>> Sessão {session_num} | Fase: {current_phase_key} ({progress['phase_sessions']}/{cfg['max_sessions']})")
        success, tempo = run_session()

        if not success:
            log_msg("Sessão falhou. Tentando novamente na próxima iteração...")
            time.sleep(5)
            continue

        summary = read_last_summary()
        wr = summary.get("win_rate", 0.0) if summary else 0.0
        wr_decimal = wr / 100.0 if wr > 1.0 else wr

        if wr_decimal >= cfg["target_wr"]:
            progress["stable_sessions"] += 1
            log_msg(f"Target WR atingido! Estabilidade: {progress['stable_sessions']}/{cfg['patience']}")

            if progress["stable_sessions"] >= cfg["patience"]:
                log_msg(f"=== TRANSIÇÃO: {current_phase_key} CONCLUÍDA ===")
                backup_brain("blue_brain.pkl", cfg["brain_suffix"]) # Backup de fase concluída

                progress["current_phase_idx"] += 1
                progress["phase_sessions"] = 0
                progress["stable_sessions"] = 0
                progress["frozen_created"] = False
        else:
            progress["stable_sessions"] = 0
            log_msg(f"WR abaixo do target ({cfg['target_wr']:.0%}). Resetando estabilidade.")

        with open(circuit_state_path, "w") as f:
            json.dump(progress, f, indent=2)

        time.sleep(3)

    log_msg("=== CIRCUITO COMPLETO: TREINAMENTO FINALIZADO ===")

if __name__ == "__main__":
    main()