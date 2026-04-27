import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "circuit_state.json")

DEFAULT_CONFIG = {
    "phase": "maxdamage",
    "opponent": "maxdamage",
    "n_battles": 10000,
    "session_number": 1,
    "brain_filename": "blue_brain.pkl",  # <--- NOME LIMPO
    "frozen_brain": None,
    "target_wr": 0.75,
    "patience": 3,
    "epsilon_override": None,
    "block_size": 500  # <--- SINCRONIZADO COM O AGENTE
}

def load_circuit_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_circuit_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)