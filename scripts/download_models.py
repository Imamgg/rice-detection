"""Optionally prefetch all weights before starting Streamlit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_store import model_manifest, ensure_weights

if __name__ == '__main__':
    for name in model_manifest():
        print(f'Menyiapkan {name}...', flush=True)
        print(ensure_weights(name), flush=True)
