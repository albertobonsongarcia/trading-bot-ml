"""
Loop infinito que actualiza predicciones cada hora.
Mantenlo corriendo en una terminal aparte mientras la app este en uso.
Detener con Ctrl+C.
"""
import time
import subprocess
import sys
import os
from datetime import datetime

INTERVAL_SECONDS = 3600  # 1 hora

def run_update():
    """Corre update_predictions.py una vez."""
    print(f"\n{'='*60}")
    print(f"Actualizando predicciones — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_predictions.py")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"⚠ Update fallo con codigo {result.returncode}, reintentara en 1 hora")

def main():
    print("=== Auto-update iniciado ===")
    print(f"Intervalo: cada {INTERVAL_SECONDS // 60} minutos")
    print("Presiona Ctrl+C para detener.\n")

    while True:
        try:
            run_update()
            next_run = datetime.now().timestamp() + INTERVAL_SECONDS
            print(f"\nProximo update: {datetime.fromtimestamp(next_run).strftime('%H:%M:%S')}")
            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n\nAuto-update detenido por el usuario.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Reintentando en 60 segundos...")
            time.sleep(60)

if __name__ == "__main__":
    main()