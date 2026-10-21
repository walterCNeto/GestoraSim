import sys

from gestorasim.mtm import resumo, rodar_serie

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(resumo(rodar_serie(seed=42)))
