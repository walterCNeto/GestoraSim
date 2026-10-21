import sys

from gestorasim.boletagem import resumo, rodar_dia_exemplo

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(resumo(rodar_dia_exemplo(seed=42)))
