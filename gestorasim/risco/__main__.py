import sys
import time

from gestorasim.risco import executar_risco, resumo

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    print(resumo(executar_risco(seed=42)))
    print(f"tempo de execução (s): {time.time() - t0:.3f}")
