import sys
import time

from gestorasim.cota.relatorio import executar_cota, resumo

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    print(resumo(executar_cota(seed=42)))
    print(f"tempo de execução (s): {time.time() - t0:.3f}")
