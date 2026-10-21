"""CLI do orquestrador.

  python -m gestorasim.orquestrador --de 2023-01-02 --ate 2024-12-30
  python -m gestorasim.orquestrador --data 2023-06-15 --reprocessar
"""

from __future__ import annotations

import argparse
import sys
import time

from gestorasim.orquestrador import ConfigCiclo, Orquestrador, resumo_ciclo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gestorasim.orquestrador", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--de", help="primeiro pregão (AAAA-MM-DD); default: início do calendário")
    ap.add_argument("--ate", help="último pregão (AAAA-MM-DD); default: fim do calendário")
    ap.add_argument("--data", help="pregão a reprocessar (com --reprocessar)")
    ap.add_argument("--reprocessar", action="store_true", help="recalcula --data e as datas seguintes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--destino", help="pasta de saída (default: data/sim)")
    ap.add_argument("--sem-conciliacao", action="store_true")
    ap.add_argument("--sem-risco", action="store_true", help="pula risco e pós-trade (mais rápido)")
    ap.add_argument("--passo-ordens", type=int, default=5)
    a = ap.parse_args(argv)
    if a.reprocessar and not a.data:
        ap.error("--reprocessar exige --data")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = ConfigCiclo(seed=a.seed, destino=a.destino, com_conciliacao=not a.sem_conciliacao,
                      com_risco=not a.sem_risco, com_pos_trade=not a.sem_risco, passo_ordens=a.passo_ordens)
    orq = Orquestrador(cfg)
    t0 = time.time()
    res = orq.reprocessar(a.data) if a.reprocessar else orq.executar(a.de, a.ate)
    print(resumo_ciclo(res))
    print(f"tempo de execução (s): {time.time() - t0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
