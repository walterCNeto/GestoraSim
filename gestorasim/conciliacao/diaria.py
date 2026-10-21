"""Conciliação de um pregão: arquivo XML do administrador (com injeção determinística por dia) e quebras."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from gestorasim.conciliacao import xml_adm
from gestorasim.conciliacao.estado import construir_estados_gestor
from gestorasim.conciliacao.injecao import TIPOS, aplicar_injecao, elegivel
from gestorasim.conciliacao.motor import Tolerancias, conciliar_dia

TAXA_INJECAO_PADRAO = 0.02  # probabilidade de um (dia, fundo) receber uma divergência


class ConciliacaoDiaria:
    """Sem estado entre pregões: a injeção de cada (dia, fundo) sai de uma seed `[seed, 700, índice do dia, fundo]`,
    o que torna o reprocessamento de uma data idempotente."""

    def __init__(self, seed: int, meta: pd.DataFrame, destino: Path | str | None = None,
                 taxa_injecao: float = TAXA_INJECAO_PADRAO, escrever_xml: bool = True,
                 tol: Tolerancias = Tolerancias()) -> None:
        self.seed, self.meta, self.tol = seed, meta, tol
        self.universo = list(meta.index)
        self.taxa = taxa_injecao
        self.pasta = None if destino is None or not escrever_xml else Path(destino) / "adm_xml"

    def dia(self, i: int, serie_dia: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
        """`serie_dia`: tabelas do pregão (cota_diaria, pl_diario, mtm_diario, ledger, alocacoes_serie)."""
        gestor = construir_estados_gestor(serie_dia, self.meta)
        quebras, injecoes, paralela = [], [], []
        for fi, (k, g) in enumerate(sorted(gestor.items(), key=lambda kv: kv[0][1])):
            adm = copy.deepcopy(g)
            rng = np.random.default_rng([self.seed, 700, i, fi])
            if rng.random() < self.taxa:
                tipos = [t for t in TIPOS if elegivel(t, g)]
                tipo = tipos[int(rng.integers(len(tipos)))]
                item, original, injetado = aplicar_injecao(tipo, adm, g, rng, self.universo, self.meta)
                injecoes.append({"id_injecao": f"INJ-{i:03d}-{fi}", "data": k[0], "fundo_cnpj": k[1], "tipo": tipo,
                                 "item": item, "valor_original": original, "valor_injetado": injetado})
            lido = (xml_adm.ler(xml_adm.gravar(adm, self.pasta)) if self.pasta is not None
                    else xml_adm.de_xml(xml_adm.para_xml(adm)))
            quebras += conciliar_dia(g, lido, self.tol)
            paralela.append({"data": k[0], "fundo_cnpj": k[1], "cota_gestor": g["cota"], "cota_adm": lido["cota"],
                             "diferenca_bp": (lido["cota"] / g["cota"] - 1) * 1e4})
        return {"quebras": quebras, "injecoes": injecoes, "cota_paralela": paralela}
