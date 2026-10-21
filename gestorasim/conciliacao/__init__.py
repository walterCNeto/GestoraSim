"""Conciliação diária gestor × administrador: arquivo XML sintético, divergências injetadas, quebras e cota paralela."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gestorasim.conciliacao import xml_adm
from gestorasim.conciliacao.estado import construir_estados_gestor
from gestorasim.conciliacao.injecao import QUEBRAS_ESPERADAS, injetar_divergencias
from gestorasim.conciliacao.motor import (
    Tolerancias, avaliar_deteccao, conciliar, conciliar_dia, cota_paralela,
)

__all__ = ["QUEBRAS_ESPERADAS", "Tolerancias", "avaliar_deteccao", "conciliar", "conciliar_dia", "construir_estados_gestor",
           "cota_paralela", "executar_conciliacao", "injetar_divergencias", "xml_adm"]


def executar_conciliacao(serie: dict[str, pd.DataFrame], meta: pd.DataFrame, seed: int = 42,
                         destino: Path | str | None = None, injetar: bool = True, n_por_tipo: int = 4,
                         datas=None, tol: Tolerancias = Tolerancias(), escrever_xml: bool = True
                         ) -> dict[str, pd.DataFrame]:
    """Gera o arquivo do administrador (com ou sem divergências injetadas), lê de volta do XML e concilia.

    Devolve quebras_conciliacao, cota_paralela, injecoes (rótulos + detecção) e falsos_positivos.
    """
    gestor = construir_estados_gestor(serie, meta, datas)
    if injetar:
        adm, injecoes = injetar_divergencias(gestor, seed, n_por_tipo, list(meta.index), meta)
    else:
        import copy
        adm, injecoes = copy.deepcopy(gestor), injetar_divergencias({}, seed, 0)[1]
    pasta = None if destino is None else Path(destino) / "adm_xml"
    lidos = {}
    for k, estado in adm.items():
        if pasta is not None and escrever_xml:
            lidos[k] = xml_adm.ler(xml_adm.gravar(estado, pasta))
        else:  # ida e volta em memória, sem disco
            lidos[k] = xml_adm.de_xml(xml_adm.para_xml(estado))
    quebras = conciliar(gestor, lidos, tol)
    inj, falsos = avaliar_deteccao(injecoes, quebras)
    return {"quebras_conciliacao": quebras, "cota_paralela": cota_paralela(gestor, lidos), "injecoes": inj,
            "falsos_positivos": falsos}
