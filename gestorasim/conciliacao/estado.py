"""Estado diário de cada fundo segundo o gestor (fonte da verdade do sistema), a partir das tabelas da série."""

from __future__ import annotations

import pandas as pd

CHAVE = tuple  # (data, fundo_cnpj)


def construir_estados_gestor(serie: dict[str, pd.DataFrame], meta: pd.DataFrame, datas=None,
                             cnpj_adm: str = "99.999.999/0001-99", nome_adm: str = "Administradora Sim S.A."
                             ) -> dict[CHAVE, dict]:
    """Um dicionário por (data, fundo): cabeçalho/cota, caixa, posições, eventos creditados, provisões e ordens do dia."""
    cota, pl = serie["cota_diaria"], serie["pl_diario"].set_index(["data", "fundo_cnpj"])
    mtm = serie["mtm_diario"]
    led = serie["ledger"]
    evt = led[(led["tipo"] == "CAIXA") & led["origem"].str.startswith("EVT:")]
    eventos: dict[CHAVE, list[dict]] = {}
    for r in evt.itertuples():
        _, tipo, ticker, _ = r.origem.split(":")
        eventos.setdefault((r.data, r.fundo_cnpj), []).append({"tipo": tipo, "ticker": ticker, "valor": float(r.delta)})
    pos: dict[CHAVE, dict[str, dict]] = {}
    for r in mtm.itertuples():
        pos.setdefault((r.data, r.fundo_cnpj), {})[r.ticker] = {
            "classe": meta.loc[r.ticker, "classe"], "quantidade": int(r.qtd), "pu": float(r.preco)}
    al = serie["alocacoes_serie"]
    ordens: dict[CHAVE, list[dict]] = {}
    if len(al):
        for r in al[al["quantidade_alocada"] > 0].itertuples():
            ordens.setdefault((r.data_serie, r.fundo_cnpj), []).append(
                {"ordem_id": r.ordem_id, "ticker": r.ticker, "lado": r.lado, "quantidade": int(r.quantidade_alocada)})
    estados = {}
    alvo = None if datas is None else set(pd.to_datetime(list(datas)))
    for r in cota.itertuples():
        if alvo is not None and r.data not in alvo:
            continue
        k = (r.data, r.fundo_cnpj)
        estados[k] = {
            "data": r.data, "cnpj": r.fundo_cnpj, "nome": r.fundo, "cnpj_adm": cnpj_adm, "nome_adm": nome_adm,
            "cota": float(r.cota_liquida), "pl_liquido": float(r.pl_liquido), "cotas": float(r.cotas),
            "caixa_disponivel": float(pl.loc[k, "caixa_disponivel"]), "caixa_projetado": float(pl.loc[k, "caixa_projetado"]),
            "posicoes": pos.get(k, {}), "eventos": eventos.get(k, []),
            "provisoes": {"ADM": float(r.adm_a_pagar), "PERFORMANCE": float(r.perf_provisao),
                          "RESGATES_A_PAGAR": float(r.resgates_a_pagar)},
            "ordens_dia": ordens.get(k, []),
        }
    return estados
