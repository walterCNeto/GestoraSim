"""Injeção de divergências rotuladas no arquivo do administrador (seed fixa)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

# tipo injetado -> tipos de quebra que a divergência pode legitimamente produzir
QUEBRAS_ESPERADAS = {
    "QUANTIDADE": {"QUANTIDADE"},
    "PRECO": {"PRECO"},
    "EVENTO_NAO_CREDITADO": {"EVENTO", "CAIXA"},
    "ATIVO_FALTANTE": {"ATIVO_FALTANTE"},
    "ATIVO_SOBRANDO": {"ATIVO_SOBRANDO"},
    "CAIXA": {"CAIXA"},
    "COTA": {"COTA"},
    "PROVISAO_TAXA": {"PROVISAO_ADM"},
}
TIPOS = tuple(QUEBRAS_ESPERADAS)


def _elegiveis(estados: dict, tipo: str) -> list:
    out = []
    for k, e in estados.items():
        if tipo in ("QUANTIDADE", "PRECO", "ATIVO_FALTANTE") and e["posicoes"]:
            out.append(k)
        elif tipo == "EVENTO_NAO_CREDITADO" and e["eventos"]:
            out.append(k)
        elif tipo == "PROVISAO_TAXA" and e["provisoes"]["ADM"] > 1.0:
            out.append(k)
        elif tipo in ("ATIVO_SOBRANDO", "CAIXA", "COTA"):
            out.append(k)
    return sorted(out, key=lambda k: (k[0], k[1]))


def _troca_um_digito(valor: float, rng: np.random.Generator) -> tuple[float, int]:
    """Altera um único dígito decimal (casas 2 a 7) da cota formatada com 10 casas."""
    texto = f"{valor:.10f}"
    ponto = texto.index(".")
    pos = ponto + int(rng.integers(2, 8))
    novo = (int(texto[pos]) + int(rng.integers(1, 10))) % 10
    return float(texto[:pos] + str(novo) + texto[pos + 1:]), pos - ponto


def injetar_divergencias(estados_gestor: dict, seed: int, n_por_tipo: int = 4,
                         tickers_universo: list[str] | None = None, meta: pd.DataFrame | None = None
                         ) -> tuple[dict, pd.DataFrame]:
    """Copia os estados e injeta `n_por_tipo` divergências de cada tipo em pares (data, fundo) distintos.
    Devolve (estados do administrador, tabela de divergências injetadas com rótulo)."""
    rng = np.random.default_rng([seed, 700])
    adm = copy.deepcopy(estados_gestor)
    usados: set = set()
    rotulos = []
    for tipo in TIPOS:
        candidatos = [k for k in _elegiveis(estados_gestor, tipo) if k not in usados]
        if not candidatos:
            continue
        escolhidos = rng.choice(len(candidatos), size=min(n_por_tipo, len(candidatos)), replace=False)
        for idx in escolhidos:
            k = candidatos[int(idx)]
            usados.add(k)
            e, g = adm[k], estados_gestor[k]
            item, original, injetado = "", np.nan, np.nan
            if tipo == "QUANTIDADE":
                item = sorted(e["posicoes"])[int(rng.integers(len(e["posicoes"])))]
                q = e["posicoes"][item]["quantidade"]
                d = max(1, int(round(rng.uniform(0.01, 0.05) * abs(q)))) * int(rng.choice([-1, 1]))
                original, injetado = q, q + d
                e["posicoes"][item]["quantidade"] = q + d
            elif tipo == "PRECO":
                item = sorted(e["posicoes"])[int(rng.integers(len(e["posicoes"])))]
                pu = e["posicoes"][item]["pu"]
                original, injetado = pu, pu * (1 + rng.uniform(0.002, 0.02) * rng.choice([-1, 1]))
                e["posicoes"][item]["pu"] = injetado
            elif tipo == "ATIVO_FALTANTE":
                item = sorted(e["posicoes"])[int(rng.integers(len(e["posicoes"])))]
                original = e["posicoes"][item]["quantidade"]
                del e["posicoes"][item]
            elif tipo == "ATIVO_SOBRANDO":
                livres = [t for t in (tickers_universo or []) if t not in g["posicoes"]]
                item = livres[int(rng.integers(len(livres)))]
                classe = meta.loc[item, "classe"]
                pu = float(rng.uniform(10, 1000))
                e["posicoes"][item] = {"classe": classe, "quantidade": int(rng.integers(100, 5000)), "pu": pu}
                injetado = e["posicoes"][item]["quantidade"]
            elif tipo == "EVENTO_NAO_CREDITADO":
                j = int(rng.integers(len(e["eventos"])))
                ev = e["eventos"].pop(j)
                item, original = ev["ticker"], ev["valor"]
                e["caixa_disponivel"] -= ev["valor"]
                e["caixa_projetado"] -= ev["valor"]
            elif tipo == "CAIXA":
                item, delta = "caixa", float(rng.uniform(1_000, 50_000) * rng.choice([-1, 1]))
                original, injetado = e["caixa_disponivel"], e["caixa_disponivel"] + delta
                e["caixa_disponivel"] += delta
                e["caixa_projetado"] += delta
            elif tipo == "COTA":
                item = "cota"
                original = e["cota"]
                injetado, _ = _troca_um_digito(original, rng)
                e["cota"] = injetado
            elif tipo == "PROVISAO_TAXA":
                item, original = "ADM", e["provisoes"]["ADM"]
                injetado = original * (1 + rng.uniform(0.05, 0.30) * rng.choice([-1, 1]))
                e["provisoes"]["ADM"] = injetado
            rotulos.append({"id_injecao": f"INJ{len(rotulos) + 1:03d}", "data": k[0], "fundo_cnpj": k[1], "tipo": tipo,
                            "item": item, "valor_original": original, "valor_injetado": injetado})
    return adm, pd.DataFrame(rotulos, columns=["id_injecao", "data", "fundo_cnpj", "tipo", "item", "valor_original",
                                               "valor_injetado"])
