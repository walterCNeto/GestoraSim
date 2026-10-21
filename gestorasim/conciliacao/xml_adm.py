"""Arquivo diário sintético do administrador/custodiante em XML (layout simplificado, inspirado no XML ANBIMA
de posição de carteira): header do fundo, caixa, ações, títulos públicos, debêntures, futuros, provisões, eventos.

Os números são gravados com `repr(float)` (ida e volta exata); o cabeçalho (cota, PL, quantidade de cotas) é um
campo independente e não é recalculado a partir das posições, o que isola cada divergência injetada.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

SECOES = {"ACAO": "acoes", "TITULO_PUBLICO": "titpublico", "DEBENTURE": "debenture", "FUTURO_DI": "futuros"}
ELEMENTO = {"ACAO": "acao", "TITULO_PUBLICO": "titpublico", "DEBENTURE": "debenture", "FUTURO_DI": "futuro"}
CLASSE_DA_SECAO = {v: k for k, v in SECOES.items()}


def _f(x: float) -> str:
    return repr(float(x))


def para_xml(estado: dict) -> ET.Element:
    raiz = ET.Element("arquivoposicao_4_01")
    fundo = ET.SubElement(raiz, "fundo")
    ET.SubElement(fundo, "header", {
        "cnpj": estado["cnpj"], "nome": estado["nome"], "dtposicao": f"{pd.Timestamp(estado['data']):%Y%m%d}",
        "cnpjadm": estado["cnpj_adm"], "nomeadm": estado["nome_adm"], "valorcota": f"{estado['cota']:.10f}",
        "quantidade": _f(estado["cotas"]), "patliq": _f(estado["pl_liquido"])})
    ET.SubElement(fundo, "caixa", {"saldo_disponivel": _f(estado["caixa_disponivel"]),
                                   "saldo_projetado": _f(estado["caixa_projetado"])})
    secoes = {c: ET.SubElement(fundo, nome) for c, nome in SECOES.items()}
    for ticker, p in sorted(estado["posicoes"].items()):
        ET.SubElement(secoes[p["classe"]], ELEMENTO[p["classe"]], {
            "codativo": ticker, "qtdisponivel": str(int(p["quantidade"])), "puposicao": _f(p["pu"]),
            "valorfin": _f(p["quantidade"] * p["pu"])})
    prov = ET.SubElement(fundo, "provisao")
    for tipo, valor in estado["provisoes"].items():
        ET.SubElement(prov, "item", {"tipo": tipo, "valor": _f(valor)})
    evs = ET.SubElement(fundo, "eventos")
    for e in estado["eventos"]:
        ET.SubElement(evs, "evento", {"tipo": e["tipo"], "codativo": e["ticker"], "valor": _f(e["valor"])})
    return raiz


def de_xml(raiz: ET.Element) -> dict:
    fundo = raiz.find("fundo")
    h, cx = fundo.find("header").attrib, fundo.find("caixa").attrib
    posicoes = {}
    for secao, classe in CLASSE_DA_SECAO.items():
        for el in fundo.find(secao):
            posicoes[el.attrib["codativo"]] = {"classe": classe, "quantidade": int(el.attrib["qtdisponivel"]),
                                               "pu": float(el.attrib["puposicao"])}
    return {
        "data": pd.Timestamp(h["dtposicao"]), "cnpj": h["cnpj"], "nome": h["nome"], "cnpj_adm": h["cnpjadm"],
        "nome_adm": h["nomeadm"], "cota": float(h["valorcota"]), "cotas": float(h["quantidade"]),
        "pl_liquido": float(h["patliq"]), "caixa_disponivel": float(cx["saldo_disponivel"]),
        "caixa_projetado": float(cx["saldo_projetado"]), "posicoes": posicoes,
        "provisoes": {i.attrib["tipo"]: float(i.attrib["valor"]) for i in fundo.find("provisao")},
        "eventos": [{"tipo": e.attrib["tipo"], "ticker": e.attrib["codativo"], "valor": float(e.attrib["valor"])}
                    for e in fundo.find("eventos")],
    }


def nome_arquivo(estado: dict) -> str:
    return f"{pd.Timestamp(estado['data']):%Y%m%d}_{''.join(c for c in estado['cnpj'] if c.isdigit())}.xml"


def gravar(estado: dict, pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome_arquivo(estado)
    arvore = ET.ElementTree(para_xml(estado))
    ET.indent(arvore)
    arvore.write(caminho, encoding="utf-8", xml_declaration=True)
    return caminho


def ler(caminho: Path) -> dict:
    return de_xml(ET.parse(caminho).getroot())
