"""Motor de conciliação gestor × administrador, diagnóstico automático e controle de cota paralela."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Tolerancias:
    quantidade: float = 0.0  # unidades (exato)
    preco_relativo: float = 1e-6
    caixa_absoluto: float = 0.01  # R$
    cota: float = 1e-8
    provisao_absoluta: float = 0.01  # R$
    pl_absoluto: float = 0.01  # R$
    cotas_relativo: float = 1e-9


COLUNAS_QUEBRA = ["id_quebra", "data", "fundo_cnpj", "tipo", "item", "valor_gestor", "valor_adm", "diferenca",
                  "status", "diagnostico"]


def _quebra(buf: list, estado: dict, tipo: str, item: str, vg: float, va: float, diag: str) -> None:
    buf.append({"data": estado["data"], "fundo_cnpj": estado["cnpj"], "tipo": tipo, "item": item,
                "valor_gestor": vg, "valor_adm": va, "diferenca": va - vg, "status": "ABERTA", "diagnostico": diag})


def _digitos_diferentes(a: float, b: float) -> list[int]:
    ta, tb = f"{a:.10f}", f"{b:.10f}"
    return [k for k, (x, y) in enumerate(zip(ta, tb)) if x != y]


def conciliar_dia(g: dict, a: dict, tol: Tolerancias = Tolerancias()) -> list[dict]:
    """Compara o estado do gestor (g) com o do administrador (a) e devolve as quebras, com diagnóstico provável."""
    q: list[dict] = []
    eventos_g = {(e["tipo"], e["ticker"]): e["valor"] for e in g["eventos"]}
    eventos_a = {(e["tipo"], e["ticker"]): e["valor"] for e in a["eventos"]}
    ordens = {o["ticker"]: o for o in g["ordens_dia"]}

    # ---- evento de data-com/pagamento não creditado (e o caixa que ele explica)
    nao_creditados = {k: v for k, v in eventos_g.items() if k not in eventos_a}
    for (tipo, tk), v in nao_creditados.items():
        _quebra(q, g, "EVENTO", tk, v, 0.0, f"evento de {tipo.lower()} de {tk} (data-com) não creditado pelo administrador")
    for (tipo, tk), v in eventos_a.items():
        if (tipo, tk) not in eventos_g:
            _quebra(q, g, "EVENTO", tk, 0.0, v, f"administrador creditou {tipo.lower()} de {tk} que o gestor não tem")

    # ---- caixa
    soma_nao_creditados = sum(nao_creditados.values())
    for campo, item in (("caixa_disponivel", "caixa_disponivel"), ("caixa_projetado", "caixa_projetado")):
        d = a[campo] - g[campo]
        if abs(d) > tol.caixa_absoluto:
            if nao_creditados and abs(d + soma_nao_creditados) <= max(tol.caixa_absoluto, 1e-9 * abs(soma_nao_creditados)):
                diag = "diferença de caixa igual ao evento não creditado: " + ", ".join(
                    f"{tk} ({v:.3f})" for (_, tk), v in nao_creditados.items())
            elif g["ordens_dia"]:
                diag = "diferença de caixa em dia com ordens executadas: verificar liquidação financeira (" + ", ".join(
                    sorted(o["ordem_id"] for o in g["ordens_dia"])[:3]) + ")"
            else:
                diag = "diferença de caixa sem evento nem ordem no dia: verificar lançamentos manuais / extrato do custodiante"
            _quebra(q, g, "CAIXA", item, g[campo], a[campo], diag)

    # ---- posições
    for tk in sorted(set(g["posicoes"]) | set(a["posicoes"])):
        pg, pa = g["posicoes"].get(tk), a["posicoes"].get(tk)
        if pa is None:
            recente = f" (ordem {ordens[tk]['ordem_id']} no dia)" if tk in ordens else ""
            _quebra(q, g, "ATIVO_FALTANTE", tk, pg["quantidade"], 0.0,
                    f"ativo ausente na carteira do administrador: verificar boleta/liquidação{recente}")
            continue
        if pg is None:
            _quebra(q, g, "ATIVO_SOBRANDO", tk, 0.0, pa["quantidade"],
                    "ativo extra no administrador: posição não reconhecida pelo gestor (verificar boleta/rateio)")
            continue
        if abs(pa["quantidade"] - pg["quantidade"]) > tol.quantidade:
            if tk in ordens:
                diag = f"quantidade divergente em dia de ordem {ordens[tk]['ordem_id']}: rateio/execução não refletido"
            else:
                diag = "quantidade divergente sem movimento do gestor no dia: possível falha de custódia ou evento (amortização)"
            _quebra(q, g, "QUANTIDADE", tk, pg["quantidade"], pa["quantidade"], diag)
        if abs(pa["pu"] / pg["pu"] - 1) > tol.preco_relativo:
            _quebra(q, g, "PRECO", tk, pg["pu"], pa["pu"],
                    f"preço/PU divergente ({(pa['pu'] / pg['pu'] - 1) * 1e4:+.1f} bp): conferir fonte de marcação e curva do dia")

    # ---- cota, PL, quantidade de cotas
    if abs(a["cota"] - g["cota"]) > tol.cota:
        dig = _digitos_diferentes(a["cota"], g["cota"])
        diag = (f"cota difere em {len(dig)} dígito(s) (posição {dig[0]} do texto): provável erro de digitação"
                if len(dig) == 1 else "cota divergente: reflexo de diferença de PL, provisão ou quantidade de cotas")
        _quebra(q, g, "COTA", "cota", g["cota"], a["cota"], diag)
    if abs(a["pl_liquido"] - g["pl_liquido"]) > tol.pl_absoluto:
        _quebra(q, g, "PL", "pl_liquido", g["pl_liquido"], a["pl_liquido"], "PL líquido divergente: ver quebras de posição/caixa/provisão")
    if abs(a["cotas"] / g["cotas"] - 1) > tol.cotas_relativo:
        _quebra(q, g, "QUANTIDADE_COTAS", "cotas", g["cotas"], a["cotas"],
                "quantidade de cotas divergente: conferir conversão de aplicações/resgates e come-cotas")

    # ---- provisões
    nomes = {"ADM": ("PROVISAO_ADM", "provisão de taxa de administração diferente: conferir taxa e base de PL"),
             "PERFORMANCE": ("PROVISAO_PERF", "provisão de performance diferente: conferir benchmark e linha d'água"),
             "RESGATES_A_PAGAR": ("RESGATES_A_PAGAR", "resgates a pagar divergentes: conferir datas de conversão e pagamento")}
    for chave, (tipo, diag) in nomes.items():
        if abs(a["provisoes"][chave] - g["provisoes"][chave]) > tol.provisao_absoluta:
            _quebra(q, g, tipo, chave, g["provisoes"][chave], a["provisoes"][chave], diag)
    return q


def conciliar(estados_gestor: dict, estados_adm: dict, tol: Tolerancias = Tolerancias()) -> pd.DataFrame:
    """Concilia todos os (data, fundo); um arquivo do administrador ausente vira quebra ARQUIVO_AUSENTE."""
    quebras = []
    for k in sorted(estados_gestor, key=lambda k: (k[0], k[1])):
        if k not in estados_adm:
            g = estados_gestor[k]
            _quebra(quebras, g, "ARQUIVO_AUSENTE", "arquivo", 1.0, 0.0, "administrador não enviou o arquivo do dia")
            continue
        quebras += conciliar_dia(estados_gestor[k], estados_adm[k], tol)
    df = pd.DataFrame(quebras, columns=[c for c in COLUNAS_QUEBRA if c != "id_quebra"])
    df.insert(0, "id_quebra", [f"Q{n:05d}" for n in range(1, len(df) + 1)])
    return df


def cota_paralela(estados_gestor: dict, estados_adm: dict) -> pd.DataFrame:
    """Cota do gestor × cota do administrador por dia e fundo, com a diferença em bp."""
    linhas = []
    for k in sorted(estados_gestor, key=lambda k: (k[1], k[0])):
        if k in estados_adm:
            cg, ca = estados_gestor[k]["cota"], estados_adm[k]["cota"]
            linhas.append({"data": k[0], "fundo_cnpj": k[1], "cota_gestor": cg, "cota_adm": ca,
                           "diferenca_bp": (ca / cg - 1) * 1e4})
    return pd.DataFrame(linhas, columns=["data", "fundo_cnpj", "cota_gestor", "cota_adm", "diferenca_bp"])


def avaliar_deteccao(injecoes: pd.DataFrame, quebras: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cada divergência injetada é detectada se há quebra no mesmo (data, fundo) de um tipo esperado (e, quando a
    injeção é de um ativo, do mesmo ativo). Quebras sem injeção correspondente são falsos positivos."""
    from gestorasim.conciliacao.injecao import QUEBRAS_ESPERADAS

    inj = injecoes.copy()
    explicada = pd.Series(False, index=quebras.index)
    detectada = []
    for r in inj.itertuples():
        mesmas = quebras[(quebras["data"] == r.data) & (quebras["fundo_cnpj"] == r.fundo_cnpj)
                         & quebras["tipo"].isin(QUEBRAS_ESPERADAS[r.tipo])]
        if r.tipo in ("QUANTIDADE", "PRECO", "ATIVO_FALTANTE", "ATIVO_SOBRANDO"):
            mesmas = mesmas[mesmas["item"] == r.item]
        elif r.tipo == "EVENTO_NAO_CREDITADO":
            mesmas = mesmas[(mesmas["tipo"] == "CAIXA") | (mesmas["item"] == r.item)]
        detectada.append(len(mesmas) > 0)
        explicada.loc[mesmas.index] = True
    inj["detectada"] = detectada
    return inj, quebras[~explicada]
