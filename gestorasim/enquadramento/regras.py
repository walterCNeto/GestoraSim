"""Regras de enquadramento ao regulamento, puras e reutilizáveis (pré-trade no bloco 2, pós-trade no bloco 4).

Convenções (documentadas porque o MVP ainda não tem MTM nem ajuste diário de futuros):
- PL = caixa + valor de mercado dos ativos, exceto futuros de DI (valor 0; só exposição nocional).
- Exposição = |quantidade × preço|; percentuais por classe usam exposição / PL.
- Limite por emissor não se aplica a Tesouro Nacional nem a futuros (contraparte central).
- Liquidez D+1 = (caixa + ações + títulos públicos) / PL.
- Alavancagem = Σ exposição / PL (caixa fora).
- Rating sem limite próprio herda o do rating pior mais próximo presente no regulamento; pior que todos = 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from gestorasim.dominio import RATINGS, ClasseAtivo, LimitesRegulamento

EPS = 1e-9
OK, VIOLADO = "OK", "VIOLADO"
LIQUIDAS = (ClasseAtivo.ACAO, ClasseAtivo.TITULO_PUBLICO)
ISENTAS_EMISSOR = (ClasseAtivo.TITULO_PUBLICO, ClasseAtivo.FUTURO_DI)
COLUNAS_POSICOES = ["quantidade", "preco", "valor", "exposicao", "classe", "emissor", "rating"]


@dataclass(frozen=True)
class ResultadoRegra:
    regra: str
    detalhe: str
    valor_pos_ordem: float
    limite: float
    folga: float  # positiva = dentro do limite
    status: str
    sentido: str = "max"  # "max": valor <= limite; "min": valor >= limite

    def asdict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Carteira:
    caixa: float
    posicoes: pd.DataFrame  # índice = ticker; colunas COLUNAS_POSICOES

    @property
    def pl(self) -> float:
        return self.caixa + float(self.posicoes["valor"].sum())


def montar_carteira(quantidades: dict[str, int], caixa: float, meta: pd.DataFrame,
                    precos: pd.Series) -> Carteira:
    """`meta` indexado por ticker (classe, emissor, rating); `precos` = fechamento do dia."""
    linhas = {}
    for tk, q in quantidades.items():
        if q == 0:
            continue
        m = meta.loc[tk]
        px = float(precos[tk])
        classe = ClasseAtivo(m["classe"])
        nocional = q * px
        linhas[tk] = {
            "quantidade": q, "preco": px,
            "valor": 0.0 if classe == ClasseAtivo.FUTURO_DI else nocional,
            "exposicao": abs(nocional), "classe": classe, "emissor": m["emissor"],
            "rating": None if pd.isna(m["rating"]) else m["rating"],
        }
    df = pd.DataFrame.from_dict(linhas, orient="index", columns=COLUNAS_POSICOES)
    if df.empty:
        df = pd.DataFrame(columns=COLUNAS_POSICOES)
    return Carteira(caixa=float(caixa), posicoes=df)


def _res(regra, detalhe, valor, limite, sentido="max") -> ResultadoRegra:
    folga = limite - valor if sentido == "max" else valor - limite
    return ResultadoRegra(regra, detalhe, float(valor), float(limite), float(folga),
                          OK if folga >= -EPS else VIOLADO, sentido)


def _frac(x: float, pl: float) -> float:
    return x / pl if pl > EPS else float("inf")


def regra_caixa(cart: Carteira) -> list[ResultadoRegra]:
    return [_res("caixa_suficiente", "caixa", cart.caixa, 0.0, "min")]


def regra_posicao_vendida(cart: Carteira) -> list[ResultadoRegra]:
    """Venda a descoberto só é admitida em futuros."""
    p = cart.posicoes
    vendidas = p[(p["quantidade"] < 0) & (p["classe"] != ClasseAtivo.FUTURO_DI)]
    return [_res("sem_venda_descoberto", "ativos a vista/titulos",
                 float(vendidas["exposicao"].sum()), 0.0)]


def regra_emissor(cart: Carteira, reg: LimitesRegulamento) -> list[ResultadoRegra]:
    """Pior emissor da carteira (apenas o de maior concentração é reportado)."""
    p = cart.posicoes
    p = p[~p["classe"].isin(ISENTAS_EMISSOR)]
    if p.empty:
        return [_res("limite_emissor", "-", 0.0, reg.limite_emissor)]
    por_emissor = p.groupby("emissor")["valor"].sum()
    pior = por_emissor.idxmax()
    return [_res("limite_emissor", str(pior), _frac(por_emissor[pior], cart.pl), reg.limite_emissor)]


def regra_classe(cart: Carteira, reg: LimitesRegulamento) -> list[ResultadoRegra]:
    p = cart.posicoes
    out = []
    for classe in ClasseAtivo:
        pct = _frac(float(p.loc[p["classe"] == classe, "exposicao"].sum()), cart.pl)
        out.append(_res("limite_classe", classe.value, pct, reg.limite_classe.get(classe, 0.0)))
    for classe, minimo in reg.minimo_classe.items():
        pct = _frac(float(p.loc[p["classe"] == classe, "exposicao"].sum()), cart.pl)
        out.append(_res("minimo_classe", classe.value, pct, minimo, "min"))
    return out


def limite_do_rating(rating: str, limite_rating: dict[str, float]) -> float:
    if rating in limite_rating:
        return limite_rating[rating]
    piores = [r for r in limite_rating if RATINGS.index(r) > RATINGS.index(rating)]
    if not piores:
        return 0.0
    return limite_rating[min(piores, key=RATINGS.index)]


def regra_rating(cart: Carteira, reg: LimitesRegulamento) -> list[ResultadoRegra]:
    if not reg.limite_rating:
        return []
    p = cart.posicoes
    deb = p[p["classe"] == ClasseAtivo.DEBENTURE]
    por_rating = deb.groupby("rating")["exposicao"].sum()
    return [
        _res("limite_rating", r, _frac(float(v), cart.pl), limite_do_rating(r, reg.limite_rating))
        for r, v in por_rating.items()
    ]


def regra_liquidez(cart: Carteira, reg: LimitesRegulamento) -> list[ResultadoRegra]:
    p = cart.posicoes
    liquido = cart.caixa + float(p.loc[p["classe"].isin(LIQUIDAS), "valor"].sum())
    return [_res("liquidez_minima", "D+1", _frac(liquido, cart.pl), reg.liquidez_minima, "min")]


def regra_alavancagem(cart: Carteira, reg: LimitesRegulamento) -> list[ResultadoRegra]:
    return [_res("alavancagem", "exposicao/PL", _frac(float(cart.posicoes["exposicao"].sum()), cart.pl),
                 reg.alavancagem_maxima)]


def todas_as_regras(cart: Carteira, reg: LimitesRegulamento) -> list[ResultadoRegra]:
    return (regra_caixa(cart) + regra_posicao_vendida(cart) + regra_emissor(cart, reg)
            + regra_classe(cart, reg) + regra_rating(cart, reg) + regra_liquidez(cart, reg)
            + regra_alavancagem(cart, reg))
