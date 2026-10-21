"""Regras de enquadramento ao regulamento, puras e reutilizáveis (pré-trade no bloco 2, pós-trade no bloco 4).

Convenções (documentadas porque o MVP ainda não tem margem nem ajuste diário de futuros além do mtm):
- PL = caixa + valor de mercado dos ativos, exceto futuros de DI (valor 0; só exposição nocional).
- Exposição = |quantidade × preço|; percentuais por classe usam exposição / PL.
- Limite por emissor não se aplica a Tesouro Nacional nem a futuros (contraparte central).
- Liquidez D+1 = (caixa + títulos públicos + ações × fração liquidável) / PL, com fração liquidável =
  min(1, 20% do volume do dia / |quantidade|); sem volume informado, a fração é 1 (regra antiga).
- Alavancagem = Σ exposição / PL (caixa fora).
- Rating sem limite próprio herda o do rating pior mais próximo presente no regulamento; pior que todos = 0.

Desempenho: a carteira guarda linhas simples (sem pandas); o DataFrame `posicoes` só é montado se alguém o pedir.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import pandas as pd

from gestorasim.dominio import PARTICIPACAO_LIQUIDEZ, RATINGS, ClasseAtivo, LimitesRegulamento

EPS = 1e-9
OK, VIOLADO = "OK", "VIOLADO"
LIQUIDAS = (ClasseAtivo.ACAO, ClasseAtivo.TITULO_PUBLICO)
ISENTAS_EMISSOR = (ClasseAtivo.TITULO_PUBLICO, ClasseAtivo.FUTURO_DI)
COLUNAS_POSICOES = ["quantidade", "preco", "valor", "exposicao", "classe", "emissor", "rating", "fracao_d1"]
_INDICE_RATING = {r: i for i, r in enumerate(RATINGS)}
_INFO: dict[int, tuple[pd.DataFrame, dict]] = {}


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
        return dict(self.__dict__)


class Linha(NamedTuple):
    ticker: str
    quantidade: int
    preco: float
    valor: float
    exposicao: float
    classe: ClasseAtivo
    emissor: str
    rating: str | None
    fracao_d1: float


@dataclass(eq=False)
class Carteira:
    caixa: float  # projetado
    linhas: list[Linha]
    caixa_disponivel: float | None = None  # liquidado; só informado no pós-trade

    @property
    def pl(self) -> float:
        return self.caixa + self.valor_mercado

    @property
    def valor_mercado(self) -> float:
        return sum(l.valor for l in self.linhas)

    def linha(self, ticker: str) -> Linha:
        for l in self.linhas:
            if l.ticker == ticker:
                return l
        raise KeyError(ticker)

    @cached_property
    def posicoes(self) -> pd.DataFrame:
        """DataFrame (índice = ticker) para quem precisa de operações tabulares; montado sob demanda."""
        if not self.linhas:
            return pd.DataFrame(columns=COLUNAS_POSICOES)
        return pd.DataFrame([l[1:] for l in self.linhas], index=[l.ticker for l in self.linhas],
                            columns=COLUNAS_POSICOES)


def info_ativos(meta: pd.DataFrame) -> dict[str, tuple[ClasseAtivo, str, str | None]]:
    """ticker -> (classe, emissor, rating), em cache no próprio DataFrame (evita `.loc` em colunas de string)."""
    entrada = _INFO.get(id(meta))
    if entrada is None or entrada[0] is not meta:  # `attrs` seria copiado (deepcopy) a cada operação do pandas
        d = {tk: (ClasseAtivo(c), e, None if pd.isna(r) else r)
             for tk, c, e, r in zip(meta.index, meta["classe"], meta["emissor"], meta["rating"])}
        entrada = _INFO[id(meta)] = (meta, d)
    return entrada[1]


def classe_de(meta: pd.DataFrame, ticker: str) -> str:
    """Classe do ativo como string (ex.: "ACAO"), sem `.loc` no DataFrame."""
    return info_ativos(meta)[ticker][0].value


def montar_carteira(quantidades: dict[str, int], caixa: float, meta: pd.DataFrame,
                    precos: pd.Series, volume: pd.Series | None = None,
                    caixa_disponivel: float | None = None) -> Carteira:
    """`meta` indexado por ticker (classe, emissor, rating); `precos` = preço oficial do dia (mtm);
    `volume` = volume do dia (limita a liquidez D+1 das ações)."""
    info = info_ativos(meta)
    linhas = []
    for tk, q in quantidades.items():
        if q == 0:
            continue
        classe, emissor, rating = info[tk]
        px = float(precos[tk])
        nocional = q * px
        fracao = 1.0
        if volume is not None and classe == ClasseAtivo.ACAO:
            fracao = min(1.0, PARTICIPACAO_LIQUIDEZ * float(volume.get(tk, 0.0)) / abs(q))
        linhas.append(Linha(tk, q, px, 0.0 if classe == ClasseAtivo.FUTURO_DI else nocional, abs(nocional),
                            classe, emissor, rating, fracao))
    return Carteira(caixa=float(caixa), linhas=linhas, caixa_disponivel=caixa_disponivel)


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
    vendidas = sum(l.exposicao for l in cart.linhas if l.quantidade < 0 and l.classe != ClasseAtivo.FUTURO_DI)
    return [_res("sem_venda_descoberto", "ativos a vista/titulos", float(vendidas), 0.0)]


def regra_emissor(cart: Carteira, reg: LimitesRegulamento, pl: float | None = None) -> list[ResultadoRegra]:
    """Pior emissor da carteira (apenas o de maior concentração é reportado)."""
    pl = cart.pl if pl is None else pl
    por_emissor: dict[str, float] = {}
    for l in cart.linhas:
        if l.classe not in ISENTAS_EMISSOR:
            por_emissor[l.emissor] = por_emissor.get(l.emissor, 0.0) + l.valor
    if not por_emissor:
        return [_res("limite_emissor", "-", 0.0, reg.limite_emissor)]
    pior = max(por_emissor, key=por_emissor.get)
    return [_res("limite_emissor", str(pior), _frac(por_emissor[pior], pl), reg.limite_emissor)]


def regra_classe(cart: Carteira, reg: LimitesRegulamento, pl: float | None = None) -> list[ResultadoRegra]:
    pl = cart.pl if pl is None else pl
    expo = dict.fromkeys(ClasseAtivo, 0.0)
    for l in cart.linhas:
        expo[l.classe] += l.exposicao
    out = [_res("limite_classe", c.value, _frac(expo[c], pl), reg.limite_classe.get(c, 0.0)) for c in ClasseAtivo]
    for classe, minimo in reg.minimo_classe.items():
        out.append(_res("minimo_classe", classe.value, _frac(expo[classe], pl), minimo, "min"))
    return out


def limite_do_rating(rating: str, limite_rating: dict[str, float]) -> float:
    if rating in limite_rating:
        return limite_rating[rating]
    piores = [r for r in limite_rating if _INDICE_RATING[r] > _INDICE_RATING[rating]]
    if not piores:
        return 0.0
    return limite_rating[min(piores, key=_INDICE_RATING.get)]


def regra_rating(cart: Carteira, reg: LimitesRegulamento, pl: float | None = None) -> list[ResultadoRegra]:
    if not reg.limite_rating:
        return []
    pl = cart.pl if pl is None else pl
    por_rating: dict[str, float] = {}
    for l in cart.linhas:
        if l.classe == ClasseAtivo.DEBENTURE and l.rating is not None:
            por_rating[l.rating] = por_rating.get(l.rating, 0.0) + l.exposicao
    return [_res("limite_rating", r, _frac(v, pl), limite_do_rating(r, reg.limite_rating))
            for r, v in sorted(por_rating.items())]


def liquidez_d1(cart: Carteira, corrigida: bool = True, pl: float | None = None) -> float:
    """Fração do PL liquidável em D+1. `corrigida=False` reproduz a regra do bloco 2 (ações 100%)."""
    liquido = cart.caixa
    for l in cart.linhas:
        if l.classe == ClasseAtivo.ACAO:
            liquido += l.valor * (l.fracao_d1 if corrigida else 1.0)
        elif l.classe == ClasseAtivo.TITULO_PUBLICO:
            liquido += l.valor
    return _frac(liquido, cart.pl if pl is None else pl)


def regra_liquidez(cart: Carteira, reg: LimitesRegulamento, pl: float | None = None) -> list[ResultadoRegra]:
    return [_res("liquidez_minima", "D+1", liquidez_d1(cart, True, pl), reg.liquidez_minima, "min")]


def regra_caixa_negativo(cart: Carteira, pl: float | None = None) -> list[ResultadoRegra]:
    """Pós-trade: saldo liquidado negativo é descoberto e paga CDI + spread (fração do PL >= 0)."""
    if cart.caixa_disponivel is None:
        return []
    return [_res("caixa_negativo", "caixa disponivel", _frac(cart.caixa_disponivel, cart.pl if pl is None else pl),
                 0.0, "min")]


def regra_alavancagem(cart: Carteira, reg: LimitesRegulamento, pl: float | None = None) -> list[ResultadoRegra]:
    return [_res("alavancagem", "exposicao/PL", _frac(sum(l.exposicao for l in cart.linhas),
                                                       cart.pl if pl is None else pl), reg.alavancagem_maxima)]


def todas_as_regras(cart: Carteira, reg: LimitesRegulamento) -> list[ResultadoRegra]:
    pl = cart.pl
    return (regra_caixa(cart) + regra_posicao_vendida(cart) + regra_emissor(cart, reg, pl)
            + regra_classe(cart, reg, pl) + regra_rating(cart, reg, pl) + regra_liquidez(cart, reg, pl)
            + regra_alavancagem(cart, reg, pl) + regra_caixa_negativo(cart, pl))
