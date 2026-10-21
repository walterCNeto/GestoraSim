import dataclasses

import numpy as np
import pandas as pd
import pytest

from gestorasim.boletagem import EstadoOrdem, Mesa, OrdemAgregada
from gestorasim.boletagem.dia_exemplo import precos_do_dia
from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.dados import calendario, carregar_fundos, gerar_universo
from gestorasim.dominio import ClasseAtivo, ClasseFundo
from gestorasim.mtm import (
    CAL, Precificador, ajuste_di, pu_debenture, pu_di, pu_ltn, pu_ntnb, rodar_serie, serie_pl,
    taxa_flat_forward, vna,
)
from gestorasim.posicao import Ledger, ProcessadorEventos, aplicar_execucao, data_liquidacao

SEED = 42
TOL = 1e-6


@pytest.fixture(scope="module")
def u(tmp_path_factory):
    return gerar_universo(SEED, tmp_path_factory.mktemp("u"))


@pytest.fixture(scope="module")
def prec(u):
    return Precificador(u, SEED)


@pytest.fixture(scope="module")
def painel(prec):
    return prec.painel()


@pytest.fixture(scope="module")
def serie(tmp_path_factory):
    return rodar_serie(SEED, tmp_path_factory.mktemp("serie"), fim="2023-04-14")


# ----------------------------------------------------------------------- curva e funções puras

def test_flat_forward_reproduz_vertices_e_tem_termo_constante(prec):
    e = prec.estados[10]
    for v, t in zip(e.vert_du, e.taxas):
        assert taxa_flat_forward(e.vert_du, e.taxas, v) == pytest.approx(t, abs=1e-12)
    a, b = 252.0, 504.0
    df = lambda du: (1 + taxa_flat_forward(e.vert_du, e.taxas, du)) ** (-du / 252)
    meio = (a + b) / 2
    # taxa a termo constante entre vértices <=> DF(meio)^2 = DF(a) DF(b)
    assert df(meio) ** 2 == pytest.approx(df(a) * df(b), rel=1e-12)
    # fora da grade: constante antes do 1º vértice, prolongada depois do último
    assert taxa_flat_forward(e.vert_du, e.taxas, 5) == pytest.approx(e.taxas[0])
    assert np.isfinite(taxa_flat_forward(e.vert_du, e.taxas, 4000))


def test_pu_ltn_no_vencimento_e_1000(prec):
    est = prec.estados[-1]
    assert pu_ltn(est, 0) == 1000.0
    venc = pd.Timestamp(prec.meta.loc["LTN250701", "vencimento"])
    assert prec.preco_em_estado("LTN250701", venc, est) == pytest.approx(1000.0)
    # abaixo do par antes do vencimento e crescente com o tempo (taxa positiva)
    assert 0 < pu_ltn(est, 500) < pu_ltn(est, 100) < 1000


def test_pu_ntnb_no_vencimento_e_vna_vezes_1(prec):
    est = prec.estados[-1]
    for tk in ("NTNB260817", "NTNB350515"):
        venc = pd.Timestamp(prec.meta.loc[tk, "vencimento"])
        v = pu_ntnb(est, venc, venc, prec.cupons_ntnb[tk], prec.ipca0)
        assert v == pytest.approx(vna(est.ipca_indice, prec.ipca0) * 1.0, abs=1e-9)
    # antes do vencimento: VNA × cotação (cupom 6% descontado a ~8% real ≈ próximo do par)
    tk = "NTNB280815"
    cot = prec.preco(tk, 100) / vna(prec.estados[100].ipca_indice, prec.ipca0)
    assert 0.7 < cot < 1.3


def test_vna_corrigido_pelo_ipca_pro_rata(u, prec):
    ip = u["ipca"]
    assert vna(ip["indice"].iloc[0], prec.ipca0) == 1000.0
    # pro rata no mês: índice cresce todo dia útil, o salto mensal soma a variação do mês
    r = ip["indice"].pct_change().dropna()
    assert (r > -0.005).all() and (r < 0.01).all()


def test_debenture_ao_par_na_emissao_e_ex_cupom(u, prec, painel):
    for tk in prec.deb:
        assert painel[tk].iloc[0] == pytest.approx(1000.0, abs=1e-6)
    ev = u["eventos"]
    for tk in ("ALEN11", "BSAN12", "GROD13"):
        cupons = ev[(ev.ticker == tk) & (ev.tipo == "CUPOM")]
        assert len(cupons) >= 2
        for _, c in cupons.iterrows():
            i = painel.index.get_loc(c["data_com"])
            fluxo = c["valor_por_unidade"] + (ev[(ev.ticker == tk) & (ev.tipo == "AMORTIZACAO")
                                                 & (ev.data_com == c["data_com"])]["valor_por_unidade"].sum())
            # o PU ex-cupom cai o valor pago (±2%: movimento de mercado de um dia)
            assert painel[tk].iloc[i] + fluxo == pytest.approx(painel[tk].iloc[i - 1], rel=0.02)
            assert painel[tk].iloc[i] < painel[tk].iloc[i - 1]


def test_debenture_spread_de_credito_move_o_preco(prec):
    d = prec.deb["EVAR15"]
    est = prec.estados[50]
    alargado = dataclasses.replace(est, z_credito={**est.z_credito, "A-": est.z_credito["A-"] + 0.01})
    p0 = pu_debenture(d, est.data, est, prec.cdi_acum_em, prec.ipca0, prec.ref0["EVAR15"])
    p1 = pu_debenture(d, est.data, alargado, prec.cdi_acum_em, prec.ipca0, prec.ref0["EVAR15"])
    assert p1 < p0


def test_curva_real_e_pre_menos_inflacao_implicita(prec):
    est = prec.estados[30]
    assert est.taxa_real(252) == pytest.approx((1 + est.taxa_pre(252)) / (1 + est.inflacao_implicita) - 1)


def test_painel_sem_nulos_ate_o_vencimento_e_precos_positivos(painel):
    assert painel.shape == (504, 36)
    assert painel.notna().all().all() and (painel > 0).all().all()


# ----------------------------------------------------------------------- futuro de DI

@pytest.mark.parametrize("qtd", [7, -3])
def test_soma_dos_ajustes_do_di_igual_pu_saida_menos_pu_entrada_corrigido(prec, painel, qtd):
    tk, entrada, saida = "DI1F27", 10, 70
    pu = painel[tk]
    soma, k_pos = 0.0, []
    for d in range(entrada + 1, saida + 1):
        k = (1 + prec.estados[d - 1].cdi_aa) ** (1 / 252)
        soma_d = ajuste_di(pu.iloc[d], pu.iloc[d - 1], prec.estados[d - 1].cdi_aa, qtd)
        # cada ajuste é capitalizado pelo CDI até a saída (liquidação em caixa que rende CDI)
        fator = np.prod([(1 + prec.estados[j - 1].cdi_aa) ** (1 / 252) for j in range(d + 1, saida + 1)])
        soma += soma_d * fator
        k_pos.append(k)
    corrigido = pu.iloc[entrada] * np.prod(k_pos)
    assert soma == pytest.approx((pu.iloc[saida] - corrigido) * qtd, rel=1e-9, abs=1e-6)


def test_ajuste_di_sinal_do_vendido():
    comprado = ajuste_di(101_000.0, 100_000.0, 0.10, 5)
    vendido = ajuste_di(101_000.0, 100_000.0, 0.10, -5)
    assert comprado == pytest.approx(-vendido) and comprado > 0
    assert ajuste_di(100_000.0 * 1.1 ** (1 / 252), 100_000.0, 0.10, 3) == pytest.approx(0.0, abs=1e-9)


def test_pu_di_no_vencimento(prec):
    assert pu_di(prec.estados[0], 0) == 100_000.0


# ----------------------------------------------------------------------- eventos no ledger

def _eventos(*linhas):
    return pd.DataFrame(linhas, columns=["tipo", "ticker", "data_com", "data_pagamento", "valor_por_unidade"])


def test_dividendo_so_para_quem_tinha_posicao_na_data_com():
    cal = calendario.calendario_b3("2023-03-01", 30)
    d0, com, pos, pag = cal[0], cal[5], cal[6], cal[15]
    led = Ledger()
    for f in "ABCD":
        led.integralizar(d0, f, 1_000_000)
    ex = lambda d, f, lado, q=100: aplicar_execucao(led, data=d, fundo_cnpj=f, ticker="ALFA3", classe="ACAO",
                                                    lado=lado, quantidade=q, preco=10.0, origem="t")
    ex(d0, "A", "C")
    ex(d0, "C", "C")
    ex(com, "D", "C")  # compra e vende no próprio dia da data-com: zerado no fechamento
    ex(com, "D", "V")
    ex(pos, "B", "C")  # compra depois da data-com: não tem direito
    ex(pos, "C", "V")  # vende depois da data-com: mantém o direito
    p = ProcessadorEventos(led, _eventos(("DIVIDENDO", "ALFA3", com, pag, 0.5)))
    antes = {f: led.caixa_projetado(f) for f in "ABCD"}
    creditos = p.processar(pag, list("ABCD"))
    assert {c["fundo_cnpj"] for c in creditos} == {"A", "C"}
    for f in "AC":
        assert led.caixa_projetado(f) - antes[f] == pytest.approx(100 * 0.5)
    for f in "BD":
        assert led.caixa_projetado(f) == antes[f]
    assert p.processar(pag, list("ABCD")) == []  # idempotente


def test_cupom_vai_para_quem_tinha_posicao_no_pregao_anterior():
    cal = calendario.calendario_b3("2023-03-01", 10)
    led = Ledger()
    for f in "AB":
        led.integralizar(cal[0], f, 1_000_000)
    for f, d in (("A", cal[1]), ("B", cal[5])):  # B compra no dia do pagamento (já ex-cupom)
        aplicar_execucao(led, data=d, fundo_cnpj=f, ticker="NTNB280815", classe="TITULO_PUBLICO",
                         lado="C", quantidade=10, preco=900.0, origem="t")
    p = ProcessadorEventos(led, _eventos(("CUPOM", "NTNB280815", cal[5], cal[5], 29.0)))
    cred = p.processar(cal[5], ["A", "B"])
    assert [c["fundo_cnpj"] for c in cred] == ["A"] and cred[0]["valor"] == pytest.approx(290.0)


def test_amortizacao_credita_caixa_e_reduz_vne():
    cal = calendario.calendario_b3("2023-03-01", 10)
    led = Ledger()
    led.integralizar(cal[0], "A", 1_000_000)
    aplicar_execucao(led, data=cal[1], fundo_cnpj="A", ticker="GROD13", classe="DEBENTURE", lado="C",
                     quantidade=100, preco=1000.0, origem="t")
    p = ProcessadorEventos(led, _eventos(("AMORTIZACAO", "GROD13", cal[5], cal[5], 500.0)),
                           {("GROD13", cal[5]): 0.5})
    antes = led.caixa_projetado("A")
    p.processar(cal[5], ["A"])
    assert led.caixa_projetado("A") - antes == pytest.approx(50_000.0)
    assert led.fator_vne("A", "GROD13") == pytest.approx(0.5)
    assert led.quantidades("A")["GROD13"] == 100  # quantidade de papéis intacta; VNE reduzido


def test_fracao_de_amortizacao_recuperada_dos_eventos(prec):
    fr = [v for (tk, _), v in prec.fracoes.items() if tk == "GROD13"]
    assert fr == [pytest.approx(0.5, abs=1e-9)]


def test_ledger_rejeita_lancamento_fora_de_ordem():
    led = Ledger()
    led.integralizar("2023-01-03", "A", 100.0)
    with pytest.raises(ValueError):
        led.integralizar("2023-01-02", "A", 100.0)


# ----------------------------------------------------------------------- liquidação

def test_liquidacao_d2_acoes_e_d0_titulos():
    cal = calendario.calendario_b3("2023-03-01", 10)
    assert data_liquidacao(cal[0], "ACAO") == cal[2]
    assert data_liquidacao(cal[0], ClasseAtivo.TITULO_PUBLICO) == cal[0]
    assert data_liquidacao(cal[0], "FUTURO_DI") == cal[1]
    led = Ledger()
    led.integralizar(cal[0], "A", 1_000.0)
    aplicar_execucao(led, data=cal[1], fundo_cnpj="A", ticker="ALFA3", classe="ACAO", lado="C", quantidade=10,
                     preco=20.0, origem="c")  # paga 200 em cal[3]
    aplicar_execucao(led, data=cal[1], fundo_cnpj="A", ticker="LTN270104", classe="TITULO_PUBLICO", lado="C",
                     quantidade=1, preco=500.0, origem="c")  # paga 500 já em cal[1]
    assert led.caixa_projetado("A") == pytest.approx(300.0)
    assert led.caixa_disponivel("A", cal[1]) == pytest.approx(1_000.0 - 500.0)
    assert led.caixa_disponivel("A", cal[2]) == pytest.approx(500.0)
    assert led.caixa_disponivel("A", cal[3]) == pytest.approx(300.0)
    # venda de ações: o projetado sobe na hora, o disponível só em D+2
    aplicar_execucao(led, data=cal[4], fundo_cnpj="A", ticker="ALFA3", classe="ACAO", lado="V", quantidade=10,
                     preco=30.0, origem="v")
    assert led.caixa_projetado("A") == pytest.approx(600.0)
    assert led.caixa_disponivel("A", cal[4]) == pytest.approx(300.0)
    assert led.caixa_disponivel("A", cal[5]) == pytest.approx(300.0)
    assert led.caixa_disponivel("A", cal[6]) == pytest.approx(600.0)


def test_pre_trade_usa_caixa_projetado_nao_o_disponivel(tmp_path_factory, u, prec, painel):
    destino = tmp_path_factory.mktemp("pt")
    gerar_universo(SEED, destino)
    fundos = carregar_fundos(destino)
    mm = next(f for f in fundos if f.classe == ClasseFundo.MULTIMERCADO)
    from gestorasim.posicao import gerar_carteira_inicial
    led = Ledger()
    gerar_carteira_inicial(led, fundos, prec.meta, painel.iloc[0], prec.datas[0], SEED)
    d1 = prec.datas[1]
    # em D1 as compras de ações de D0 ainda não liquidaram: disponível > projetado
    assert led.caixa_disponivel(mm.cnpj, d1) > led.caixa_projetado(mm.cnpj) + 1e6
    vol = gerar_volume(prec.meta, prec.datas, SEED).query("data == @d1").set_index("ticker")["volume"]
    mesa = Mesa(led, fundos, prec.meta, d1, painel.iloc[1], vol)
    qtd = int(0.30 * 100e6 // painel.iloc[1]["LTN270104"])  # 30% do PL > caixa projetado (~20%)
    o = mesa.processar(OrdemAgregada(id="X1", data=d1.date(), ticker="LTN270104",
                                     classe=ClasseAtivo.TITULO_PUBLICO, lado="C", alvos={mm.cnpj: qtd}))
    assert o.estado == EstadoOrdem.BLOQUEADA
    pre = mesa.tabelas()["pretrade"]
    assert (pre.query("regra == 'caixa_suficiente'")["status"] == "VIOLADO").all()


# ----------------------------------------------------------------------- série, MTM e P&L

def test_fechamento_da_atribuicao_de_pnl(serie):
    pl = serie["pl_diario"]
    assert pl["erro_fechamento"].abs().max() < TOL
    pnl = serie["pnl_atribuicao"]
    por_dia = pnl.groupby(["data", "fundo_cnpj"])["total"].sum()
    ref = pl.set_index(["data", "fundo_cnpj"])
    assert (por_dia - ref.loc[por_dia.index, "variacao_pl"]).abs().max() < TOL
    # soma por fundo de toda a série = PL final − PL inicial (cotistas = 0)
    for c, g in pl.groupby("fundo_cnpj"):
        assert pnl[pnl.fundo_cnpj == c]["total"].sum() == pytest.approx(g["pl"].iloc[-1] - g["pl"].iloc[0], abs=TOL)


def test_atribuicao_tem_todas_as_componentes(serie, u):
    p = serie["pnl_atribuicao"]
    for col in ("efeito_preco", "carrego", "eventos", "ajuste_futuros", "custos"):
        assert (p[col] != 0).any(), col
    assert (p["custos"] <= 1e-9).all()  # slippage só custa
    acoes = p[p.ticker.isin(u["ativos"].query("classe == 'ACAO'")["ticker"])]
    assert (acoes["carrego"] == 0).all()


def test_mtm_diario_colunas_e_fontes(serie):
    m = serie["mtm_diario"]
    assert list(m.columns[:7]) == ["data", "fundo_cnpj", "ticker", "qtd", "preco", "valor_mercado", "exposicao"]
    assert "fonte_preco" in m.columns
    assert set(m["fonte_preco"]) >= {"FECHAMENTO", "CURVA_PRE_FLAT_FORWARD", "CURVA_PRE_DI",
                                     "CURVA_REF+SPREAD_RATING", "VNA_IPCA+CURVA_REAL"}
    assert np.allclose(m["valor_mercado"].where(m["fonte_preco"] != "CURVA_PRE_DI", 0.0),
                       (m["qtd"] * m["preco"]).where(m["fonte_preco"] != "CURVA_PRE_DI", 0.0))
    assert (m.loc[m.fonte_preco == "CURVA_PRE_DI", "valor_mercado"] == 0).all()


def test_pl_igual_caixa_projetado_mais_mercado_e_liquidacao_aparece(serie):
    pl = serie["pl_diario"]
    assert np.allclose(pl["pl"], pl["caixa_projetado"] + pl["valor_mercado"])
    assert (pl["caixa_disponivel"] != pl["caixa_projetado"]).any()


def test_eventos_creditados_na_serie(serie):
    led = serie["ledger"]
    evt = led[led["origem"].str.startswith("EVT:")]
    assert {"DIVIDENDO"} <= {o.split(":")[1] for o in evt["origem"]}
    assert (evt["delta"] > 0).all() and (evt["data_liquidacao"] == evt["data"]).all()
    assert (led["origem"] == "AJUSTE_DI_DIA").any()
    di = led[led["origem"] == "AJUSTE_DI_DIA"]
    assert (di["data_liquidacao"] > di["data"]).all()  # D+1


def test_serie_pl_de_um_fundo(serie):
    s = serie_pl("Sim Ações FIA", resultado=serie)
    assert s.index.is_monotonic_increasing and len(s) == serie["pl_diario"]["data"].nunique()
    assert s.iloc[0] == pytest.approx(100_000_000.0)
    corte = serie_pl("Sim Ações FIA", inicio="2023-02-01", fim="2023-02-28", resultado=serie)
    assert corte.index.min() >= pd.Timestamp("2023-02-01") and corte.index.max() <= pd.Timestamp("2023-02-28")
    with pytest.raises(KeyError):
        serie_pl("inexistente", resultado=serie)


def test_serie_deterministica_e_parquets(serie, tmp_path_factory):
    d = tmp_path_factory.mktemp("det")
    a = rodar_serie(SEED, d, fim="2023-01-20")
    b = rodar_serie(SEED, tmp_path_factory.mktemp("det2"), fim="2023-01-20", escrever=False)
    for nome in ("pl_diario", "mtm_diario", "pnl_atribuicao", "posicoes_diarias"):
        pd.testing.assert_frame_equal(a[nome], b[nome])
        assert (d / f"{nome}.parquet").exists()
    assert a["pl_diario"]["erro_fechamento"].abs().max() < TOL


def test_ordens_da_serie_inclui_bloqueios(serie):
    oe = dict(zip(serie["ordens_por_estado"]["estado"], serie["ordens_por_estado"]["n"]))
    assert oe.get("ALOCADA", 0) > 0 and oe.get("BLOQUEADA", 0) > 0
