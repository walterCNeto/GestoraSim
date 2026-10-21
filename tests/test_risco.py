import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from gestorasim.boletagem import EstadoOrdem, Mesa, OrdemAgregada
from gestorasim.boletagem.dia_exemplo import precos_do_dia, rodar_dia_exemplo
from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.dados import carregar_fundos, gerar_universo
from gestorasim.dominio import ClasseAtivo, ClasseFundo
from gestorasim.enquadramento import montar_carteira
from gestorasim.enquadramento.pos_trade import (
    chave_episodio, nivel_alerta, regras_politica, rodar_pos_trade, utilizacao,
)
from gestorasim.enquadramento.pre_trade import caixa_minimo_projetado, regra_caixa_minimo
from gestorasim.enquadramento.regras import ResultadoRegra, liquidez_d1
from gestorasim.mtm import Precificador, painel_oficial
from gestorasim.mtm.curvas import CAL, VERTICES_RISCO, peso_vertice
from gestorasim.posicao import Ledger, aplicar_execucao, gerar_carteira_inicial
from gestorasim.posicao.caixa import SPREAD_CAIXA_NEGATIVO, juros_caixa
from gestorasim.risco.estatistica import (
    Z99, ewma_cov, kupiec_pof, pnl_delta_gama, var_delta_normal, var_es_historico, zona_basileia,
)
from gestorasim.risco.fatores import construir_fatores
from gestorasim.risco.liquidez import bucket, dias_para_zerar, matriz_liquidez
from gestorasim.risco.motor import metricas_politica
from gestorasim.risco.sensibilidades import CENARIOS, CalculadoraSens, bump_pre, bump_real

SEED = 42


@pytest.fixture(scope="module")
def u(tmp_path_factory):
    return gerar_universo(SEED, tmp_path_factory.mktemp("ur"))


@pytest.fixture(scope="module")
def prec(u):
    return Precificador(u, SEED)


@pytest.fixture(scope="module")
def painel(u):
    return painel_oficial(u, SEED)


@pytest.fixture(scope="module")
def fatores(u, prec, painel):
    return construir_fatores(u, prec, painel)


@pytest.fixture(scope="module")
def calc(prec, fatores, painel):
    return CalculadoraSens(prec, fatores, painel)


# ----------------------------------------------------------------------- estatística

def test_var_parametrico_de_um_ativo_e_z_sigma_v(calc, fatores, painel):
    i, q = 80, 12_000
    delta, _ = calc.carteira({"ALFA3": q}, i)
    cov = fatores.cov[i]
    j = fatores.idx["ACAO:ALFA3"]
    v = q * painel["ALFA3"].iloc[i]
    esperado = Z99 * math.sqrt(cov[j, j]) * v
    assert var_delta_normal(delta, cov) == pytest.approx(esperado, rel=1e-12)
    assert np.count_nonzero(delta) == 1 and delta[j] == pytest.approx(v)


def test_z99_e_quantil_da_normal():
    # Φ(Z99) = 99%
    assert 0.5 * (1 + math.erf(Z99 / math.sqrt(2))) == pytest.approx(0.99, abs=1e-12)


def test_ewma_cov_recursao_e_psd(fatores):
    d_f, cov = fatores.d_f, fatores.cov
    t = 200
    esperado = 0.94 * cov[t - 1] + 0.06 * np.outer(d_f[t], d_f[t])
    assert np.allclose(cov[t], esperado)
    assert np.isnan(cov[10]).all() and not np.isnan(cov[20]).any()
    for t in (30, 250, 503):
        assert np.allclose(cov[t], cov[t].T)
        assert np.linalg.eigvalsh(cov[t]).min() > -1e-12


def test_var_es_historico_em_serie_conhecida():
    pnl = -np.arange(1.0, 1001.0)  # perdas de 1 a 1000
    var, es = var_es_historico(pnl)
    assert var == pytest.approx(990.01, abs=1e-6)  # quantil 1% interpolado
    assert es == pytest.approx(np.mean(np.arange(976.0, 1001.0)), abs=1.0)  # média da cauda de 2,5%
    assert es > var * 0.98


def test_pnl_delta_gama_formula():
    d, g = np.array([2.0, -1.0]), np.array([0.5, 0.2])
    x = np.array([[1.0, 2.0], [0.0, -3.0]])
    assert pnl_delta_gama(d, g, x) == pytest.approx([2 - 2 + 0.5 * (0.5 + 0.8), 3 + 0.5 * 1.8])


def test_kupiec_com_excecoes_esperadas_da_p_proximo_de_1():
    lr, p = kupiec_pof(1000, 10, 0.01)  # 10 exceções = número esperado
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0, abs=1e-6)
    lr, p = kupiec_pof(250, 2, 0.01)
    assert p > 0.5
    assert kupiec_pof(250, 12, 0.01)[1] < 0.001  # muitas exceções: rejeita
    assert 0.02 < kupiec_pof(250, 0, 0.01)[1] < 0.03  # zero exceções em 250 dias: VaR conservador demais (p≈0,025)


def test_zonas_de_basileia_reproduzem_a_tabela_oficial():
    zonas = [zona_basileia(250, x) for x in range(0, 13)]
    assert zonas[:5] == ["VERDE"] * 5 and zonas[5:10] == ["AMARELA"] * 5 and zonas[10:] == ["VERMELHA"] * 3


# ----------------------------------------------------------------------- sensibilidades

def test_pesos_dos_vertices_somam_um():
    for du in (5, 21, 40, 100, 300, 800, 1260, 3000):
        assert sum(peso_vertice(k, du) for k in range(len(VERTICES_RISCO))) == pytest.approx(1.0)


@pytest.mark.parametrize("ticker", ["LTN270104", "LTN290102", "DI1F27", "DI1F28"])
def test_dv01_por_bump_aproxima_duration_modificada_x_pu(prec, calc, painel, ticker):
    i = 150
    est, data = prec.estados[i], prec.datas[i]
    du = CAL.du(data, pd.Timestamp(prec.meta.loc[ticker, "vencimento"]))
    pu = float(painel[ticker].iloc[i])
    dur_mod = (du / 252) / (1 + est.taxa_pre(du))
    delta, _ = calc.sens_unitaria(ticker, i)
    dv01 = sum(delta[calc.idx[f"PRE:{v}"]] for v in VERTICES_RISCO)
    assert dv01 == pytest.approx(-dur_mod * pu * 1e-4, rel=2e-3)
    assert all(delta[calc.idx[f"REAL:{v}"]] == 0 for v in VERTICES_RISCO)


def test_dv01_ntnb_e_debenture_ipca_na_curva_real_e_cs01(prec, calc):
    i = 150
    d, _ = calc.sens_unitaria("NTNB300815", i)
    assert sum(d[calc.idx[f"REAL:{v}"]] for v in VERTICES_RISCO) < 0
    assert all(d[calc.idx[f"PRE:{v}"]] == 0 for v in VERTICES_RISCO)
    d, _ = calc.sens_unitaria("BSAN12", i)  # IPCA+ AA+: real + spread AA+
    assert d[calc.idx["SPREAD:AA+"]] < 0 and sum(d[calc.idx[f"REAL:{v}"]] for v in VERTICES_RISCO) < 0
    assert d[calc.idx["SPREAD:AAA"]] == 0
    d, _ = calc.sens_unitaria("EVAR15", i)  # CDI+ A-: spread A- sensível; curva pré quase nula (pós-fixado)
    assert d[calc.idx["SPREAD:A-"]] < -0.1
    assert abs(sum(d[calc.idx[f"PRE:{v}"]] for v in VERTICES_RISCO)) < 0.05 * abs(d[calc.idx["SPREAD:A-"]])


@pytest.mark.parametrize("ticker,curva", [("LTN270104", "PRE"), ("DI1F27", "PRE"), ("NTNB300815", "REAL"),
                                          ("BSAN12", "REAL")])
@pytest.mark.parametrize("sinal", [+1, -1])
def test_delta_gama_aproxima_reavaliacao_completa_em_10bp(prec, calc, painel, ticker, curva, sinal):
    i = 150
    est, data = prec.estados[i], prec.datas[i]
    delta, gama = calc.sens_unitaria(ticker, i)
    choque = np.zeros(len(delta))
    for v in VERTICES_RISCO:
        choque[calc.idx[f"{curva}:{v}"]] = 10.0 * sinal
    aprox = float(pnl_delta_gama(delta, gama, choque[None, :])[0])
    est2 = est
    for k in range(len(VERTICES_RISCO)):
        est2 = (bump_pre if curva == "PRE" else bump_real)(est2, k, 10.0 * sinal)
    cheio = prec.preco_em_estado(ticker, data, est2) - float(painel[ticker].iloc[i])
    assert abs(aprox / cheio - 1) < 0.01
    # só delta já é decente, o gama melhora
    assert abs(aprox - cheio) <= abs(float(choque @ delta) - cheio) + 1e-12


def test_stress_reavaliacao_completa_sinais_e_acoes(prec, calc, painel):
    i = 150
    ac = calc.stress_unitario("ALFA3", i)
    assert ac["ACOES_-15"] == pytest.approx(-0.15 * painel["ALFA3"].iloc[i]) and ac["PRE_+200"] == 0
    ltn = calc.stress_unitario("LTN270104", i)
    assert ltn["PRE_+200"] < 0 < ltn["PRE_-200"] and ltn["ACOES_-15"] == 0 and ltn["SPREAD_+100"] == 0
    ntnb = calc.stress_unitario("NTNB300815", i)
    assert ntnb["IPCA_IMPL_+100"] > 0  # inflação implícita maior => curva real menor => NTN-B sobe
    deb = calc.stress_unitario("DLOG14", i)
    assert deb["SPREAD_+100"] < 0
    comb = calc.stress_unitario("DLOG14", i)["COMBINADO"]
    assert comb < deb["SPREAD_+100"]  # soma pré +200 e spread +100
    assert set(CENARIOS) == {"PRE_+200", "PRE_-200", "ACOES_-15", "SPREAD_+100", "IPCA_IMPL_+100", "COMBINADO"}


# ----------------------------------------------------------------------- caixa, preço único, pré-trade

def test_caixa_positivo_rende_cdi_e_negativo_paga_cdi_mais_2pct():
    cdi = 0.12
    assert SPREAD_CAIXA_NEGATIVO == 0.02
    assert juros_caixa(1_000_000, cdi) == pytest.approx(1_000_000 * ((1 + cdi) ** (1 / 252) - 1))
    assert juros_caixa(-1_000_000, cdi) == pytest.approx(-1_000_000 * ((1.14) ** (1 / 252) - 1))
    assert juros_caixa(-1_000_000, cdi, 0.05) < juros_caixa(-1_000_000, cdi)  # parâmetro
    assert juros_caixa(0.0, cdi) == 0.0


def test_juros_do_caixa_na_serie(serie_curta, prec):
    led = serie_curta["ledger"]
    j = led[led["origem"] == "JUROS_CAIXA"]
    pl = serie_curta["pl_diario"].set_index(["data", "fundo_cnpj"])
    datas = list(prec.datas)
    for _, r in j.head(30).iterrows():
        i = datas.index(r["data"])
        saldo = pl.loc[(datas[i - 1], r["fundo_cnpj"]), "caixa_projetado"]
        assert r["delta"] == pytest.approx(juros_caixa(saldo, prec.estados[i - 1].cdi_aa))
    assert len(j) == 3 * (len(prec.datas[: prec.datas.get_loc(pd.Timestamp("2023-04-14")) + 1]) - 1)
    assert serie_curta["pnl_atribuicao"]["juros_caixa"].abs().sum() > 0


def test_preco_unico_em_todo_o_sistema(u, painel, serie_curta, tmp_path_factory):
    assert "precos_renda_fixa" not in u
    datas = pd.DatetimeIndex(u["calendario"]["data"])
    # ações: o painel oficial é a própria tabela (já ajustada por dividendos)
    acoes = u["precos_acoes"].pivot(index="data", columns="ticker", values="preco")
    assert np.allclose(painel[acoes.columns].to_numpy(), acoes.to_numpy())
    # bloco 2 (dia de exemplo), mtm_diario e execuções do ledger leem o mesmo preço
    d = datas[1]
    assert precos_do_dia(u, d).equals(painel.loc[d].dropna())
    ex = rodar_dia_exemplo(SEED, tmp_path_factory.mktemp("ex"))["execucoes"]
    ref = painel.loc[d]
    assert np.allclose(ex["fechamento"].to_numpy(), ref[ex["ticker"]].to_numpy())
    m = serie_curta["mtm_diario"]
    ref_m = np.array([painel.at[dt, tk] for dt, tk in zip(m["data"], m["ticker"])])
    assert np.allclose(m["preco"].to_numpy(), ref_m)
    led = serie_curta["ledger"]
    ex_l = led[(led["tipo"] == "ATIVO") & led["origem"].str.fullmatch(r"ORD\d+")]
    ref_l = np.array([painel.at[dt, tk] for dt, tk in zip(ex_l["data"], ex_l["ticker"])])
    dif = (ex_l["preco"].to_numpy() / ref_l - 1) * np.sign(ex_l["delta"].to_numpy())
    assert (dif >= -1e-12).all() and (dif <= 0.0101).all()  # preço de execução = oficial ± slippage (<=1%)


def test_caixa_minimo_projetado_regra_pura():
    assert caixa_minimo_projetado([100.0, 50.0, -10.0], 0.0) == -10.0
    assert caixa_minimo_projetado([100.0, 50.0, 40.0], 30.0) == 10.0  # o choque entra de D+1 em diante
    assert caixa_minimo_projetado([5.0, 50.0, 40.0], 30.0) == 5.0
    assert regra_caixa_minimo([100.0, 50.0, 40.0], 60.0).status == "VIOLADO"
    assert regra_caixa_minimo([100.0, 50.0, 40.0], 10.0).status == "OK"


def _mesa_mm(u, prec, painel, sigma_di, tmp_path_factory):
    destino = tmp_path_factory.mktemp("mm")
    gerar_universo(SEED, destino)
    fundos = carregar_fundos(destino)
    mm = next(f for f in fundos if f.classe == ClasseFundo.MULTIMERCADO)
    led = Ledger()
    gerar_carteira_inicial(led, fundos, prec.meta, painel.iloc[0].dropna(), prec.datas[0], SEED)
    d1 = prec.datas[1]
    vol = gerar_volume(prec.meta, prec.datas, SEED).query("data == @d1").set_index("ticker")["volume"]
    return Mesa(led, fundos, prec.meta, d1, painel.iloc[1].dropna(), vol, sigma_di), mm, led


def test_pre_trade_caixa_minimo_com_choque_de_ajuste_de_di(u, prec, painel, tmp_path_factory):
    pu_di = float(painel["DI1F26"].iloc[1])
    resultados = {}
    for rotulo, sigma in (("sem_choque", None), ("com_choque", {"DI1F26": 0.02, "DI1F27": 0.02, "DI1F28": 0.02})):
        mesa, mm, led = _mesa_mm(u, prec, painel, sigma, tmp_path_factory)
        caixa = led.caixa_projetado(mm.cnpj)
        pl = montar_carteira(led.quantidades(mm.cnpj), caixa, prec.meta, painel.iloc[1].dropna()).pl
        alvo = caixa / 1.01 - 0.15 * 1e6  # sobra ~150 mil de caixa depois da compra (com reserva de 1%)
        q = int(alvo // painel.iloc[1]["LTN270104"])
        o = mesa.processar(OrdemAgregada(id="Z1", data=prec.datas[1].date(), ticker="LTN270104",
                                         classe=ClasseAtivo.TITULO_PUBLICO, lado="C", alvos={mm.cnpj: q}))
        resultados[rotulo] = (o.estado, mesa.tabelas()["pretrade"])
    assert resultados["sem_choque"][0] == EstadoOrdem.ALOCADA
    estado, pre = resultados["com_choque"]
    assert estado == EstadoOrdem.BLOQUEADA
    viol = pre.query("status == 'VIOLADO'")
    assert set(viol["regra"]) == {"caixa_minimo_projetado"}
    assert viol.iloc[0]["valor_pos_ordem"] < 0 < pre.query("regra == 'caixa_suficiente'").iloc[0]["valor_pos_ordem"]
    assert pu_di > 0


# ----------------------------------------------------------------------- liquidez

def test_dias_para_zerar_e_buckets():
    assert dias_para_zerar(1_000, 10_000) == pytest.approx(0.5)  # 1000 / (20% × 10.000)
    assert dias_para_zerar(-4_000, 10_000) == pytest.approx(2.0)
    assert dias_para_zerar(1, 0) == math.inf
    assert [bucket(d) for d in (0.5, 1, 4, 20, 63, 64)] == ["D+1", "D+1", "D+5", "D+21", "D+63", ">D+63"]


def test_matriz_de_liquidez_acumulada_e_monotonica():
    m = matriz_liquidez([(60.0, 0.5), (30.0, 10.0)], caixa=10.0, pl=100.0)
    assert m["liq_d1"] == pytest.approx((10 + 60 + 30 * 0.1) / 100)
    assert m["liq_d5"] == pytest.approx((10 + 60 + 30 * 0.5) / 100)
    assert m["liq_d21"] == pytest.approx(1.0) == m["liq_d63"] and m["liq_gt63"] == pytest.approx(0.0)
    vals = [m[k] for k in ("liq_d1", "liq_d5", "liq_d21", "liq_d63")]
    assert vals == sorted(vals)


def test_liquidez_d1_limita_acoes_a_fracao_liquidavel(prec, painel):
    meta, P = prec.meta, painel.iloc[10].dropna()
    q = {"ALFA3": 1_000_000, "LTN270104": 20_000}
    caixa = 10_000_000.0
    vol_fino = pd.Series({"ALFA3": 1_000_000.0, "LTN270104": 60_000.0})  # 20% = 200 mil papéis de 1 milhão
    c0 = montar_carteira(q, caixa, meta, P)
    c1 = montar_carteira(q, caixa, meta, P, vol_fino)
    assert c1.posicoes.loc["ALFA3", "fracao_d1"] == pytest.approx(0.2)
    assert liquidez_d1(c1) < liquidez_d1(c0, corrigida=False) == pytest.approx(liquidez_d1(c0))
    esperado = (caixa + 0.2 * 1_000_000 * P["ALFA3"] + 20_000 * P["LTN270104"]) / c1.pl
    assert liquidez_d1(c1) == pytest.approx(esperado)


def test_liquidez_corrigida_nunca_maior_que_a_antiga(risco_curto):
    r = risco_curto["risco_diario"]
    assert (r["liquidez_d1"] <= r["liquidez_d1_antes"] + 1e-12).all()
    liq = risco_curto["liquidez"]
    assert (liq["dias_para_zerar"] > 0).all() and set(liq["bucket"]) <= {"D+1", "D+5", "D+21", "D+63", ">D+63"}
    assert (liq.query("ticker == 'GROD13' or ticker == 'EVAR15'")["dias_para_zerar"] > 0).all()
    for c in ("liq_d1", "liq_d5", "liq_d21", "liq_d63"):
        assert r[c].notna().all()
    assert (r["liq_d5"] >= r["liq_d1"] - 1e-12).all() and (r["liq_d63"] >= r["liq_d21"] - 1e-12).all()


# ----------------------------------------------------------------------- pipeline de risco

def test_tabelas_do_risco_e_parquets(risco_curto, destino_serie):
    for nome in ("risco_diario", "sensibilidades", "stress", "backtest_var", "excecoes_var", "liquidez", "enquadramento_diario",
                 "alertas", "episodios"):
        assert nome in risco_curto
        assert (destino_serie / f"{nome}.parquet").exists()
    r = risco_curto["risco_diario"]
    assert {"var_param_99", "var_hist_99", "es_hist_975", "dv01_total", "duration_pl", "stress_pior_pct"} <= set(r)
    assert r["var_param_99"].notna().sum() > 0 and r["var_hist_99"].notna().sum() > 0
    assert set(risco_curto["stress"]["cenario"]) == set(CENARIOS) | {"PIOR_DIA_HIST"}
    assert (risco_curto["sensibilidades"]["fator"].str.split(":").str[0].isin(["ACAO", "PRE", "REAL", "SPREAD"])).all()


def test_var_e_es_coerentes(risco_curto):
    r = risco_curto["risco_diario"].dropna(subset=["var_hist_99"])
    assert (r["var_param_99"].dropna() > 0).all() and (r["var_hist_99"] > 0).all() and (r["es_hist_975"] > 0).all()
    assert (r["var_hist_pct"] < 0.2).all() and (r["var_param_pct"].dropna() < 0.2).all()
    assert (r["stress_pior_pct"] <= 0).all()


def test_stress_acoes_e_combinado(risco_curto, serie_curta):
    st, m = risco_curto["stress"], serie_curta["mtm_diario"]
    fia = serie_curta["pl_diario"].query("fundo == 'Sim Ações FIA'")["fundo_cnpj"].iloc[0]
    d = m["data"].iloc[-1]
    ac_mv = m.query("fundo_cnpj == @fia and data == @d and fonte_preco == 'FECHAMENTO'")["valor_mercado"].sum()
    s = st.query("fundo_cnpj == @fia and data == @d").set_index("cenario")["pnl"]
    assert s["ACOES_-15"] == pytest.approx(-0.15 * ac_mv)
    assert s["COMBINADO"] < s["ACOES_-15"] < 0
    assert s["SPREAD_+100"] == 0  # FIA não tem crédito


def test_dv01_total_e_duration_por_fundo(risco_curto):
    r = risco_curto["risco_diario"]
    por = r.groupby("fundo").agg(dv=("dv01_total", "mean"), dur=("duration_pl", "mean"))
    assert (por["dv"] < 0).all() and (por["dur"] > 0).all()
    assert por.loc["Sim Crédito Privado RF", "dur"] > 0.1  # RF tem a maior duration
    assert (por.loc["Sim Ações FIA", "dur"]) < (por.loc["Sim Crédito Privado RF", "dur"])


def test_backtest_var_excecoes_kupiec_e_zona_limpo_e_sujo(risco_curto):
    bt, r = risco_curto["backtest_var"], risco_curto["risco_diario"]
    assert len(bt) == 12 and set(bt["metodo"]) == {"PARAMETRICO", "HISTORICO"} and set(bt["base"]) == {"LIMPO", "SUJO"}
    for _, b in bt.iterrows():
        g = r[r["fundo_cnpj"] == b["fundo_cnpj"]]
        col = ("excecao_param" if b["metodo"] == "PARAMETRICO" else "excecao_hist") + ("" if b["base"] == "LIMPO" else "_sujo")
        obs = g.loc[~g["em_burn_in"], col].dropna().astype(bool)  # burn-in fora do backtest
        assert b["n_obs"] == len(obs) and b["excecoes"] == int(obs.sum())
        assert b["n_obs_com_burn_in"] == g[col].notna().sum() and b["excecoes_com_burn_in"] >= b["excecoes"]
        assert b["esperado"] == pytest.approx(0.01 * len(obs))
        assert 0 <= b["kupiec_p"] <= 1 and b["zona_basileia"] in ("VERDE", "AMARELA", "VERMELHA", "")
    # a exceção é perda hipotética acima do VaR do dia anterior (limpo = coluna principal; sujo = comparação)
    g = r[r["fundo_cnpj"] == r["fundo_cnpj"].iloc[0]].reset_index(drop=True)
    for k in range(1, len(g)):
        if pd.notna(g.loc[k, "excecao_param"]):
            assert bool(g.loc[k, "excecao_param"]) == (g.loc[k, "pnl_hipotetico_limpo"] < -g.loc[k - 1, "var_param_99"])
            assert bool(g.loc[k, "excecao_param_sujo"]) == (g.loc[k, "pnl_hipotetico"] < -g.loc[k - 1, "var_param_99"])


def test_pnl_hipotetico_limpo_so_efeito_de_fatores(risco_curto, serie_curta):
    r = risco_curto["risco_diario"].set_index(["data", "fundo_cnpj"])
    pnl = serie_curta["pnl_atribuicao"].groupby(["data", "fundo_cnpj"])[
        ["efeito_preco", "carrego", "eventos", "ajuste_futuros", "juros_caixa"]].sum()
    k = [k for k in r.index if k in pnl.index and pd.notna(r.loc[k, "pnl_hipotetico_limpo"])]
    assert len(k) > 100
    dif = r.loc[k, "pnl_hipotetico"] - r.loc[k, "pnl_hipotetico_limpo"]  # sujo − limpo = carrego + eventos de RF
    assert (dif - pnl.loc[k, "carrego"]).abs().max() < pnl.loc[k, "eventos"].abs().max() + 1e-6
    # sem juros do caixa em nenhuma das duas bases
    dias_sem_div = r.loc[k]
    assert np.isfinite(dias_sem_div["pnl_hipotetico_limpo"]).all()


def test_burn_in_e_marcado_e_fora_do_backtest(risco_curto):
    r, ex = risco_curto["risco_diario"], risco_curto["excecoes_var"]
    primeira = r["data"].min()
    pregoes = sorted(r["data"].unique())
    assert (r.loc[r["data"].isin(pregoes[:60]), "em_burn_in"]).all()
    assert not r.loc[r["data"].isin(pregoes[60:]), "em_burn_in"].any()
    assert set(ex.columns) >= {"data", "fundo_cnpj", "metodo", "base", "pnl", "var_dia_anterior", "em_burn_in"}
    assert (ex["pnl"] < -ex["var_dia_anterior"]).all() and ex["data"].min() > primeira
    bt = risco_curto["backtest_var"]
    for _, b in bt.iterrows():  # as exceções dentro do burn-in explicam a diferença com/sem burn-in
        n = len(ex[(ex["fundo_cnpj"] == b["fundo_cnpj"]) & (ex["metodo"] == b["metodo"]) & (ex["base"] == b["base"])])
        assert n == b["excecoes_com_burn_in"]


def test_pnl_hipotetico_bate_com_variacao_do_pl_sem_trades(risco_curto, serie_curta):
    # em dias sem ordens executadas nem juros, PL_t − PL_{t−1} = P&L hipotético + juros do caixa − dividendo por data de pagamento
    r = risco_curto["risco_diario"].set_index(["data", "fundo_cnpj"])
    pnl = serie_curta["pnl_atribuicao"].groupby(["data", "fundo_cnpj"])[
        ["efeito_preco", "carrego", "eventos", "ajuste_futuros", "custos", "juros_caixa"]].sum()
    ok = 0
    for k in r.index[3:]:
        if k in pnl.index and pnl.loc[k, "custos"] == 0 and pd.notna(r.loc[k, "pnl_hipotetico"]):
            ok += 1
            assert np.isfinite(r.loc[k, "pnl_hipotetico"])
    assert ok > 10


# ----------------------------------------------------------------------- pós-trade

def test_utilizacao_e_niveis_de_alerta():
    r = lambda v, lim, st="OK", sent="max": ResultadoRegra("x", "d", v, lim, 0.0, st, sent)  # noqa: E731
    assert utilizacao(r(0.08, 0.10)) == pytest.approx(0.8) and nivel_alerta(r(0.08, 0.10)) == "SOFT"
    assert nivel_alerta(r(0.079, 0.10)) == ""
    assert nivel_alerta(r(0.12, 0.10, "VIOLADO")) == "HARD"
    assert utilizacao(r(0.10, 0.08, sent="min")) == pytest.approx(0.8)  # mínimo: limite/valor
    assert nivel_alerta(r(0.10, 0.08, sent="min")) == "SOFT"
    assert np.isnan(utilizacao(r(0.0, 0.0))) and nivel_alerta(r(0.0, 0.0)) == ""
    assert chave_episodio("limite_emissor", "X") == chave_episodio("limite_emissor", "Y")
    assert chave_episodio("limite_rating", "A-") != chave_episodio("limite_rating", "AA")


def test_politica_de_risco_em_regras():
    res = regras_politica(0.02, 0.15, {"var": 0.03, "stress": 0.10})
    assert [r.regra for r in res] == ["politica_var", "politica_stress"]
    assert [r.status for r in res] == ["OK", "VIOLADO"]
    assert regras_politica(float("nan"), float("nan"), {"var": 1, "stress": 1}) == []


def test_alertas_soft_e_hard_na_serie(risco_curto):
    a, e = risco_curto["alertas"], risco_curto["enquadramento_diario"]
    assert set(a["nivel"]) <= {"SOFT", "HARD"} and (a["nivel"] == "SOFT").any()
    soft = a[a["nivel"] == "SOFT"]
    assert (soft["utilizacao"] >= 0.8).all() and (soft["utilizacao"] < 1).all()
    hard = e[e["alerta"] == "HARD"]
    assert (hard["status"] == "VIOLADO").all()
    assert {"politica_var", "politica_stress", "caixa_negativo", "liquidez_minima"} <= set(e["regra"])


def test_episodios_ativos_so_com_ordem_no_dia(risco_curto, serie_curta):
    ep = risco_curto["episodios"]
    al = serie_curta["alocacoes_serie"]
    com_ordem = set(map(tuple, al.loc[al["quantidade_alocada"] > 0, ["data_serie", "fundo_cnpj"]].to_numpy()))
    ativos = ep[ep["classe"] == "ATIVO"]
    for _, e in ativos.iterrows():
        assert (e["inicio"], e["fundo_cnpj"]) in com_ordem and e["ordens_causa"]
    # sem ordem executada no dia, o desenquadramento que nasce é passivo
    for _, e in ep[ep["classe"] == "PASSIVO"].iterrows():
        assert e["ordens_causa"] == ""
    e = risco_curto["enquadramento_diario"]
    assert (e.loc[e["classificacao"] == "ATIVO", "prazo_du"] == 0).all()


# ----------------------------------------------------------------------- classificação ativo/passivo/vencido (cenário montado)

def _serie_mini(u, prec, painel, n_dias, trades):
    """Duas+ datas: carteira inicial em D0; `trades` = {indice_do_dia: [(fundo_idx, ticker, lado, qtd, origem)]}."""
    fundos = carregar_fundos_cache(u)
    meta = prec.meta
    led = Ledger()
    gerar_carteira_inicial(led, fundos, meta, painel.iloc[0].dropna(), prec.datas[0], SEED)
    pos, pls, aloc, aud = [], [], [], []
    for i in range(n_dias):
        data = prec.datas[i]
        for fi, tk, lado, qtd, origem in trades.get(i, []):
            aplicar_execucao(led, data=data, fundo_cnpj=fundos[fi].cnpj, ticker=tk, classe=meta.loc[tk, "classe"],
                             lado=lado, quantidade=qtd, preco=float(painel.at[data, tk]), origem=origem)
            aloc.append({"ordem_id": origem, "fundo_cnpj": fundos[fi].cnpj, "ticker": tk, "lado": lado,
                         "quantidade_alocada": qtd, "financeiro": qtd * float(painel.at[data, tk]),
                         "data_serie": data})
            aud.append({"ordem_id": origem, "para": "LIBERADA_OVERRIDE", "data_serie": data})
        for f in fundos:
            q = led.quantidades(f.cnpj)
            cart = montar_carteira(q, led.caixa_projetado(f.cnpj), meta, painel.iloc[i].dropna())
            pls.append({"data": data, "fundo_cnpj": f.cnpj, "fundo": f.nome, "pl": cart.pl,
                        "caixa_projetado": led.caixa_projetado(f.cnpj),
                        "caixa_disponivel": led.caixa_disponivel(f.cnpj, data)})
            pos += [{"data": data, "fundo_cnpj": f.cnpj, "ticker": t, "quantidade": v} for t, v in q.items()]
    serie = {"posicoes_diarias": pd.DataFrame(pos), "pl_diario": pd.DataFrame(pls), "ledger": led.para_df(),
             "alocacoes_serie": pd.DataFrame(aloc), "auditoria_serie": pd.DataFrame(aud)}
    risco = serie["pl_diario"][["data", "fundo_cnpj"]].assign(var_param_pct=0.001, stress_pior_pct=-0.01)
    return serie, risco, fundos


def carregar_fundos_cache(u):
    from gestorasim.dados.fundos import gerar_fundos
    return gerar_fundos(u["calendario"]["data"].iloc[0])


def test_desenquadramento_causado_por_ordem_e_ativo_com_a_ordem_na_trilha(u, prec, painel, calc):
    vol = gerar_volume(prec.meta, prec.datas, SEED).pivot(index="data", columns="ticker", values="volume")
    qtd = int(0.05 * 100e6 // painel.at[prec.datas[1], "DLOG14"])
    serie, risco, fundos = _serie_mini(u, prec, painel, 4, {1: [(0, "DLOG14", "C", qtd, "ORD0007")]})  # FIA compra debênture
    res = rodar_pos_trade(serie, risco, prec, painel, vol, fundos, calc)
    ep = res["episodios"]
    fia = fundos[0].cnpj
    e = ep[(ep["fundo_cnpj"] == fia) & (ep["regra"] == "limite_classe") & (ep["detalhe"] == "DEBENTURE")]
    assert len(e) == 1 and e.iloc[0]["classe"] == "ATIVO" and "ORD0007(OVERRIDE)" in e.iloc[0]["ordens_causa"]
    assert e.iloc[0]["inicio"] == prec.datas[1]
    diario = res["enquadramento_diario"]
    linha = diario[(diario["fundo_cnpj"] == fia) & (diario["regra"] == "limite_classe") & (diario["detalhe"] == "DEBENTURE")
                   & (diario["data"] == prec.datas[1])].iloc[0]
    assert linha["status"] == "VIOLADO" and linha["classificacao"] == "ATIVO" and linha["alerta"] == "HARD"
    # os demais fundos, sem ordens, não têm episódio ativo
    assert set(ep.loc[ep["classe"] == "ATIVO", "fundo_cnpj"]) == {fia}


def test_desenquadramento_de_mercado_e_passivo_e_vence_apos_o_prazo(u, prec, painel, calc):
    vol = gerar_volume(prec.meta, prec.datas, SEED).pivot(index="data", columns="ticker", values="volume")
    serie, risco, fundos = _serie_mini(u, prec, painel, 14, {})  # nenhuma ordem
    risco.loc[risco["data"] >= prec.datas[2], "var_param_pct"] = 0.05  # VaR acima da política do RF (0,7%) a partir de D2
    res = rodar_pos_trade(serie, risco, prec, painel, vol, fundos, calc, prazo_reenquadramento_du=10)
    ep = res["episodios"]
    rf = next(f for f in fundos if f.classe == ClasseFundo.RENDA_FIXA)
    e = ep[(ep["fundo_cnpj"] == rf.cnpj) & (ep["regra"] == "politica_var")].iloc[0]
    assert e["classe"] == "PASSIVO" and e["ordens_causa"] == "" and e["inicio"] == prec.datas[2]
    assert e["du_max"] == 11 and bool(e["vencido"])  # 13 pregões - 2 = 11 du > prazo 10
    d = res["enquadramento_diario"]
    d = d[(d["fundo_cnpj"] == rf.cnpj) & (d["regra"] == "politica_var")].set_index("data")
    assert not d.loc[prec.datas[10], "vencido"] and d.loc[prec.datas[10], "du_desde_inicio"] == 8
    assert d.loc[prec.datas[13], "vencido"] and (d["prazo_du"].iloc[2:] == 10).all()
    assert set(res["episodios"].loc[res["episodios"]["classe"] == "ATIVO", "fundo_cnpj"]) == set()


def test_desenquadramento_reenquadrado_encerra_o_episodio(u, prec, painel, calc):
    vol = gerar_volume(prec.meta, prec.datas, SEED).pivot(index="data", columns="ticker", values="volume")
    serie, risco, fundos = _serie_mini(u, prec, painel, 8, {})
    mascara = (risco["data"] >= prec.datas[2]) & (risco["data"] <= prec.datas[4])
    risco.loc[mascara, "var_param_pct"] = 0.05
    ep = rodar_pos_trade(serie, risco, prec, painel, vol, fundos, calc)["episodios"]
    e = ep[ep["regra"] == "politica_var"]
    assert len(e) == 3 and (e["fim"] == prec.datas[5]).all() and not e["vencido"].any()


def test_metricas_de_politica_batem_com_risco(calc, fatores, painel, prec):
    q = {"ALFA3": 50_000, "LTN270104": 10_000}
    i = 100
    m = metricas_politica(calc, fatores, q, i, 1e8)
    delta, _ = calc.carteira(q, i)
    assert m["var_pct"] == pytest.approx(var_delta_normal(delta, fatores.cov[i]) / 1e8)
    assert m["stress_pct"] >= 0


# ----------------------------------------------------------------------- cache do mtm

def test_cache_do_painel_em_memoria_e_em_disco(u, tmp_path):
    from gestorasim.mtm import cache
    p1 = painel_oficial(u, SEED, tmp_path)
    assert (tmp_path / "mtm_painel.parquet").exists() and (tmp_path / "mtm_painel.json").exists()
    assert painel_oficial(u, SEED, tmp_path) is p1  # memória
    cache._MEMO.clear()
    p2 = painel_oficial(u, SEED, tmp_path)  # lido do parquet, sem recalcular
    assert p2 is not p1 and np.allclose(p1.to_numpy(), p2.to_numpy()) and p2.index.equals(p1.index)
    # entradas diferentes => outra impressão digital (cache não é reaproveitado)
    assert cache.impressao_digital(u, SEED) != cache.impressao_digital(u, SEED + 1)
    alterado = {**u, "cdi": u["cdi"].assign(cdi_aa=u["cdi"]["cdi_aa"] + 1e-4)}
    assert cache.impressao_digital(alterado, SEED) != cache.impressao_digital(u, SEED)


# ----------------------------------------------------------------------- ranking das ordens por contribuição

def test_ordens_causa_ranqueadas_pela_contribuicao_a_metrica(u, prec, painel, calc):
    vol = gerar_volume(prec.meta, prec.datas, SEED).pivot(index="data", columns="ticker", values="volume")
    q_grande = int(0.05 * 100e6 // painel.at[prec.datas[1], "DLOG14"])
    q_pequena = int(0.01 * 100e6 // painel.at[prec.datas[1], "DLOG14"])
    # a ordem pequena é criada primeiro (id menor); a grande contribui mais e deve liderar
    serie, risco, fundos = _serie_mini(u, prec, painel, 4, {1: [(0, "DLOG14", "C", q_pequena, "ORD0003"),
                                                                (0, "DLOG14", "C", q_grande, "ORD0009")]})
    ep = rodar_pos_trade(serie, risco, prec, painel, vol, fundos, calc)["episodios"]
    e = ep[(ep["regra"] == "limite_classe") & (ep["detalhe"] == "DEBENTURE") & (ep["classe"] == "ATIVO")].iloc[0]
    assert e["ordem_principal"] == "ORD0009" and e["n_ordens"] == 2
    ranque = e["ordens_causa"].split(",")
    assert ranque[0].startswith("ORD0009") and ranque[1].startswith("ORD0003")
    contrib = [float(x.split("(")[-1].rstrip(")")) for x in ranque]
    assert contrib[0] > contrib[1] > 0
    assert contrib[0] == pytest.approx(0.05, abs=0.002) and contrib[1] == pytest.approx(0.01, abs=0.002)  # frações do PL


def test_ranquear_ordens_unitario():
    from gestorasim.enquadramento.pos_trade import ranquear_ordens
    regra = ResultadoRegra("limite_classe", "ACAO", 0.50, 0.40, -0.10, "VIOLADO", "max")
    sem = {"A": 0.48, "B": 0.40, "C": 0.50}  # valor da regra retirando cada ordem
    alocs = [{"ordem_id": o, "ticker": o, "lado": "C", "quantidade": 1, "financeiro": 1.0, "override": o == "B"} for o in sem]
    texto, principal, n = ranquear_ordens(alocs, regra, lambda a: [ResultadoRegra("limite_classe", "ACAO", sem[a["ordem_id"]], 0.4, 0, "OK")])
    assert principal == "B" and n == 3
    assert texto.split(",")[0].startswith("B(OVERRIDE)") and texto.split(",")[-1].startswith("C(+0.0000")
    minimo = ResultadoRegra("minimo_classe", "ACAO", 0.50, 0.67, -0.17, "VIOLADO", "min")
    t2, p2, _ = ranquear_ordens(alocs[:1], minimo, lambda a: [ResultadoRegra("minimo_classe", "ACAO", 0.55, 0.67, 0, "OK", "min")])
    assert p2 == "A" and "(+0.0500)" in t2  # em regras de mínimo, tirar a ordem melhora => ela piorou 5 pontos
