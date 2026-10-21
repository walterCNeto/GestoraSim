import copy

import numpy as np
import pandas as pd
import pytest

from gestorasim.boletagem import EstadoOrdem, Mesa
from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.boletagem.geradores import _Seq
from gestorasim.boletagem.gestao_caixa import ordens_gestao_caixa, ordens_reenquadramento
from gestorasim.conciliacao import Tolerancias, avaliar_deteccao, conciliar, xml_adm
from gestorasim.conciliacao.estado import construir_estados_gestor
from gestorasim.conciliacao.injecao import QUEBRAS_ESPERADAS, injetar_divergencias
from gestorasim.conciliacao.motor import conciliar_dia, cota_paralela
from gestorasim.cota import ParametrosPassivo, PassivoFundo, agendar, come_cotas, resgatar_fifo
from gestorasim.cota.benchmark import indice_acoes
from gestorasim.cota.relatorio import executar_cota, resumo
from gestorasim.cota.tributos import Lote, aliquota_iof, aliquota_ir_regressiva
from gestorasim.dados import calendario, gerar_universo
from gestorasim.dados.fundos import gerar_fundos
from gestorasim.dominio import ClasseFundo
from gestorasim.enquadramento import montar_carteira
from gestorasim.mtm import Precificador, painel_oficial
from gestorasim.posicao import Ledger, aplicar_execucao, gerar_carteira_inicial

SEED = 42
FUNDOS = {f.classe: f for f in gerar_fundos("2023-01-02")}
VAZIO = pd.DataFrame(columns=["id_cotista", "fundo_cnpj", "data", "tipo", "valor"])


def fluxos(classe, *itens):
    """itens = (cotista, 'APLICACAO'|'RESGATE', data, valor)"""
    c = FUNDOS[classe].cnpj
    return pd.DataFrame([(i, c, pd.Timestamp(d), t, v) for i, t, d, v in itens],
                        columns=["id_cotista", "fundo_cnpj", "data", "tipo", "valor"])


def rodar_sintetico(classe, dias, retorno=lambda k: 0.0004, bench_dia=lambda k: 0.0004, fl=None, A0=100e6, params=None):
    """Roda o engine sem ledger: o PL dos ativos evolui por `retorno` e absorve os fluxos de caixa do passivo."""
    params = params or ParametrosPassivo()
    eng = PassivoFundo(FUNDOS[classe], params.regras[classe], params, VAZIO if fl is None else fl)
    A, bench, saldo = A0, 1.0, []
    for k, d in enumerate(calendario.calendario_b3("2023-01-02", dias)):
        if k:
            A *= 1 + retorno(k)
            bench *= 1 + bench_dia(k)
        r = eng.fechar_dia(d, A, bench)
        A += sum(v for v, _ in r.fluxos)
    return eng, pd.DataFrame(eng.cota_rows), pd.DataFrame(eng.prov_rows)


@pytest.fixture(scope="module")
def u(tmp_path_factory):
    return gerar_universo(SEED, tmp_path_factory.mktemp("uc"))


@pytest.fixture(scope="module")
def prec(u):
    return Precificador(u, SEED)


@pytest.fixture(scope="module")
def painel(u):
    return painel_oficial(u, SEED)


# ======================================================================= engine: cota e PL

def test_pl_liquido_igual_cotas_x_cota_sintetico_com_fluxos():
    fl = fluxos(ClasseFundo.RENDA_FIXA,
                ("C1", "APLICACAO", "2023-01-10", 5e6), ("C2", "APLICACAO", "2023-02-14", 2e6),
                ("C1", "RESGATE", "2023-03-20", 1e6), ("C2", "RESGATE", "2023-06-15", 3e6))
    eng, c, _ = rodar_sintetico(ClasseFundo.RENDA_FIXA, 300, lambda k: 0.0005 + 0.002 * np.sin(k), fl=fl)
    assert len(c) == 300
    assert ((c["pl_liquido"] - c["cotas"] * c["cota_liquida"]).abs() <= 1e-8 * c["pl_liquido"]).all()
    assert (c["erro_fechamento_pl"].abs() < 1e-6).all()
    assert c["aplicacoes"].sum() == pytest.approx(7e6) and c["resgates_liquidos"].sum() > 0


def test_cota_bruta_e_liquida_e_provisoes_nao_negativas():
    _, c, p = rodar_sintetico(ClasseFundo.MULTIMERCADO, 200, lambda k: 0.001)
    assert (c["cota_bruta"] >= c["cota_liquida"] - 1e-15).all()
    assert (p["perf_saldo"] >= -1e-9).all() and (p["adm_a_pagar"] >= -1e-9).all()
    assert c["cota_liquida"].iloc[0] == pytest.approx(1.0) and c["cotas"].iloc[0] == pytest.approx(100e6)


# ======================================================================= tributos

def test_come_cotas_reduz_cotas_e_nao_altera_a_cota():
    lote = Lote("L1", "C1", pd.Timestamp("2023-01-02"), cotas=1_000.0, cota_aplicacao=1.0, capital=1_000.0, base_cc=1.0)
    imposto, canc = come_cotas(lote, 1.20, 0.15)
    assert imposto == pytest.approx(0.15 * 200.0) and canc == pytest.approx(30.0 / 1.20)
    assert lote.cotas == pytest.approx(1_000.0 - 25.0)
    assert lote.cotas * 1.20 == pytest.approx(1_200.0 - 30.0)  # valor cai pelo imposto; a cota (1,20) é a mesma
    assert lote.base_cc == 1.20 and lote.cc_pago == pytest.approx(30.0)
    assert come_cotas(lote, 1.10, 0.15) == (0.0, 0.0) and lote.base_cc == 1.20  # perda: sem imposto, base preservada


def test_come_cotas_no_engine_no_ultimo_pregao_de_maio():
    eng, c, _ = rodar_sintetico(ClasseFundo.RENDA_FIXA, 105, lambda k: 0.0008)  # até ~31/05/2023
    cc = c[c["come_cotas"] > 0]
    assert list(cc["data"].dt.strftime("%Y-%m-%d")) == ["2023-05-31"]
    k = cc.index[0]
    antes, dia = c.iloc[k - 1], c.iloc[k]
    cancelado = antes["cotas"] - dia["cotas"]
    assert cancelado > 0 and cancelado * dia["cota_liquida"] == pytest.approx(dia["come_cotas"], rel=1e-9)
    # a cota do dia é a mesma que existiria sem o come-cotas: (PL + imposto) / (cotas + canceladas)
    assert (dia["pl_liquido"] + dia["come_cotas"]) / (dia["cotas"] + cancelado) == pytest.approx(dia["cota_liquida"], rel=1e-12)
    assert dia["pl_liquido"] == pytest.approx(dia["cotas"] * dia["cota_liquida"], rel=1e-12)
    t = pd.DataFrame(eng.tributos)
    assert set(t["tipo"]) == {"COME_COTAS"} and t["valor"].sum() == pytest.approx(dia["come_cotas"])


def test_fia_nao_tem_come_cotas():
    _, c, _ = rodar_sintetico(ClasseFundo.FIA, 130, lambda k: 0.0008)
    assert (c["come_cotas"] == 0).all()


def test_ir_regressiva_fifo_e_abatimento_do_come_cotas():
    assert [aliquota_ir_regressiva(d) for d in (1, 180, 181, 360, 361, 720, 721)] == [0.225, 0.225, 0.20, 0.20, 0.175, 0.175, 0.15]
    d0 = pd.Timestamp("2023-01-02")
    velho = Lote("A", "C1", d0, 100.0, 1.0, 100.0, 1.0)
    novo = Lote("B", "C1", d0 + pd.Timedelta(days=400), 100.0, 1.5, 150.0, 1.5)
    lotes = [velho, novo]
    partes, ir, iof = resgatar_fifo(lotes, 150.0, 2.0, d0 + pd.Timedelta(days=800), "REGRESSIVA")
    assert [p["lote_id"] for p in partes] == ["A", "B"] and [round(p["cotas"], 6) for p in partes] == [100.0, 50.0]
    # A: 800 dias (15%) sobre ganho 100; B: 400 dias (17,5%) sobre 50% de ganho (50 × 0,5 = 25)
    assert ir == pytest.approx(0.15 * 100.0 + 0.175 * 25.0) and iof == 0.0
    assert lotes == [novo] and novo.cotas == pytest.approx(50.0) and novo.capital == pytest.approx(75.0)
    # come-cotas já pago abate o IR (ganho total = valor + cc_pago − capital)
    l = Lote("C", "C2", d0, 80.0, 1.0, 100.0, 1.0, cc_pago=10.0)
    _, ir2, _ = resgatar_fifo([l], 80.0, 2.0, d0 + pd.Timedelta(days=800), "REGRESSIVA")
    assert ir2 == pytest.approx(0.15 * (160.0 + 10.0 - 100.0) - 10.0)
    # FIA: 15% fixo, qualquer prazo; perda não gera imposto nem crédito
    _, ir3, _ = resgatar_fifo([Lote("D", "C3", d0, 10.0, 1.0, 10.0, 1.0)], 10.0, 2.0, d0 + pd.Timedelta(days=10), "FIXA15")
    assert ir3 == pytest.approx(0.15 * 10.0)
    _, ir4, _ = resgatar_fifo([Lote("E", "C3", d0, 10.0, 2.0, 20.0, 2.0)], 10.0, 1.0, d0 + pd.Timedelta(days=10), "FIXA15")
    assert ir4 == 0.0


def test_iof_desligado_por_padrao_e_ligavel():
    d0 = pd.Timestamp("2023-01-02")
    mk = lambda: [Lote("A", "C1", d0, 100.0, 1.0, 100.0, 1.0)]  # noqa: E731
    assert resgatar_fifo(mk(), 100.0, 1.1, d0 + pd.Timedelta(days=10), "REGRESSIVA")[2] == 0.0
    _, ir, iof = resgatar_fifo(mk(), 100.0, 1.1, d0 + pd.Timedelta(days=10), "REGRESSIVA", iof_ligado=True)
    assert iof == pytest.approx(aliquota_iof(10) * 10.0) and aliquota_iof(10) == 0.66
    assert ir == pytest.approx(0.225 * (10.0 - iof)) and aliquota_iof(30) == 0.0
    assert ParametrosPassivo().iof_ligado is False


# ======================================================================= taxas

def test_soma_da_provisao_de_administracao_no_ano_aproxima_pl_medio_x_taxa():
    fl = fluxos(ClasseFundo.MULTIMERCADO, ("C1", "APLICACAO", "2023-03-01", 20e6), ("C1", "RESGATE", "2023-09-04", 5e6))
    eng, c, p = rodar_sintetico(ClasseFundo.MULTIMERCADO, 252, lambda k: 0.0004 + 0.003 * np.sin(k / 5), fl=fl)
    tx = eng.r.taxa_adm
    pl_anterior = c["pl_liquido"].shift(1).iloc[1:]
    esperado = pl_anterior.mean() * tx * 251 / 252
    assert p["adm_dia"].sum() == pytest.approx(esperado, rel=0.015)
    assert p["adm_dia"].iloc[1] == pytest.approx(c["pl_liquido"].iloc[0] * ((1 + tx) ** (1 / 252) - 1))
    # paga no 1º pregão do mês o acumulado do mês anterior (saldo em aberto = só o mês corrente)
    assert p["adm_pago"].sum() == pytest.approx(p["adm_dia"].sum() - p["adm_a_pagar"].iloc[-1])
    pagos = p[p["adm_pago"] > 0]["data"]
    assert all(calendario.somar_du(d, -1).month != d.month for d in pagos)


def test_sem_performance_abaixo_da_linha_dagua():
    # fundo parado contra CDI: nunca passa da linha d'água
    eng, c, p = rodar_sintetico(ClasseFundo.RENDA_FIXA, 190, lambda k: 0.0, bench_dia=lambda k: 0.0004)
    assert (c["cota_bruta"] <= c["linha_dagua"] + 1e-12).all()
    assert (p["perf_saldo"] == 0).all() and (p["perf_pago"] == 0).all() and (p["perf_dia"] == 0).all()
    # sobe acima do benchmark, depois cai abaixo da linha d'água: a provisão volta a zero
    ret = lambda k: 0.0012 if k < 60 else (-0.004 if k < 90 else 0.0)  # noqa: E731
    _, c, p = rodar_sintetico(ClasseFundo.RENDA_FIXA, 100, ret, bench_dia=lambda k: 0.0004)
    acima = c["cota_bruta"] > c["linha_dagua"]
    assert acima.any() and (~acima).any()
    assert (p.loc[~acima, "perf_saldo"] == 0).all() and (p.loc[acima, "perf_saldo"] > 0).all()
    perf_cota = 0.20 * (c["cota_bruta"] - c["linha_dagua"]).clip(lower=0)
    assert np.allclose(c["cota_liquida"], c["cota_bruta"] - perf_cota)
    assert np.allclose(p["perf_saldo"], perf_cota * c["cotas"])  # sem fluxos: provisão = 20% do excesso × cotas


def test_cobranca_semestral_de_performance_e_nova_linha_dagua():
    eng, c, p = rodar_sintetico(ClasseFundo.MULTIMERCADO, 135, lambda k: 0.0010, bench_dia=lambda k: 0.0004)
    pago = p[p["perf_pago"] > 0]
    assert list(pago["data"].dt.strftime("%Y-%m-%d")) == ["2023-06-30"]  # último pregão de junho
    k = pago.index[0]
    assert p["perf_saldo"].iloc[k] == 0 and p["perf_saldo"].iloc[k - 1] > 0
    # PL líquido não muda com a cobrança (caixa e provisão caem juntos) e a linha d'água reinicia na cota líquida
    assert c["pl_liquido"].iloc[k] == pytest.approx(c["cotas"].iloc[k] * c["cota_liquida"].iloc[k])
    assert c["linha_dagua"].iloc[k + 1] == pytest.approx(
        c["cota_liquida"].iloc[k] * c["benchmark"].iloc[k + 1] / c["benchmark"].iloc[k])
    assert c["cota_bruta"].iloc[k + 1] > c["linha_dagua"].iloc[k + 1]


# ======================================================================= conversão e pagamento

def test_resgate_mm_converte_em_d30_corridos_e_paga_d1():
    r = ParametrosPassivo().regras[ClasseFundo.MULTIMERCADO]
    mov = agendar(fluxos(ClasseFundo.MULTIMERCADO, ("C1", "APLICACAO", "2023-03-01", 1e6), ("C1", "RESGATE", "2023-03-01", 5e5),
                         ("C1", "RESGATE", "2023-03-02", 5e5)), r, "t")
    res = mov[mov["tipo"] == "RESGATE"].sort_values("data_solicitacao")
    assert res["data_conversao"].tolist() == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-04-03")]  # 02/03+30d = sáb 01/04
    assert ((res["data_conversao"] - res["data_solicitacao"]).dt.days >= 30).all()
    assert res["data_pagamento"].tolist() == [pd.Timestamp("2023-04-03"), pd.Timestamp("2023-04-04")]
    assert mov.loc[mov["tipo"] == "APLICACAO", "data_conversao"].iloc[0] == pd.Timestamp("2023-03-02")


def test_conversoes_e_pagamentos_por_classe():
    p = ParametrosPassivo().regras
    mk = lambda cl: agendar(fluxos(cl, ("C1", "APLICACAO", "2023-03-01", 1.0), ("C1", "RESGATE", "2023-03-01", 1.0)), p[cl], "t")  # noqa: E731
    rf, mm, fia = (mk(ClasseFundo.RENDA_FIXA), mk(ClasseFundo.MULTIMERCADO), mk(ClasseFundo.FIA))
    d = pd.Timestamp("2023-03-01")
    assert rf["data_conversao"].tolist() == [d, d] and rf["data_pagamento"].iloc[1] == d  # D+0 / D+0 / D+0
    assert mm["data_conversao"].iloc[0] == pd.Timestamp("2023-03-02")
    assert fia["data_conversao"].tolist() == [pd.Timestamp("2023-03-02")] * 2
    assert fia["data_pagamento"].iloc[1] == pd.Timestamp("2023-03-06")  # conversão 02/03 + 2 du


def test_resgate_mm_so_converte_na_data_e_vira_passivo_ate_o_pagamento():
    fl = fluxos(ClasseFundo.MULTIMERCADO, ("C1", "APLICACAO", "2023-01-03", 4e6), ("C1", "RESGATE", "2023-01-10", 1e6))
    eng, c, _ = rodar_sintetico(ClasseFundo.MULTIMERCADO, 50, lambda k: 0.0005, fl=fl)
    conv = pd.Timestamp("2023-02-09")  # 10/01 + 30 dias
    assert (c.loc[c["data"] < conv, "resgates_liquidos"] == 0).all() and (c.loc[c["data"] < conv, "resgates_a_pagar"] == 0).all()
    dia = c[c["data"] == conv].iloc[0]
    assert dia["resgates_liquidos"] > 0 and dia["resgates_a_pagar"] == pytest.approx(
        dia["resgates_liquidos"] + dia["ir_resgate"])  # bruto fica a pagar (IR retido incluído)
    pagto = c[c["data"] == pd.Timestamp("2023-02-10")].iloc[0]  # D+1 após a conversão
    assert pagto["resgates_a_pagar"] == 0
    # PL líquido não depende do passivo estar em caixa ou a pagar
    assert dia["pl_liquido"] == pytest.approx(dia["cotas"] * dia["cota_liquida"], rel=1e-12)
    m = eng.mov[eng.mov["tipo"] == "RESGATE"].iloc[0]
    assert m["status"] == "PAGO" and m["data_conversao"] == conv


def test_resgate_acima_da_posse_e_limitado_e_sem_posse_nao_converte():
    fl = fluxos(ClasseFundo.RENDA_FIXA, ("C1", "APLICACAO", "2023-01-03", 1e6), ("C1", "RESGATE", "2023-01-10", 9e9),
                ("C9", "RESGATE", "2023-01-11", 1e6))
    eng, c, _ = rodar_sintetico(ClasseFundo.RENDA_FIXA, 30, fl=fl)
    r = eng.mov[eng.mov["tipo"] == "RESGATE"].set_index("id_cotista")
    assert r.loc["C9", "status"] == "SEM_SALDO" and r.loc["C1", "status"] == "PAGO"
    assert eng.cotas_do_cotista("C1") == pytest.approx(0.0, abs=1e-6)
    assert r.loc["C1", "valor_bruto"] < 9e9


def test_agenda_de_saidas_previstas():
    fl = fluxos(ClasseFundo.FIA, ("C1", "APLICACAO", "2023-01-03", 5e6), ("C1", "RESGATE", "2023-01-10", 1e6))
    eng = PassivoFundo(FUNDOS[ClasseFundo.FIA], ParametrosPassivo().regras[ClasseFundo.FIA], ParametrosPassivo(), fl)
    datas = list(calendario.calendario_b3("2023-01-11", 5))
    assert eng.saidas_previstas(pd.Timestamp("2023-01-09"), datas) == [0.0] * 5  # pedido ainda não feito
    s = eng.saidas_previstas(pd.Timestamp("2023-01-10"), datas)  # pedido de 10/01: conv 11/01, pago 13/01
    assert s[2] == pytest.approx(-1e6) and sum(s) == pytest.approx(-1e6)


# ======================================================================= benchmark

def test_indice_de_acoes_ponderado_por_valor(u, painel):
    idx = indice_acoes(u, painel, SEED)
    assert idx[0] == 1.0 and len(idx) == 504 and (idx > 0).all()
    assert np.array_equal(idx, indice_acoes(u, painel, SEED)) and not np.array_equal(idx, indice_acoes(u, painel, SEED + 1))
    ret = np.diff(np.log(painel.iloc[:, :20].to_numpy()), axis=0)
    assert ret.min(axis=1).min() < np.diff(np.log(idx)).min() + 0.2  # o índice é uma média (sanidade de escala)


# ======================================================================= série com passivo

def test_serie_pl_liquido_cotas_e_fechamento_do_pnl(serie_curta):
    c, pl = serie_curta["cota_diaria"], serie_curta["pl_diario"]
    assert ((c["pl_liquido"] - c["cotas"] * c["cota_liquida"]).abs() <= 1e-8 * c["pl_liquido"]).all()
    assert c["erro_fechamento_pl"].abs().max() < 1e-6 and pl["erro_fechamento"].abs().max() < 1e-6
    m = c.merge(pl[["data", "fundo_cnpj", "pl", "fluxo_passivo"]], on=["data", "fundo_cnpj"])
    liquido = m["pl"] - m["adm_a_pagar"] - m["perf_provisao"] - m["resgates_a_pagar"]
    assert np.allclose(liquido, m["pl_liquido"], atol=1e-6)  # o PL do ledger menos passivos = PL líquido do engine
    # variação do PL = P&L dos ativos + aplicações − resgates − taxas − tributos retidos
    esperado = m["pnl_ativos"] + m["aplicacoes"] - m["resgates_liquidos"] - m["taxas"] - m["tributos_retidos"]
    depois = m["data"] > m["data"].min()  # o 1º dia de cada fundo só inicializa a cota
    assert depois.sum() > 100 and np.allclose(m.loc[depois, "variacao_pl_liquido"], esperado[depois], atol=1e-6)


def test_pnl_de_ativos_do_passivo_bate_com_a_atribuicao_do_bloco_3(serie_curta):
    c, p = serie_curta["cota_diaria"], serie_curta["pnl_atribuicao"].groupby(["data", "fundo_cnpj"])["total"].sum()
    k = c.set_index(["data", "fundo_cnpj"])["pnl_ativos"].iloc[3:]
    comum = k.index.intersection(p.index)
    assert (k.loc[comum] - p.loc[comum]).abs().max() < 1e-6


def test_movimentacoes_respeitam_as_regras_de_cada_fundo(serie_curta, u):
    mov = serie_curta["cotistas_mov"]
    por = {f.cnpj: f.classe for f in gerar_fundos("2023-01-02")}
    mov = mov.assign(classe=mov["fundo_cnpj"].map(por))
    conv = mov[mov["status"].isin(["CONVERTIDO", "PAGO"])]
    assert len(conv) > 50
    res = conv[conv["tipo"] == "RESGATE"]
    mm = res[res["classe"] == ClasseFundo.MULTIMERCADO]
    assert ((mm["data_conversao"] - mm["data_solicitacao"]).dt.days >= 30).all()
    for cl, n in ((ClasseFundo.RENDA_FIXA, 0), (ClasseFundo.FIA, 1), (ClasseFundo.MULTIMERCADO, 1)):
        a = conv[(conv["tipo"] == "APLICACAO") & (conv["classe"] == cl)]
        assert all(calendario.somar_du(s, n) == c for s, c in zip(a["data_solicitacao"], a["data_conversao"]))
    pagos = res[res["status"] == "PAGO"]
    esperado = {ClasseFundo.RENDA_FIXA: 0, ClasseFundo.MULTIMERCADO: 1, ClasseFundo.FIA: 2}
    assert all(calendario.somar_du(c, esperado[cl]) == p for c, p, cl in zip(pagos["data_conversao"], pagos["data_pagamento"], pagos["classe"]))
    assert (conv["cota_conversao"] > 0).all()
    assert np.allclose(conv.loc[conv["tipo"] == "APLICACAO", "cotas"] * conv.loc[conv["tipo"] == "APLICACAO", "cota_conversao"],
                       conv.loc[conv["tipo"] == "APLICACAO", "valor_solicitado"])


def test_lotes_somam_as_cotas_do_fundo(serie_curta):
    c, lotes = serie_curta["cota_diaria"], serie_curta["lotes_cotistas"]
    final = c.sort_values("data").groupby("fundo_cnpj").last()
    soma = lotes.groupby("fundo_cnpj")["cotas"].sum()
    assert np.allclose(soma, final.loc[soma.index, "cotas"], rtol=1e-12)
    assert (lotes["cotas"] > 0).all() and "CAPITAL" in set(lotes["cotista"])
    assert np.allclose(lotes.groupby("fundo_cnpj")["valor_atual"].sum(), final.loc[soma.index, "pl_liquido"], rtol=1e-9)


def test_aplicacoes_entram_no_caixa_e_pagamentos_saem(serie_curta):
    led = serie_curta["ledger"]
    p = led[led["origem"].str.startswith("PASSIVO:")]
    assert {"PASSIVO:APLICACAO", "PASSIVO:RESGATE"} <= set(p["origem"])
    assert (p.loc[p["origem"] == "PASSIVO:APLICACAO", "delta"] > 0).all() and (p.loc[p["origem"] == "PASSIVO:RESGATE", "delta"] < 0).all()
    pl = serie_curta["pl_diario"]
    flux = p.groupby(["data", "fundo_cnpj"])["delta"].sum()
    comp = pl.set_index(["data", "fundo_cnpj"])["fluxo_passivo"]
    assert np.allclose(flux, comp.loc[flux.index])


# ======================================================================= conciliação

@pytest.fixture(scope="module")
def estados(serie_curta, u):
    return construir_estados_gestor(serie_curta, u["ativos"].set_index("ticker"))


def test_conciliacao_sem_injecao_nao_gera_quebras(serie_curta, u, tmp_path):
    meta = u["ativos"].set_index("ticker")
    r = executar_conciliacao_sem(serie_curta, meta, tmp_path)
    assert len(r["quebras_conciliacao"]) == 0 and len(r["injecoes"]) == 0 and len(r["falsos_positivos"]) == 0
    assert r["cota_paralela"]["diferenca_bp"].abs().max() < 1e-4  # só o arredondamento de 10 casas do XML
    assert len(list((tmp_path / "adm_xml").glob("*.xml"))) == len(serie_curta["cota_diaria"])


def executar_conciliacao_sem(serie, meta, destino):
    from gestorasim.conciliacao import executar_conciliacao
    return executar_conciliacao(serie, meta, SEED, destino, injetar=False)


def test_conciliacao_detecta_100pct_das_divergencias_injetadas(serie_curta, u, tmp_path):
    from gestorasim.conciliacao import executar_conciliacao
    meta = u["ativos"].set_index("ticker")
    r = executar_conciliacao(serie_curta, meta, SEED, tmp_path, injetar=True, n_por_tipo=5)
    inj, q = r["injecoes"], r["quebras_conciliacao"]
    assert set(inj["tipo"]) == set(QUEBRAS_ESPERADAS) and len(inj) == 5 * len(QUEBRAS_ESPERADAS)
    assert inj["detectada"].all()
    assert len(r["falsos_positivos"]) == 0
    assert (q["status"] == "ABERTA").all() and (q["diagnostico"].str.len() > 10).all()
    assert (q["diferenca"] - (q["valor_adm"] - q["valor_gestor"])).abs().max() < 1e-9
    esperado_tipos = set().union(*QUEBRAS_ESPERADAS.values())
    assert set(q["tipo"]) <= esperado_tipos
    # (data, fundo) com injeção = (data, fundo) com quebra
    assert set(zip(q["data"], q["fundo_cnpj"])) == set(zip(inj["data"], inj["fundo_cnpj"]))


def test_diagnosticos_automaticos(serie_curta, u, tmp_path):
    from gestorasim.conciliacao import executar_conciliacao
    r = executar_conciliacao(serie_curta, u["ativos"].set_index("ticker"), SEED, tmp_path, injetar=True, n_por_tipo=4,
                             escrever_xml=False)
    q = r["quebras_conciliacao"]
    d = lambda tipo: " ".join(q.loc[q["tipo"] == tipo, "diagnostico"])  # noqa: E731
    assert "não creditado" in d("EVENTO") and "igual ao evento não creditado" in " ".join(
        q.loc[(q["tipo"] == "CAIXA"), "diagnostico"])
    assert "1 dígito" in d("COTA") and "taxa de administração" in d("PROVISAO_ADM")
    assert "ausente" in d("ATIVO_FALTANTE") and "extra" in d("ATIVO_SOBRANDO") and "preço" in d("PRECO")
    assert "quantidade divergente" in d("QUANTIDADE")


def test_tolerancias_parametrizadas(estados):
    k = next(k for k, e in estados.items() if e["posicoes"])
    g, a = estados[k], copy.deepcopy(estados[k])
    tk = next(iter(g["posicoes"]))
    a["posicoes"][tk]["pu"] *= 1 + 1e-5
    a["caixa_disponivel"] += 0.5
    assert {x["tipo"] for x in conciliar_dia(g, a)} == {"PRECO", "CAIXA"}
    assert conciliar_dia(g, a, Tolerancias(preco_relativo=1e-4, caixa_absoluto=1.0)) == []
    a["cota"] += 5e-9
    assert conciliar_dia(g, a, Tolerancias(preco_relativo=1e-4, caixa_absoluto=1.0)) == []  # abaixo de 1e-8
    a["cota"] += 1e-8
    assert [x["tipo"] for x in conciliar_dia(g, a, Tolerancias(preco_relativo=1e-4, caixa_absoluto=1.0))] == ["COTA"]


def test_injecao_deterministica_e_rotulada(estados, u):
    meta = u["ativos"].set_index("ticker")
    a1, i1 = injetar_divergencias(estados, SEED, 3, list(meta.index), meta)
    a2, i2 = injetar_divergencias(estados, SEED, 3, list(meta.index), meta)
    pd.testing.assert_frame_equal(i1, i2)
    _, i3 = injetar_divergencias(estados, SEED + 1, 3, list(meta.index), meta)
    assert not i1[["data", "fundo_cnpj", "tipo"]].equals(i3[["data", "fundo_cnpj", "tipo"]])
    assert i1["id_injecao"].is_unique and not i1.duplicated(["data", "fundo_cnpj"]).any()
    assert estados[next(iter(estados))] is not a1[next(iter(a1))]  # o gestor não é alterado


def test_cota_com_um_digito_alterado():
    from gestorasim.conciliacao.injecao import _troca_um_digito
    rng = np.random.default_rng(1)
    for v in (1.0234567891, 0.9876543219, 1.5):
        novo, pos = _troca_um_digito(v, rng)
        a, b = f"{v:.10f}", f"{novo:.10f}"
        assert sum(x != y for x, y in zip(a, b)) == 1 and abs(novo - v) >= 1e-8


def test_xml_do_administrador_ida_e_volta(estados, tmp_path):
    k = next(k for k, e in estados.items() if len(e["posicoes"]) > 5 and e["eventos"])
    e = estados[k]
    caminho = xml_adm.gravar(e, tmp_path)
    texto = caminho.read_text(encoding="utf-8")
    for tag in ("<header", "<caixa", "<acoes", "<titpublico", "<debenture", "<futuros", "<provisao", "<eventos"):
        assert tag in texto
    lido = xml_adm.ler(caminho)
    assert lido["posicoes"] == e["posicoes"] and lido["eventos"] == e["eventos"]
    assert lido["caixa_disponivel"] == e["caixa_disponivel"] and lido["provisoes"] == e["provisoes"]
    assert lido["cota"] == pytest.approx(e["cota"], abs=1e-10) and lido["cnpj"] == e["cnpj"]
    assert caminho.name == f"{pd.Timestamp(e['data']):%Y%m%d}_{''.join(ch for ch in e['cnpj'] if ch.isdigit())}.xml"


def test_cota_paralela_em_bp(estados):
    adm = copy.deepcopy(estados)
    k = sorted(adm)[10]
    adm[k]["cota"] *= 1 + 25e-4
    cp = cota_paralela(estados, adm)
    linha = cp[(cp["data"] == k[0]) & (cp["fundo_cnpj"] == k[1])].iloc[0]
    assert linha["diferenca_bp"] == pytest.approx(25.0) and (cp["diferenca_bp"].abs() > 1e-9).sum() == 1


def test_arquivo_ausente_vira_quebra(estados):
    adm = dict(estados)
    k = sorted(adm)[0]
    del adm[k]
    q = conciliar(estados, adm)
    assert len(q) == 1 and q.iloc[0]["tipo"] == "ARQUIVO_AUSENTE"


# ======================================================================= resumo / CLI

def test_resumo_do_cota(serie_curta, tmp_path):
    res = executar_cota(SEED, tmp_path, serie=serie_curta, n_por_tipo=2)
    texto = resumo(res)
    for trecho in ("cota líquida", "taxas cobradas", "tributos retidos", "quebras de conciliação", "taxa de detecção: 1.000"):
        assert trecho in texto
    for nome in ("quebras_conciliacao", "cota_paralela", "injecoes"):
        assert (tmp_path / f"{nome}.parquet").exists()
    assert "0.0" not in texto.split("taxa de detecção:")[1][:4]  # três casas decimais


# ======================================================================= ajustes do bloco 4: caixa e reenquadramento

def _mesa(prec, painel, tmp_path_factory, agenda=None, sigma_di=None):
    destino = tmp_path_factory.mktemp("cx")
    from gestorasim.dados import carregar_fundos
    gerar_universo(SEED, destino)
    fundos = carregar_fundos(destino)
    led = Ledger()
    gerar_carteira_inicial(led, fundos, prec.meta, painel.iloc[0].dropna(), prec.datas[0], SEED)
    d1 = prec.datas[1]
    vol = gerar_volume(prec.meta, prec.datas, SEED).query("data == @d1").set_index("ticker")["volume"]
    mesa = Mesa(led, fundos, prec.meta, d1, painel.iloc[1].dropna(), vol, sigma_di, agenda)
    por = {f.classe: f for f in fundos}
    alvo = {f.cnpj: led.caixa_projetado(f.cnpj) / 100e6 for f in fundos}
    return mesa, por, alvo


def test_perfil_de_caixa_inclui_agenda_do_passivo(prec, painel, tmp_path_factory):
    mesa, por, _ = _mesa(prec, painel, tmp_path_factory)
    c = por[ClasseFundo.FIA].cnpj
    base = mesa.perfil_caixa(c)
    mesa.agenda = {c: [-1e6, 0.0, -2e6]}
    p = mesa.perfil_caixa(c)
    assert p[0] == pytest.approx(base[0] - 1e6) and p[1] == pytest.approx(base[1] - 1e6) and p[2] == pytest.approx(base[2] - 3e6)


def test_gestao_de_caixa_vende_o_ativo_mais_liquido_que_liquida_a_tempo(prec, painel, tmp_path_factory):
    mesa, por, alvo = _mesa(prec, painel, tmp_path_factory)
    fia = por[ClasseFundo.FIA]
    led = mesa.ledger
    led.creditar(mesa.data, fia.cnpj, "CAIXA", -(led.caixa_projetado(fia.cnpj) + 1e6), "teste")  # caixa projetado = −1 mi
    ordens = [o for o in ordens_gestao_caixa(mesa, alvo, _Seq()) if fia.cnpj in o.alvos]
    assert len(ordens) == 1
    o = ordens[0]
    assert o.origem == "GESTAO_CAIXA" and o.lado.value == "V" and o.ticker.startswith("LTN")  # ações só liquidam em D+2
    assert o.alvos[fia.cnpj] * float(mesa.precos[o.ticker]) == pytest.approx(1.02e6, abs=float(mesa.precos[o.ticker]))
    antes = min(mesa.perfil_caixa(fia.cnpj))
    assert antes < 0 and mesa.processar(o).estado == EstadoOrdem.ALOCADA
    assert min(mesa.perfil_caixa(fia.cnpj)) > 0


def test_gestao_de_caixa_usa_buffer_de_3_sigma_do_di(prec, painel, tmp_path_factory):
    sigma = {t: 0.01 for t in prec.meta.index[prec.meta["classe"] == "FUTURO_DI"]}
    mesa, por, alvo = _mesa(prec, painel, tmp_path_factory, sigma_di=sigma)
    mm = por[ClasseFundo.MULTIMERCADO]
    q = mesa.ledger.quantidades(mm.cnpj)
    nocional = sum(abs(v) * float(mesa.precos[t]) for t, v in q.items() if t.startswith("DI1"))
    choque3 = 3 * 0.01 * nocional
    mesa.ledger.creditar(mesa.data, mm.cnpj, "CAIXA", -(mesa.ledger.caixa_projetado(mm.cnpj) - 0.5 * choque3), "teste")
    o = [x for x in ordens_gestao_caixa(mesa, alvo, _Seq()) if mm.cnpj in x.alvos]
    assert len(o) == 1 and o[0].lado.value == "V"
    # o caixa que sobra cobre ≈ o choque de 3σ do DI (a venda recompõe a falta de 0,5 × choque)
    assert o[0].alvos[mm.cnpj] * float(mesa.precos[o[0].ticker]) >= 0.5 * choque3


def test_caixa_excedente_e_reinvestido_nas_posicoes_existentes(prec, painel, tmp_path_factory):
    mesa, por, alvo = _mesa(prec, painel, tmp_path_factory)
    fia = por[ClasseFundo.FIA]
    mesa.ledger.creditar(mesa.data, fia.cnpj, "CAIXA", 30e6, "teste")  # entrada de recursos
    ordens = [o for o in ordens_gestao_caixa(mesa, alvo, _Seq()) if fia.cnpj in o.alvos]
    assert len(ordens) > 5 and all(o.lado.value == "C" and o.origem == "REINVESTIMENTO" for o in ordens)
    held = set(mesa.ledger.quantidades(fia.cnpj))
    assert {o.ticker for o in ordens} <= held
    investido = sum(o.alvos[fia.cnpj] * float(mesa.precos[o.ticker]) for o in ordens)
    pl = montar_carteira(mesa.ledger.quantidades(fia.cnpj), mesa.ledger.caixa_projetado(fia.cnpj), prec.meta, mesa.precos).pl
    sobra = mesa.ledger.caixa_projetado(fia.cnpj) - investido
    assert sobra / pl == pytest.approx(alvo[fia.cnpj] + 0.015, abs=0.005)  # volta ao alvo (+ tolerância)


def test_reenquadramento_reduz_ate_a_folga_de_5pct_e_passa_pelo_pre_trade(prec, painel, tmp_path_factory):
    mesa, por, alvo = _mesa(prec, painel, tmp_path_factory)
    fia = por[ClasseFundo.FIA]
    led, P = mesa.ledger, mesa.precos
    q = led.quantidades(fia.cnpj)
    tk = max((t for t in q if prec.meta.loc[t, "classe"] == "ACAO"), key=lambda t: q[t] * P[t])
    led.integralizar(mesa.data, fia.cnpj, 20e6)  # recursos para a compra (sem mexer em outros limites)
    aplicar_execucao(led, data=mesa.data, fundo_cnpj=fia.cnpj, ticker=tk, classe="ACAO", lado="C",
                     quantidade=int(12e6 // P[tk] // 100 * 100), preco=float(P[tk]), origem="teste")
    pl = montar_carteira(led.quantidades(fia.cnpj), led.caixa_projetado(fia.cnpj), prec.meta, P).pl
    assert led.quantidades(fia.cnpj)[tk] * P[tk] / pl > 0.15  # estourou o limite de emissor
    ordens = [o for o in ordens_reenquadramento(mesa, _Seq(), 1) if fia.cnpj in o.alvos]
    assert len(ordens) == 1 and ordens[0].ticker == tk and ordens[0].lado.value == "V"
    assert ordens[0].origem == "REENQUADRAMENTO:limite_emissor" and ordens[0].usuario == "gestor.reenquadramento"
    n = ordens[0].alvos[fia.cnpj]
    restante = (led.quantidades(fia.cnpj)[tk] - n) * P[tk] / pl
    assert restante == pytest.approx(0.95 * 0.15, abs=0.001)  # folga de 5% sobre o limite de 15%
    o = mesa.processar(ordens[0])
    assert o.estado == EstadoOrdem.ALOCADA
    au = mesa.tabelas()["auditoria"]
    assert "REENQUADRAMENTO" in au.iloc[0]["motivo"] and au.iloc[0]["para"] == "CRIADA"


def test_reenquadramento_nao_gera_ordem_sem_violacao(prec, painel, tmp_path_factory):
    mesa, *_ = _mesa(prec, painel, tmp_path_factory)
    assert ordens_reenquadramento(mesa, _Seq(), 1) == []
