import numpy as np
import pandas as pd
import pytest

from gestorasim.boletagem.geradores import _Seq
from gestorasim.boletagem.gestao_caixa import ordens_gestao_caixa, ordens_reenquadramento
from gestorasim.dominio import ClasseFundo
from gestorasim.mtm import rodar_serie
from gestorasim.orquestrador import ETAPAS, ConfigCiclo, Orquestrador, Simulacao, resumo_ciclo
from gestorasim.orquestrador.ciclo import TABELAS_POR_DIA
from gestorasim.risco import executar_risco

from test_cota import _mesa  # noqa: F401  (helper de ambiente para os testes de caixa/reenquadramento)

SEED = 42
DE, ATE = "2023-01-02", "2023-02-10"
CFG_CURTA = dict(janela_hist=15, burn_in=5, ckpt_cada=10)


@pytest.fixture(scope="module")
def prec(tmp_path_factory):
    from gestorasim.dados import gerar_universo
    from gestorasim.mtm import Precificador
    return Precificador(gerar_universo(SEED, tmp_path_factory.mktemp("uo")), SEED)


@pytest.fixture(scope="module")
def painel(prec, tmp_path_factory):
    from gestorasim.dados import gerar_universo
    from gestorasim.mtm import painel_oficial
    return painel_oficial(gerar_universo(SEED, tmp_path_factory.mktemp("up")), SEED)


def cfg(destino, **kw):
    return ConfigCiclo(seed=SEED, destino=destino, **{**CFG_CURTA, **kw})


def quase_igual(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-9, atol=1e-6):
    """Mesmas linhas e colunas; números com tolerância, o resto exato."""
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in a.columns:
        if pd.api.types.is_float_dtype(a[c]):
            np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=rtol, atol=atol, equal_nan=True,
                                       err_msg=c)
        else:
            assert a[c].astype(str).tolist() == b[c].astype(str).tolist(), c


# ======================================================================= etapas e log

def test_treze_etapas_na_ordem_com_status_duracao_e_hash(ciclo_curto):
    _, res, _ = ciclo_curto
    log = res["execucoes_ciclo"]
    assert ETAPAS == ("abertura", "mercado", "eventos", "passivo", "caixa_reenquadramento", "boletagem", "liquidacao",
                      "mtm", "provisoes_cota", "conciliacao", "pos_trade", "risco", "relatorios")
    assert list(log.columns) == ["data", "etapa", "ordem", "status", "duracao_s", "hash_entrada", "mensagem"]
    n_dias = log["data"].nunique()
    assert n_dias == 30 and len(log) == 13 * n_dias
    for _, g in log.groupby("data"):
        assert g["etapa"].tolist() == list(ETAPAS) and g["ordem"].tolist() == list(range(1, 14))
    assert set(log["status"]) <= {"OK", "PULADA", "ERRO"} and not (log["status"] == "ERRO").any()
    ok = log[log["status"] == "OK"]
    assert (ok["hash_entrada"].str.len() == 16).all() and (ok["duracao_s"] >= 0).all()
    primeiro = log[log["data"] == log["data"].min()].set_index("etapa")["status"]
    assert primeiro[["abertura", "mercado", "eventos", "caixa_reenquadramento", "boletagem"]].eq("PULADA").all()
    assert primeiro[["passivo", "mtm", "provisoes_cota", "conciliacao", "pos_trade", "risco"]].eq("OK").all()
    assert (log[log["data"] > log["data"].min()]["status"] == "OK").all()
    assert "total" in resumo_ciclo(res)


def test_tabelas_do_ciclo_fecham_e_usam_pl_liquido(ciclo_curto):
    _, res, _ = ciclo_curto
    assert res["pl_diario"]["erro_fechamento"].abs().max() < 1e-6
    assert res["cota_diaria"]["erro_fechamento_pl"].abs().max() < 1e-6
    # risco em % do PL líquido (cota_diaria), não do PL dos ativos
    r = res["risco_diario"].merge(res["cota_diaria"][["data", "fundo_cnpj", "pl_liquido", "pl_ativos"]], on=["data", "fundo_cnpj"])
    assert np.allclose(r["pl"], r["pl_liquido"]) and not np.allclose(r["pl"], r["pl_ativos"])
    ok = r.dropna(subset=["var_param_99"])
    assert np.allclose(ok["var_param_pct"] * ok["pl_liquido"], ok["var_param_99"])
    st = res["stress"].merge(res["cota_diaria"][["data", "fundo_cnpj", "pl_liquido"]], on=["data", "fundo_cnpj"])
    s = st.dropna(subset=["pnl_pct_pl"])
    assert np.allclose(s["pnl_pct_pl"] * s["pl_liquido"], s["pnl"])


def test_ciclo_reproduz_a_serie_sem_as_etapas_finais(ciclo_curto, tmp_path):
    """Conciliação, pós-trade, risco e relatórios não mexem na carteira: rodar_serie dá as mesmas posições e cotas."""
    _, res, _ = ciclo_curto
    s = rodar_serie(SEED, tmp_path, fim=ATE, escrever=False)
    for nome in ("cota_diaria", "pl_diario", "pnl_atribuicao", "posicoes_diarias"):
        quase_igual(res[nome].reset_index(drop=True), s[nome].reset_index(drop=True))


def test_risco_do_ciclo_igual_ao_calculo_em_lote(ciclo_curto, tmp_path):
    _, res, _ = ciclo_curto
    serie = {n: res[n] for n in ("posicoes_diarias", "pl_diario", "cota_diaria", "pnl_atribuicao", "ledger",
                                 "alocacoes_serie", "auditoria_serie")}
    lote = executar_risco(SEED, tmp_path, serie=serie, janela=15, burn_in=5, escrever=False)
    cols = ["data", "fundo_cnpj", "pl", "var_param_99", "var_param_pct", "var_hist_99", "es_hist_975", "dv01_total",
            "duration_pl", "stress_pior", "liquidez_d1", "pnl_hipotetico_limpo", "pnl_hipotetico"]
    quase_igual(res["risco_diario"][cols].reset_index(drop=True), lote["risco_diario"][cols].reset_index(drop=True),
                rtol=1e-7)
    assert res["risco_diario"]["em_burn_in"].tolist() == lote["risco_diario"]["em_burn_in"].tolist()
    quase_igual(res["stress"], lote["stress"], rtol=1e-7)
    ep = lambda t: t[["fundo_cnpj", "regra", "detalhe", "classe", "inicio", "ordem_principal"]].astype(str)  # noqa: E731
    assert ep(res["episodios"]).values.tolist() == ep(lote["episodios"]).values.tolist()
    e1 = res["enquadramento_diario"][["data", "fundo_cnpj", "regra", "detalhe", "status", "alerta", "classificacao"]]
    e2 = lote["enquadramento_diario"][["data", "fundo_cnpj", "regra", "detalhe", "status", "alerta", "classificacao"]]
    assert e1.astype(str).values.tolist() == e2.astype(str).values.tolist()


def test_conciliacao_do_ciclo_sem_falsos_positivos(ciclo_curto):
    _, res, _ = ciclo_curto
    from gestorasim.conciliacao import avaliar_deteccao
    inj, falsos = avaliar_deteccao(res["injecoes"], res["quebras_conciliacao"])
    assert len(res["cota_paralela"]) == len(res["cota_diaria"])
    assert falsos.empty and (inj["detectada"].all() if len(inj) else True)
    q = res["quebras_conciliacao"]
    assert q["id_quebra"].is_unique and q["id_quebra"].str.startswith("Q").all()


# ======================================================================= idempotência

def _tabelas_gravadas(orq: Orquestrador):
    t = orq.carregar()
    return {k: v.copy() for k, v in t.items()}


def test_reprocessar_uma_data_nao_altera_as_anteriores_e_reproduz_as_seguintes(tmp_path):
    orq = Orquestrador(cfg(tmp_path))
    orq.executar(DE, "2023-02-03")  # 24 pregões; checkpoints a cada 10
    antes = _tabelas_gravadas(orq)
    data = pd.Timestamp("2023-01-20")  # 14º pregão: o checkpoint mais recente é o do 11º
    orq.reprocessar(data)
    depois = _tabelas_gravadas(orq)
    for nome, col in TABELAS_POR_DIA.items():
        a, d = antes[nome], depois[nome]
        if a.empty:
            assert d.empty
            continue
        # anteriores intactas e seguintes reproduzidas: a tabela inteira é igual
        quase_igual(a.reset_index(drop=True), d.reset_index(drop=True), rtol=1e-12, atol=1e-9)
        assert (a[a[col] < data].reset_index(drop=True).equals(d[d[col] < data].reset_index(drop=True))), nome
    for nome in ("cota_diaria", "ledger", "episodios", "provisoes", "cotistas_mov", "backtest_var"):
        quase_igual(antes[nome].reset_index(drop=True), depois[nome].reset_index(drop=True), rtol=1e-12, atol=1e-9)
    # o log: dias anteriores intactos (inclusive durações); dias reprocessados com os MESMOS hashes de entrada
    la, ld = antes["execucoes_ciclo"], depois["execucoes_ciclo"]
    assert la[la["data"] < data].reset_index(drop=True).equals(ld[ld["data"] < data].reset_index(drop=True))
    chave = ["data", "etapa", "status", "hash_entrada"]
    assert la[la["data"] >= data][chave].reset_index(drop=True).equals(ld[ld["data"] >= data][chave].reset_index(drop=True))
    assert len(la) == len(ld) and ld["data"].nunique() == la["data"].nunique()


def test_reprocessar_sem_checkpoint_e_o_primeiro_pregao(tmp_path):
    orq = Orquestrador(cfg(tmp_path, ckpt_cada=0))
    orq.executar(DE, "2023-01-20")
    antes = _tabelas_gravadas(orq)
    orq.reprocessar("2023-01-12")  # sem checkpoints: reexecuta do início sem tocar nas saídas anteriores
    orq.reprocessar("2023-01-02")  # reprocessar o 1º pregão = recalcular tudo
    depois = _tabelas_gravadas(orq)
    for nome in ("cota_diaria", "pl_diario", "risco_diario", "quebras_conciliacao", "ledger"):
        quase_igual(antes[nome].reset_index(drop=True), depois[nome].reset_index(drop=True), rtol=1e-12, atol=1e-9)
    h = lambda t: t[t["status"] == "OK"][["data", "etapa", "hash_entrada"]].reset_index(drop=True)  # noqa: E731
    assert h(antes["execucoes_ciclo"]).equals(h(depois["execucoes_ciclo"]))


def test_reprocessar_exige_ciclo_gravado_e_mesma_configuracao(tmp_path):
    with pytest.raises(FileNotFoundError):
        Orquestrador(cfg(tmp_path)).reprocessar("2023-01-10")
    Orquestrador(cfg(tmp_path)).executar(DE, "2023-01-13")
    with pytest.raises(ValueError, match="configuração mudou"):
        Orquestrador(cfg(tmp_path, passo_ordens=3)).reprocessar("2023-01-10")
    with pytest.raises(ValueError, match="não é um pregão"):
        Orquestrador(cfg(tmp_path)).reprocessar("2023-01-14")  # sábado


def test_erro_em_etapa_e_registrado_e_as_seguintes_ficam_puladas(tmp_path, monkeypatch):
    sim = Simulacao(cfg(tmp_path, com_conciliacao=False, com_risco=False, com_pos_trade=False), DE, "2023-01-13")
    sim.executar_dia(sim.i0)

    def quebra(self, d):
        raise ValueError("falha simulada")

    monkeypatch.setattr(Simulacao, "_etapa_mtm", quebra)
    with pytest.raises(RuntimeError, match="etapa 'mtm'"):
        sim.executar_dia(sim.i0 + 1)
    dia = sim.datas[sim.i0 + 1]
    log = sim.log().query("data == @dia").set_index("etapa")
    assert log.loc["mtm", "status"] == "ERRO" and "falha simulada" in log.loc["mtm", "mensagem"]
    assert (log.loc[["provisoes_cota", "conciliacao", "pos_trade", "risco", "relatorios"], "status"] == "PULADA").all()
    assert (log.loc[["abertura", "mercado", "eventos", "passivo", "boletagem"], "status"] == "OK").all()


# ======================================================================= histerese e tolerância (ajuste A1)

def test_histerese_do_reenquadramento_exige_persistencia(prec, painel, tmp_path_factory):
    from gestorasim.enquadramento import montar_carteira
    from gestorasim.posicao import aplicar_execucao

    mesa, por, alvo = _mesa(prec, painel, tmp_path_factory)
    fia = por[ClasseFundo.FIA]
    led, P = mesa.ledger, mesa.precos
    q = led.quantidades(fia.cnpj)
    tk = max((t for t in q if prec.meta.loc[t, "classe"] == "ACAO"), key=lambda t: q[t] * P[t])
    led.integralizar(mesa.data, fia.cnpj, 20e6)
    aplicar_execucao(led, data=mesa.data, fundo_cnpj=fia.cnpj, ticker=tk, classe="ACAO", lado="C",
                     quantidade=int(12e6 // P[tk] // 100 * 100), preco=float(P[tk]), origem="teste")
    pers: dict = {}
    nenhuma = [o for o in ordens_reenquadramento(mesa, _Seq(), 1, None, pers, 2) if fia.cnpj in o.alvos]
    assert nenhuma == [] and max(pers.values()) == 1  # 1º pregão em violação: espera
    segunda = [o for o in ordens_reenquadramento(mesa, _Seq(), 2, None, pers, 2) if fia.cnpj in o.alvos]
    assert len(segunda) == 1 and segunda[0].ticker == tk  # persistiu 2 pregões: age
    # voltou ao limite: o contador zera (violação que some não acumula)
    led.integralizar(mesa.data, fia.cnpj, 0.0)
    aplicar_execucao(led, data=mesa.data, fundo_cnpj=fia.cnpj, ticker=tk, classe="ACAO", lado="V",
                     quantidade=int(12e6 // P[tk] // 100 * 100), preco=float(P[tk]), origem="teste")
    assert ordens_reenquadramento(mesa, _Seq(), 3, None, pers, 2) == [] and not any(k[0] == fia.cnpj for k in pers)
    # sem persistência informada o comportamento é imediato (padrão do ciclo: histerese_reenquadramento_du=1)
    assert ConfigCiclo().histerese_reenquadramento_du == 1 and ConfigCiclo().tolerancia_reinvestimento == 0.015


def test_tolerancia_de_reinvestimento_do_caixa_excedente(prec, painel, tmp_path_factory):
    from gestorasim.enquadramento import montar_carteira

    mesa, por, alvo = _mesa(prec, painel, tmp_path_factory)
    fia = por[ClasseFundo.FIA]
    led, a = mesa.ledger, alvo[fia.cnpj]
    add = 0.0225 * 100e6 / (1 - a - 0.0225)  # deixa o excesso entre 1,5% e 3% do PL acima do alvo de caixa
    led.creditar(mesa.data, fia.cnpj, "CAIXA", add, "teste")
    padrao = [o for o in ordens_gestao_caixa(mesa, alvo, _Seq()) if fia.cnpj in o.alvos]
    folgado = [o for o in ordens_gestao_caixa(mesa, alvo, _Seq(), tolerancia_excesso=0.03) if fia.cnpj in o.alvos]
    assert len(padrao) > 3 and all(o.origem == "REINVESTIMENTO" for o in padrao)
    assert folgado == []


def test_ordens_de_caixa_separam_gestao_e_reinvestimento(prec, painel, tmp_path_factory):
    mesa, por, alvo = _mesa(prec, painel, tmp_path_factory)
    fia = por[ClasseFundo.FIA]
    mesa.ledger.creditar(mesa.data, fia.cnpj, "CAIXA", -(mesa.ledger.caixa_projetado(fia.cnpj) + 1e6), "teste")
    venda = [o for o in ordens_gestao_caixa(mesa, alvo, _Seq()) if fia.cnpj in o.alvos]
    assert {o.origem for o in venda} == {"GESTAO_CAIXA"}  # recompor caixa = venda do ativo mais líquido


# ======================================================================= diagnóstico do Multimercado (ajuste A1)

def test_diagnostico_mm_fecha_com_a_cota(ciclo_curto, tmp_path):
    from gestorasim.dados import gerar_universo
    from gestorasim.mtm import painel_oficial
    from gestorasim.orquestrador.diagnostico import ORIGENS, decompor_retorno, dominancia_custos, imprimir

    orq, res, destino = ciclo_curto
    tabelas = gerar_universo(SEED, tmp_path)
    painel = painel_oficial(tabelas, SEED)
    mm = res["cota_diaria"].query("fundo == 'Sim Macro Multimercado'")["fundo_cnpj"].iloc[0]
    df = decompor_retorno({**res, "ativos": tabelas["ativos"]}, painel, mm)
    comp = df["componente"].tolist()
    for esperado in ("taxa de administração", "taxa de performance", "juros do caixa", "ajustes de DI",
                     "eventos (dividendos, cupons, amortizações)"):
        assert esperado in comp
    for o in ORIGENS:
        assert f"custos de execução — {o}" in comp
    soma = df.loc[df["componente"] == "retorno da cota líquida (soma)", "contribuicao_retorno"].iloc[0]
    cota = df.loc[df["componente"].str.contains("cota final"), "contribuicao_retorno"].iloc[0]
    assert soma == pytest.approx(cota, abs=1e-9)  # a decomposição fecha exatamente com a cota
    dom = dominancia_custos(df)
    assert 0 <= dom["peso"] <= 1
    texto = imprimir(df)
    assert "contribuição ao retorno" in texto and "referência do bloco 3" in texto
    # custos por origem somam o custo de execução da atribuição de P&L
    custos_pnl = res["pnl_atribuicao"].query("fundo_cnpj == @mm")["custos"].sum()
    assert df[df["componente"].str.startswith("custos de execução")]["valor_brl"].sum() == pytest.approx(custos_pnl, rel=1e-6, abs=1e-3)


def test_perfil_lista_as_10_funcoes_mais_caras():
    from gestorasim.orquestrador.perfil import perfilar

    tabela = perfilar("2023-01-13", top=10)
    assert len(tabela) == 10 and {"funcao", "chamadas", "tempo_acumulado_s", "tempo_proprio_s"} <= set(tabela.columns)
    assert tabela["tempo_acumulado_s"].is_monotonic_decreasing
