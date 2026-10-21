import pandas as pd
import pytest

from gestorasim.boletagem import (
    TRANSICOES, EstadoOrdem, Mesa, OrdemAgregada, TransicaoInvalida, Trilha, ratear, rodar_dia_exemplo,
)
from gestorasim.boletagem.dia_exemplo import precos_do_dia
from gestorasim.boletagem.execucao import gerar_volume, simular_fills
from gestorasim.dados import carregar_fundos, gerar_universo
from gestorasim.dominio import ClasseAtivo, ClasseFundo
from gestorasim.enquadramento import avaliar_pre_trade, montar_carteira, todas_as_regras
from gestorasim.enquadramento.regras import VIOLADO
from gestorasim.posicao import Ledger, aplicar_execucao, gerar_carteira_inicial

SEED = 42


@pytest.fixture(scope="module")
def saidas(tmp_path_factory):
    return rodar_dia_exemplo(seed=SEED, destino=tmp_path_factory.mktemp("dia"))


@pytest.fixture()
def amb(tmp_path_factory):
    """Ambiente fresco: ledger com carteira inicial e Mesa no dia D1."""
    destino = tmp_path_factory.mktemp("amb")
    t = gerar_universo(SEED, destino)
    fundos = carregar_fundos(destino)
    meta = t["ativos"].set_index("ticker")
    datas = pd.DatetimeIndex(t["calendario"]["data"])
    led = Ledger()
    gerar_carteira_inicial(led, fundos, meta, precos_do_dia(t, datas[0]), datas[0], SEED)
    precos = precos_do_dia(t, datas[1])
    vol = gerar_volume(meta, datas, SEED).query("data == @datas[1]").set_index("ticker")["volume"]
    mesa = Mesa(led, fundos, meta, datas[1], precos, vol)
    por = {f.classe: f for f in fundos}
    return mesa, por, meta, precos


def ordem(id_, tk, classe, lado, alvos, **kw):
    return OrdemAgregada(id=id_, data="2023-01-03", ticker=tk, classe=classe, lado=lado, alvos=alvos, **kw)


# ----------------------------------------------------------------------- posição / carteira inicial

def test_carteira_inicial_respeita_regulamento(amb):
    mesa, por, meta, precos = amb
    for f in por.values():
        cart = montar_carteira(mesa.ledger.quantidades(f.cnpj), mesa.ledger.caixa(f.cnpj), meta, precos)
        assert not [r for r in todas_as_regras(cart, f.regulamento) if r.status == VIOLADO]
        assert mesa.ledger.caixa(f.cnpj) > 0


def test_ledger_append_only_e_saldos():
    led = Ledger()
    led.integralizar("2023-01-02", "F", 1000.0)
    aplicar_execucao(led, data="2023-01-02", fundo_cnpj="F", ticker="X", classe="ACAO", lado="C",
                     quantidade=10, preco=20.0, origem="t")
    aplicar_execucao(led, data="2023-01-02", fundo_cnpj="F", ticker="DI", classe="FUTURO_DI", lado="V",
                     quantidade=2, preco=1000.0, origem="t")
    assert led.quantidades("F") == {"X": 10, "DI": -2}
    assert led.caixa("F") == pytest.approx(800.0)  # futuro não move caixa
    n = len(led)
    df = led.para_df()
    df.loc[:, "delta"] = 0  # alterar a cópia não altera o ledger
    assert len(led) == n and led.caixa("F") == pytest.approx(800.0)
    assert df["seq"].tolist() == list(range(1, n + 1))
    assert not hasattr(led, "remover") and not hasattr(led, "editar")


# ----------------------------------------------------------------------- rateio

def test_rateio_proporcional_lote_e_residuo():
    aloc, criterio = ratear({"A": 1000, "B": 3000, "C": 2000}, 3000, 100)
    assert sum(aloc.values()) == 3000
    assert aloc == {"A": 500, "B": 1500 + 0, "C": 1000}  # exato: 3000*1000/6000 = 500 etc.
    assert "maior alvo (B)" in criterio and "lote=100" in criterio
    aloc, _ = ratear({"A": 1000, "B": 3000, "C": 1300}, 5000, 100)
    assert sum(aloc.values()) == 5000
    assert all(q % 100 == 0 for q in aloc.values())
    assert aloc["B"] == max(aloc.values())  # resíduo vai ao maior alvo


def test_rateio_empate_vai_ao_primeiro_e_rejeita_excesso():
    aloc, _ = ratear({"A": 100, "B": 100, "C": 100}, 200, 100)
    assert aloc == {"A": 200, "B": 0, "C": 0}  # resíduo inteiro ao primeiro empatado
    with pytest.raises(ValueError):
        ratear({"A": 100}, 200, 100)


def test_soma_alocacoes_igual_executado_e_preco_unico(saidas):
    o, al, ex = saidas["ordens"], saidas["alocacoes"], saidas["execucoes"]
    alocadas = o[o["estado"] == "ALOCADA"]
    assert len(alocadas) > 20
    soma = al.groupby("ordem_id")["quantidade_alocada"].sum()
    fills = ex.groupby("ordem_id")["quantidade"].sum()
    for _, r in alocadas.iterrows():
        assert soma[r.ordem_id] == r.quantidade_executada == fills[r.ordem_id]
    # preço médio idêntico entre fundos, e igual ao VWAP dos fills
    assert (al.groupby("ordem_id")["preco_medio"].nunique() == 1).all()
    vwap = (ex.assign(f=ex.quantidade * ex.preco).groupby("ordem_id")["f"].sum() / fills)
    pm = o.set_index("ordem_id")["preco_medio"].dropna()
    assert (pm - vwap[pm.index]).abs().max() < 1e-9
    # lote padrão: ações em múltiplos de 100 (exceto o resíduo, que também é múltiplo)
    acoes = al.merge(o[["ordem_id", "classe"]], on="ordem_id").query("classe == 'ACAO'")
    assert (acoes["quantidade_alocada"] % 100 == 0).all()


def test_alocacao_multifundo_proporcional(saidas):
    o, al = saidas["ordens"], saidas["alocacoes"]
    multi = o[(o["n_fundos"] > 1) & (o["estado"] == "ALOCADA")]["ordem_id"]
    assert len(multi) > 0
    for oid in multi:
        a = al[al.ordem_id == oid]
        razao = a["quantidade_alocada"] / a["quantidade_alvo"]
        assert razao.max() - razao.min() < 0.05 or a["quantidade_alocada"].sum() == a["quantidade_alvo"].sum()


# ----------------------------------------------------------------------- pré-trade e bloqueio

def test_ordem_violadora_fica_bloqueada(amb):
    mesa, por, meta, precos = amb
    fia = por[ClasseFundo.FIA]
    o = mesa.processar(ordem("T1", "DLOG14", ClasseAtivo.DEBENTURE, "C", {fia.cnpj: 5000}))
    assert o.estado == EstadoOrdem.BLOQUEADA
    assert o.quantidade_executada == 0
    assert "limite_classe[DEBENTURE]" in mesa.tabelas()["auditoria"].iloc[-1]["motivo"]


def test_violadoras_do_exemplo_bloqueadas_exceto_override(saidas):
    o, au = saidas["ordens"], saidas["auditoria"]
    viol = o[o["origem"].str.startswith("VIOLADORA")]
    assert len(viol) == 5
    for _, r in viol.iterrows():
        if r.origem.endswith("rating"):
            assert r.estado == "ALOCADA"
            assert "LIBERADA_OVERRIDE" in set(au[au.ordem_id == r.ordem_id]["para"])
        else:
            assert r.estado == "BLOQUEADA"


def test_caixa_insuficiente_e_venda_descoberto_bloqueiam(amb):
    mesa, por, meta, precos = amb
    rf = por[ClasseFundo.RENDA_FIXA]
    q = int(2 * 100e6 // precos["LTN270104"])
    assert mesa.processar(ordem("T2", "LTN270104", ClasseAtivo.TITULO_PUBLICO, "C", {rf.cnpj: q})).estado \
        == EstadoOrdem.BLOQUEADA
    fia = por[ClasseFundo.FIA]
    assert mesa.processar(ordem("T3", "ALFA3", ClasseAtivo.ACAO, "V", {fia.cnpj: 10**7})).estado \
        == EstadoOrdem.BLOQUEADA


def test_minimo_classe_e_liquidez_no_pre_trade(amb):
    mesa, por, meta, precos = amb
    fia = por[ClasseFundo.FIA]
    q = mesa.ledger.quantidades(fia.cnpj)
    tk = max((t for t in q if meta.loc[t, "classe"] == "ACAO"), key=lambda t: q[t] * precos[t])
    # vender mais de 18% do PL em uma ação derruba o mínimo de 67% em ações
    pl = montar_carteira(q, mesa.ledger.caixa(fia.cnpj), meta, precos).pl
    res = avaliar_pre_trade(q, mesa.ledger.caixa(fia.cnpj), fia.regulamento, meta, precos, ticker=tk,
                            lado="V", quantidade=q[tk])
    assert {r.regra for r in res if r.status == VIOLADO} == set()  # 8,5% do PL: ainda enquadrado
    q2, caixa2 = dict(q), mesa.ledger.caixa(fia.cnpj)
    for t in sorted(q, key=lambda t: -q[t] * precos[t])[:3]:
        caixa2 += q2[t] * precos[t]  # vendidas ao fechamento: o PL se mantém
        q2[t] = 0
    quarto = [t for t in sorted(q, key=lambda t: -q[t] * precos[t]) if meta.loc[t, 'classe'] == 'ACAO'][3]
    res = avaliar_pre_trade(q2, caixa2, fia.regulamento, meta, precos, ticker=quarto,
                            lado="V", quantidade=100)  # já desenquadrado (59,5%) e a venda agrava
    assert any(r.regra == "minimo_classe" and r.status == VIOLADO for r in res)
    res = avaliar_pre_trade(q2, caixa2, fia.regulamento, meta, precos, ticker=quarto,
                            lado="C", quantidade=100)  # a compra melhora: não bloqueia
    assert any(r.regra == "minimo_classe" and r.status == "DESENQUADRADO_PREEXISTENTE" for r in res)
    assert pl > 0


def test_desenquadramento_preexistente_nao_bloqueia_ordem_que_nao_piora(amb):
    mesa, por, meta, precos = amb
    rf = por[ClasseFundo.RENDA_FIXA]
    # força A- acima do limite (10%) direto no ledger e verifica que uma ordem alheia passa
    aplicar_execucao(mesa.ledger, data=mesa.data, fundo_cnpj=rf.cnpj, ticker="EVAR15", classe="DEBENTURE",
                     lado="C", quantidade=8_000, preco=float(precos["EVAR15"]), origem="teste")
    o = mesa.processar(ordem("T4", "LTN270104", ClasseAtivo.TITULO_PUBLICO, "C", {rf.cnpj: 100}))
    assert o.estado == EstadoOrdem.ALOCADA
    pre = mesa.tabelas()["pretrade"]
    assert "DESENQUADRADO_PREEXISTENTE" in set(pre["status"])


# ----------------------------------------------------------------------- override e trilha

def test_override_exige_justificativa_e_usuario(amb):
    mesa, por, *_ = amb
    fia = por[ClasseFundo.FIA]
    o = ordem("T5", "DLOG14", ClasseAtivo.DEBENTURE, "C", {fia.cnpj: 5000})
    mesa.criar(o)
    assert not mesa.pre_trade(o)
    for usuario, just in [("diretor", ""), ("diretor", "   "), ("diretor", "ok"), ("", "justificativa suficiente")]:
        with pytest.raises(ValueError):
            mesa.liberar_override(o, usuario, just)
    assert o.estado == EstadoOrdem.BLOQUEADA
    mesa.liberar_override(o, "diretor.risco", "Excecao aprovada pelo comite")
    assert o.estado == EstadoOrdem.LIBERADA_OVERRIDE
    ev = mesa.tabelas()["auditoria"].iloc[-1]
    assert ev["usuario"] == "diretor.risco" and ev["motivo"] == "Excecao aprovada pelo comite"


def test_transicoes_invalidas_rejeitadas():
    tr = Trilha(pd.Timestamp("2023-01-03 09:00").to_pydatetime())
    o = ordem("T6", "ALFA3", ClasseAtivo.ACAO, "C", {"F": 100})
    with pytest.raises(TransicaoInvalida):
        tr.transicionar(o, EstadoOrdem.ENVIADA, "u", "pula etapas")
    tr.transicionar(o, EstadoOrdem.CRIADA, "u", "ok")
    for destino in (EstadoOrdem.ENVIADA, EstadoOrdem.LIBERADA_OVERRIDE, EstadoOrdem.ALOCADA):
        with pytest.raises(TransicaoInvalida):
            tr.transicionar(o, destino, "u", "inválida")
    assert len(tr) == 1 and o.estado == EstadoOrdem.CRIADA


def test_trilha_do_exemplo_so_tem_transicoes_validas(saidas):
    au = saidas["auditoria"]
    assert au["seq"].tolist() == list(range(1, len(au) + 1))
    assert au["timestamp"].is_monotonic_increasing
    assert (au["usuario"].str.strip() != "").all() and (au["motivo"].str.strip() != "").all()
    for oid, g in au.groupby("ordem_id"):
        estados = [None] + g["para"].tolist()
        assert g["de"].tolist() == [None if e is None else e for e in estados[:-1]] or \
            g["de"].fillna("-").tolist() == ["-"] + estados[1:-1]
        for de, para in zip(estados[:-1], estados[1:]):
            de = EstadoOrdem(de) if de else None
            assert EstadoOrdem(para) in TRANSICOES[de]
    o = saidas["ordens"].set_index("ordem_id")["estado"]
    ultimo = au.groupby("ordem_id")["para"].last()
    assert (ultimo == o[ultimo.index]).all()


def test_criterio_de_rateio_na_trilha(saidas):
    au = saidas["auditoria"]
    aloc = au[au["para"] == "ALOCADA"]
    assert len(aloc) > 0
    assert aloc["motivo"].str.contains("proporcional ao alvo").all()
    assert aloc["motivo"].str.contains("lote=").all() and aloc["motivo"].str.contains("maior alvo").all()


# ----------------------------------------------------------------------- execução

def test_fills_parciais_acima_de_20pct_do_volume():
    fills = simular_fills(10_000, "C", 100.0, 20_000, "DEBENTURE")
    assert sum(q for q, _ in fills) == 4_000  # 20% de 20.000
    fills = simular_fills(1_000, "C", 100.0, 20_000, "DEBENTURE")
    assert sum(q for q, _ in fills) == 1_000
    fills = simular_fills(1_050, "C", 10.0, 1_000_000, "ACAO")
    assert all(q % 100 == 0 for q, _ in fills)


def test_slippage_proporcional_e_com_sinal():
    compra = simular_fills(2_000, "C", 100.0, 20_000, "DEBENTURE")
    venda = simular_fills(2_000, "V", 100.0, 20_000, "DEBENTURE")
    assert all(p > 100 for _, p in compra) and all(p < 100 for _, p in venda)
    pequena = simular_fills(200, "C", 100.0, 20_000, "DEBENTURE")
    assert sum(q * p for q, p in compra) / 2000 > sum(q * p for q, p in pequena) / 200


def test_exemplo_tem_execucao_parcial_e_participacao_limitada(saidas):
    ex, o = saidas["execucoes"], saidas["ordens"]
    part = ex.groupby("ordem_id")["quantidade"].sum() / ex.groupby("ordem_id")["volume_dia"].first()
    assert part.max() <= 0.2 + 1e-9
    assert (o["estado"] == "ALOCADA").sum() + (o["estado"] == "BLOQUEADA").sum() == len(o)
    assert ((o["quantidade_executada"] > 0) & (o["quantidade_executada"] < o["quantidade_alvo"])).any()


# ----------------------------------------------------------------------- determinismo e saídas

def test_determinismo_por_seed(saidas, tmp_path):
    outra = rodar_dia_exemplo(seed=SEED, destino=tmp_path)
    for nome in ("ordens", "execucoes", "alocacoes", "auditoria", "pretrade", "ledger"):
        pd.testing.assert_frame_equal(saidas[nome], outra[nome])
    diferente = rodar_dia_exemplo(seed=7, destino=tmp_path / "b")
    assert not diferente["ordens"].equals(saidas["ordens"])


def test_parquets_gravados(saidas, tmp_path_factory):
    d = tmp_path_factory.mktemp("g")
    rodar_dia_exemplo(seed=SEED, destino=d)
    for nome in ("ordens", "execucoes", "alocacoes", "auditoria"):
        assert (d / f"{nome}.parquet").exists()
