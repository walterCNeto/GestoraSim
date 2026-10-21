"""Lâmina mensal de um fundo (HTML autocontido) a partir das tabelas gravadas pelo ciclo."""

from __future__ import annotations

from html import escape

import pandas as pd

from gestorasim.util import fmt3

BENCHMARK_NOME = {"MULTIMERCADO": "CDI", "RENDA_FIXA": "CDI", "FIA": "Índice sintético de ações"}


def _ref(serie: pd.Series, limite: pd.Timestamp):
    """Último valor em data <= limite; None se não houver."""
    s = serie[serie.index <= limite]
    return None if s.empty else float(s.iloc[-1])


def _retorno(serie: pd.Series, ini: pd.Timestamp | None, fim: pd.Timestamp) -> float | None:
    """Retorno entre o último valor <= ini (ou o primeiro disponível se a série começa depois) e o último <= fim."""
    final = _ref(serie, fim)
    base = None if ini is None else _ref(serie, ini)
    if base is None:
        base = float(serie.iloc[0])
    if final is None or base is None:
        return None
    return final / base - 1


def periodos(cota: pd.DataFrame, mes: str) -> dict[str, tuple[float | None, float | None]]:
    """Rentabilidade da cota líquida e do benchmark no mês, no ano e nos últimos 12 meses (fecha no último pregão do mês)."""
    c = cota.sort_values("data").set_index("data")
    fim = c.index[c.index <= pd.Period(mes).end_time.normalize()].max()
    inicio_mes = pd.Period(mes).start_time.normalize()
    inicio_ano = pd.Timestamp(year=fim.year, month=1, day=1)
    doze = fim - pd.DateOffset(months=12)
    out = {}
    for nome, ini in (("mês", inicio_mes - pd.Timedelta(days=1)), ("ano", inicio_ano - pd.Timedelta(days=1)), ("12 meses", doze)):
        if nome == "12 meses" and c.index.min() > doze:
            out[nome] = (None, None)  # histórico insuficiente
        else:
            out[nome] = (_retorno(c["cota_liquida"], ini, fim), _retorno(c["benchmark"], ini, fim))
    return out


def gerar_lamina(res: dict[str, pd.DataFrame], cnpj: str, mes: str, classe_fundo: str = "") -> str:
    """`res`: tabelas cota_diaria, mtm_diario, risco_diario, ativos (e opcionalmente fundos). `mes`: 'AAAA-MM'."""
    cota = res["cota_diaria"].query("fundo_cnpj == @cnpj")
    if cota.empty:
        raise KeyError(f"fundo sem dados: {cnpj}")
    nome = cota["fundo"].iloc[0]
    c = cota.sort_values("data").set_index("data")
    fim = c.index[c.index <= pd.Period(mes).end_time.normalize()].max()
    ult = c.loc[fim]
    per = periodos(cota, mes)
    fmt = lambda x: "n/d" if x is None else fmt3(x)  # noqa: E731
    bench = BENCHMARK_NOME.get(classe_fundo, "Benchmark")

    linhas_rent = "".join(
        f"<tr><td>{p}</td><td>{fmt(f)}</td><td>{fmt(b)}</td><td>{fmt(None if f is None or b is None else f - b)}</td></tr>"
        for p, (f, b) in per.items())

    ativos = res["ativos"].set_index("ticker")
    mtm = res["mtm_diario"].query("fundo_cnpj == @cnpj and data == @fim").copy()
    mtm["classe"] = mtm["ticker"].map(ativos["classe"])
    pl_liq = float(ult["pl_liquido"])
    comp = mtm.groupby("classe").agg(valor_mercado=("valor_mercado", "sum"), exposicao=("exposicao", "sum"))
    caixa = float(ult["pl_ativos"]) - float(mtm["valor_mercado"].sum())
    linhas_comp = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{fmt3(r.valor_mercado)}</td><td>{fmt3(r.valor_mercado / pl_liq)}</td>"
        f"<td>{fmt3(r.exposicao / pl_liq)}</td></tr>" for k, r in comp.iterrows())
    linhas_comp += f"<tr><td>CAIXA (projetado)</td><td>{fmt3(caixa)}</td><td>{fmt3(caixa / pl_liq)}</td><td></td></tr>"

    maiores = mtm.assign(peso=mtm["exposicao"] / pl_liq).sort_values("exposicao", ascending=False).head(10)
    linhas_pos = "".join(
        f"<tr><td>{escape(r.ticker)}</td><td>{escape(str(r.classe))}</td><td>{int(r.qtd)}</td><td>{fmt3(r.preco)}</td>"
        f"<td>{fmt3(r.valor_mercado)}</td><td>{fmt3(r.peso)}</td></tr>" for r in maiores.itertuples())

    risco = res.get("risco_diario")
    bloco_risco = "<p>Risco não calculado nesta data.</p>"
    if risco is not None and len(risco):
        r = risco.query("fundo_cnpj == @cnpj and data == @fim")
        if len(r):
            r = r.iloc[0]
            bloco_risco = (
                "<table><tr><th>VaR 99% 1d (paramétrico)</th><th>VaR 99% 1d (histórico)</th><th>ES 97,5% (histórico)</th>"
                "<th>Pior stress</th></tr>"
                f"<tr><td>{fmt(None if pd.isna(r['var_param_pct']) else r['var_param_pct'])}</td>"
                f"<td>{fmt(None if pd.isna(r['var_hist_pct']) else r['var_hist_pct'])}</td>"
                f"<td>{fmt(None if pd.isna(r['es_hist_pct']) else r['es_hist_pct'])}</td>"
                f"<td>{fmt3(r['stress_pior_pct'])}</td></tr></table><p class='nota'>Em fração do PL líquido.</p>")

    return f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><title>Lâmina — {escape(nome)} — {mes}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:2rem auto;max-width:860px;color:#1b1f24;padding:0 1rem}}
h1{{margin-bottom:.2rem}} h2{{border-bottom:2px solid #1f5fa8;padding-bottom:.2rem;margin-top:1.8rem;color:#1f5fa8}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0}} th,td{{border:1px solid #cfd6dd;padding:.35rem .6rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:#eef3f9}} .nota{{color:#5b6670;font-size:.85rem}}
.kpi{{display:flex;gap:1.5rem;flex-wrap:wrap}} .kpi div{{background:#eef3f9;padding:.6rem 1rem;border-radius:6px}}
</style></head><body>
<h1>{escape(nome)}</h1>
<p class="nota">CNPJ {escape(cnpj)} (fictício) · classe {escape(classe_fundo or "-")} · posição em {fim:%d/%m/%Y} · dados 100% simulados</p>
<div class="kpi"><div><b>PL líquido (R$)</b><br>{fmt3(pl_liq)}</div><div><b>Cota líquida</b><br>{fmt3(float(ult['cota_liquida']))}</div>
<div><b>Quantidade de cotas</b><br>{fmt3(float(ult['cotas']))}</div></div>
<h2>Rentabilidade (cota líquida de taxas)</h2>
<table><tr><th>Período</th><th>Fundo</th><th>{escape(bench)}</th><th>Diferença</th></tr>{linhas_rent}</table>
<h2>Composição por classe</h2>
<table><tr><th>Classe</th><th>Valor de mercado (R$)</th><th>% do PL líquido</th><th>Exposição / PL</th></tr>{linhas_comp}</table>
<h2>Maiores posições</h2>
<table><tr><th>Ativo</th><th>Classe</th><th>Quantidade</th><th>Preço</th><th>Valor de mercado (R$)</th><th>Exposição / PL</th></tr>{linhas_pos}</table>
<h2>Risco</h2>{bloco_risco}
<p class="nota">Números com três casas decimais. Rentabilidades em fração (0,010 = 1,0%). Material educacional.</p>
</body></html>"""
