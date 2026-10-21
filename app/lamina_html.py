"""Lâmina de um fundo em HTML autocontido, no estilo de uma lâmina real, com formatação pt-BR (`app.formatacao`)."""

from __future__ import annotations

from html import escape

import pandas as pd

from app import formatacao as F
from gestorasim.relatorios.lamina import BENCHMARK_NOME, periodos

ROTULO_CLASSE = {"ACAO": "Ações", "TITULO_PUBLICO": "Títulos públicos", "DEBENTURE": "Debêntures", "FUTURO_DI": "Futuros de DI",
                 "CAIXA": "Caixa (projetado)"}
CLASSE_FUNDO = {"FIA": "Fundo de Ações", "MULTIMERCADO": "Fundo Multimercado", "RENDA_FIXA": "Renda Fixa Crédito Privado"}

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:0;color:#1b1f24;background:#f4f6f9}
.pg{max-width:900px;margin:1rem auto;background:#fff;box-shadow:0 1px 6px #0002}
.topo{background:#1f5fa8;color:#fff;padding:1.2rem 1.5rem}.topo h1{margin:0;font-size:1.5rem}
.topo p{margin:.25rem 0 0;font-size:.85rem;opacity:.9}
.kpi{display:flex;flex-wrap:wrap;gap:.6rem;padding:1rem 1.5rem;background:#eef3f9}
.kpi div{flex:1 1 160px;background:#fff;border-left:4px solid #1f5fa8;padding:.5rem .8rem}
.kpi small{display:block;color:#5b6670;font-size:.75rem}.kpi b{font-size:1.1rem}
.corpo{padding:0 1.5rem 1.2rem}h2{border-bottom:2px solid #1f5fa8;color:#1f5fa8;padding-bottom:.2rem;margin:1.4rem 0 .5rem;font-size:1.05rem}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:.35rem .55rem;border-bottom:1px solid #dde3ea;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#eef3f9;font-size:.78rem;color:#33404d}
.pos{color:#1b8a3b;font-weight:600}.neg{color:#c62828;font-weight:600}
.barra{background:#e6ebf2;height:10px;border-radius:5px;min-width:90px}.barra i{display:block;height:10px;border-radius:5px;background:#1f5fa8}
.nota{color:#5b6670;font-size:.78rem;margin-top:1rem}
.uso{background:#e6ebf2;height:12px;border-radius:6px;margin:.3rem 0}.uso i{display:block;height:12px;border-radius:6px}
"""


def _cor(v) -> str:
    return "" if v is None or pd.isna(v) else "pos" if v >= 0 else "neg"


def gerar_lamina_html(res: dict[str, pd.DataFrame], cnpj: str, data, classe_fundo: str = "") -> str:
    """`res`: cota_diaria, mtm_diario, ativos (+ risco_diario e enquadramento_diario, opcionais), todos já cortados em `data`."""
    data = pd.Timestamp(data)
    cota = res["cota_diaria"].query("fundo_cnpj == @cnpj")
    if cota.empty:
        raise KeyError(f"fundo sem dados: {cnpj}")
    nome = cota["fundo"].iloc[0]
    c = cota[cota["data"] <= data].sort_values("data").set_index("data")
    fim = c.index.max()
    ult = c.loc[fim]
    per = periodos(cota[cota["data"] <= data], f"{fim:%Y-%m}")
    bench = BENCHMARK_NOME.get(classe_fundo, "Benchmark")
    pl = float(ult["pl_liquido"])

    linhas = ""
    for p, (f, b) in per.items():
        dif = None if f is None or b is None else f - b
        linhas += (f"<tr><td>{escape(p.capitalize())}</td><td class='{_cor(f)}'>{F.pct(f)}</td><td>{F.pct(b)}</td>"
                   f"<td class='{_cor(dif)}'>{F.pct(dif)}</td></tr>")

    ativos = res["ativos"].set_index("ticker")
    mtm = res["mtm_diario"].query("fundo_cnpj == @cnpj and data == @fim").copy()
    mtm["classe"] = mtm["ticker"].map(ativos["classe"])
    comp = mtm.groupby("classe").agg(vm=("valor_mercado", "sum"), exp=("exposicao", "sum")).reset_index()
    caixa = float(ult["pl_ativos"]) - float(mtm["valor_mercado"].sum())
    comp = pd.concat([comp, pd.DataFrame([{"classe": "CAIXA", "vm": caixa, "exp": caixa}])], ignore_index=True)
    comp["peso"] = comp["vm"] / pl
    linhas_comp = "".join(
        f"<tr><td>{escape(ROTULO_CLASSE.get(r.classe, r.classe))}</td><td>{F.mi(r.vm)}</td><td>{F.pct(r.peso)}</td>"
        f"<td><div class='barra'><i style='width:{max(0, min(r.peso, 1)) * 100:.1f}%'></i></div></td><td>{F.pct(r.exp / pl)}</td></tr>"
        for r in comp.itertuples() if abs(r.vm) > 0 or abs(r.exp) > 0)

    maiores = mtm.assign(peso=mtm["exposicao"] / pl).sort_values("exposicao", ascending=False).head(10)
    linhas_pos = "".join(
        f"<tr><td>{escape(r.ticker)}</td><td>{escape(ROTULO_CLASSE.get(r.classe, str(r.classe)))}</td><td>{F.qtd(r.qtd)}</td>"
        f"<td>{F.num(r.preco)}</td><td>{F.mi(r.valor_mercado)}</td><td>{F.pct(r.peso)}</td></tr>" for r in maiores.itertuples())

    bloco_risco = "<p>Risco não calculado nesta data.</p>"
    risco = res.get("risco_diario")
    if risco is not None and len(risco):
        r = risco.query("fundo_cnpj == @cnpj and data == @fim")
        if len(r):
            r = r.iloc[0]
            lim = None
            enq = res.get("enquadramento_diario")
            if enq is not None and len(enq):
                e = enq[(enq["fundo_cnpj"] == cnpj) & (enq["regra"] == "politica_var") & (enq["data"] <= fim)]
                lim = None if e.empty else float(e.sort_values("data")["limite"].iloc[-1])
            uso = ""
            if lim and pd.notna(r["var_param_pct"]):
                u = float(r["var_param_pct"]) / lim
                cor = F.VERMELHO if u >= 1 else F.AMBAR if u >= 0.85 else F.VERDE
                uso = (f"<p>Uso do limite de VaR da política ({F.pct(lim)}): <b>{F.pct(u, 1)}</b></p>"
                       f"<div class='uso'><i style='width:{min(u, 1) * 100:.1f}%;background:{cor}'></i></div>")
            bloco_risco = (
                "<table><tr><th>VaR 99% 1d paramétrico (% PL)</th><th>VaR 99% 1d histórico (% PL)</th><th>ES 97,5% (% PL)</th>"
                "<th>Pior stress (% PL)</th></tr>"
                f"<tr><td>{F.pct(r['var_param_pct'])}</td><td>{F.pct(r['var_hist_pct'])}</td><td>{F.pct(r['es_hist_pct'])}</td>"
                f"<td>{F.pct(r['stress_pior_pct'])}</td></tr></table>{uso}")

    return f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><title>Lâmina — {escape(nome)} — {fim:%m/%Y}</title>
<style>{CSS}</style></head><body><div class="pg">
<div class="topo"><h1>{escape(nome)}</h1><p>CNPJ {escape(cnpj)} (fictício) · {escape(CLASSE_FUNDO.get(classe_fundo, classe_fundo or '-'))}
 · Benchmark: {escape(bench)} · Posição em {F.data(fim)}</p></div>
<div class="kpi"><div><small>PL líquido</small><b>{F.brl_mi(pl)}</b></div>
<div><small>Cota líquida</small><b>{F.cota(ult['cota_liquida'])}</b></div>
<div><small>Cotas em circulação</small><b>{F.qtd(ult['cotas'])}</b></div>
<div><small>Rentabilidade no mês</small><b class="{_cor(per['mês'][0])}">{F.pct(per['mês'][0])}</b></div></div>
<div class="corpo">
<h2>Rentabilidade (cota líquida de taxas)</h2>
<table><tr><th>Período</th><th>Fundo (%)</th><th>{escape(bench)} (%)</th><th>Diferença (p.p.)</th></tr>{linhas}</table>
<h2>Composição da carteira</h2>
<table><tr><th>Classe</th><th>Valor de mercado (R$ mi)</th><th>% do PL</th><th></th><th>Exposição / PL (%)</th></tr>{linhas_comp}</table>
<h2>Risco</h2>{bloco_risco}
<h2>10 maiores posições</h2>
<table><tr><th>Ativo</th><th>Classe</th><th>Quantidade</th><th>Preço / PU (R$)</th><th>Valor (R$ mi)</th><th>Exposição / PL (%)</th></tr>{linhas_pos}</table>
<p class="nota">Material educacional; dados 100% simulados. Rentabilidade passada não garante rentabilidade futura. "n/d" = histórico insuficiente.</p>
</div></div></body></html>"""
