"""Série diária: ledger rodando sobre os pregões, MTM, eventos, ordens periódicas e atribuição de P&L."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.boletagem.fluxo import Mesa
from gestorasim.cota.benchmark import indice_acoes, indice_cdi
from gestorasim.cota.engine import PassivoFundo
from gestorasim.cota.parametros import ParametrosPassivo
from gestorasim.boletagem.gestao_caixa import ordens_gestao_caixa, ordens_reenquadramento
from gestorasim.boletagem.geradores import _Seq, gerar_ordens_exemplo
from gestorasim.dados import calendario, carregar_fundos, gerar_universo
from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.enquadramento import montar_carteira
from gestorasim.mtm.cache import impressao_digital, painel_oficial
from gestorasim.mtm.estatistica import ewma_vol
from gestorasim.mtm.precificacao import Precificador, ajuste_di
from gestorasim.posicao import PL_INICIAL, Ledger, ProcessadorEventos, gerar_carteira_inicial
from gestorasim.posicao.caixa import SPREAD_CAIXA_NEGATIVO, juros_caixa
from gestorasim.util import fmt3

TOLERANCIA_FECHAMENTO = 1e-6
COMPONENTES = ["efeito_preco", "carrego", "eventos", "ajuste_futuros", "custos", "juros_caixa"]
TABELAS_SAIDA = ("posicoes_diarias", "mtm_diario", "pnl_atribuicao", "pl_diario", "ledger",
                 "ordens_serie", "alocacoes_serie", "auditoria_serie",
                 "cotistas_mov", "lotes_cotistas", "provisoes", "cota_diaria", "tributos")


def rodar_serie(seed: int = 42, destino: Path | str | None = None, inicio=None, fim=None,
                passo_ordens: int = 5, com_ordens: bool = True, escrever: bool = True,
                spread_caixa_negativo: float = SPREAD_CAIXA_NEGATIVO, com_passivo: bool = True,
                params_passivo: ParametrosPassivo | None = None,
                com_politica_reenquadramento: bool = True) -> dict[str, pd.DataFrame]:
    """Roda todos os fundos de `inicio` a `fim` (default: o calendário inteiro).

    O caixa projetado do fechamento anterior rende CDI (saldo >= 0) ou paga CDI + `spread_caixa_negativo`.
    Ordens do gerador do bloco 2 (rebalanceamento + aleatórias + violadoras sem override) a cada
    `passo_ordens` pregões. Preço de execução = MTM do dia ± slippage; cotistas sem movimentação.
    """
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    tabelas = gerar_universo(seed=seed, destino=destino)
    fundos = carregar_fundos(destino)
    nomes = {f.cnpj: f.nome for f in fundos}
    prec = Precificador(tabelas, seed)
    meta, datas, estados = prec.meta, prec.datas, prec.estados
    painel = painel_oficial(tabelas, seed, destino)  # preço oficial (mtm), em cache
    fontes = {tk: prec.fonte(tk) for tk in meta.index}
    volume = gerar_volume(meta, datas, seed).pivot(index="data", columns="ticker", values="volume")
    i0 = 0 if inicio is None else int(datas.searchsorted(pd.Timestamp(inicio)))
    i1 = len(datas) - 1 if fim is None else int(datas.searchsorted(pd.Timestamp(fim), side="right")) - 1

    ledger = Ledger()
    gerar_carteira_inicial(ledger, fundos, meta, painel.iloc[i0].dropna(), datas[i0], seed)
    proc = ProcessadorEventos(ledger, tabelas["eventos"], prec.fracoes)
    caixa_alvo = {f.cnpj: ledger.caixa_projetado(f.cnpj) / PL_INICIAL for f in fundos}  # alocação estratégica
    reducao = None
    if com_ordens and com_politica_reenquadramento:  # importações tardias: risco depende de mtm
        from gestorasim.risco.fatores import construir_fatores
        from gestorasim.risco.reenquadramento import ReducaoPolitica
        from gestorasim.risco.sensibilidades import CalculadoraSens
        fat = construir_fatores(tabelas, prec, painel)
        reducao = ReducaoPolitica(CalculadoraSens(prec, fat, painel), fat)
    cnpjs = [f.cnpj for f in fundos]
    params = params_passivo or ParametrosPassivo()
    passivo: dict[str, PassivoFundo] = {}
    bench: dict[str, object] = {}
    if com_passivo:
        cdi_idx, acoes_idx = indice_cdi(tabelas), indice_acoes(tabelas, painel, seed)
        for f in fundos:
            regras = params.regras[f.classe]
            passivo[f.cnpj] = PassivoFundo(f, regras, params,
                                           tabelas["fluxo_cotistas"].query("fundo_cnpj == @f.cnpj"))
            bench[f.cnpj] = acoes_idx if regras.benchmark == "ACOES" else cdi_idx
    di_tickers = list(meta.index[meta["classe"] == "FUTURO_DI"])
    sigma_di = pd.DataFrame({tk: ewma_vol(painel[tk].pct_change().to_numpy()) for tk in di_tickers}, index=datas)
    tab_ordens: list[pd.DataFrame] = []

    pos_rows, mtm_rows, pl_rows, pnl_rows = [], [], [], []
    ordens_estado: Counter = Counter()
    q_prev: dict[str, dict[str, int]] = {}
    cx_prev: dict[str, float] = {}
    pl_prev: dict[str, float] = {}

    for i in range(i0, i1 + 1):
        data, P = datas[i], painel.iloc[i]
        Pv = P.dropna()
        pnl = defaultdict(lambda: dict.fromkeys(COMPONENTES, 0.0))
        if i > i0:
            n0 = len(ledger)
            for c in cnpjs:  # juros do caixa sobre o saldo projetado do fechamento anterior
                j = juros_caixa(cx_prev[c], estados[i - 1].cdi_aa, spread_caixa_negativo)
                ledger.creditar(data, c, "CAIXA", j, "JUROS_CAIXA")
            proc.processar(data, cnpjs)
            Pant = painel.iloc[i - 1]
            for c in cnpjs:  # ajuste diário dos futuros de DI, liquidado em D+1
                for tk, q in q_prev[c].items():
                    if meta.loc[tk, "classe"] == "FUTURO_DI":
                        aj = ajuste_di(P[tk], Pant[tk], estados[i - 1].cdi_aa, q)
                        ledger.creditar(data, c, tk, aj, "AJUSTE_DI_DIA", calendario.somar_du(data, 1))
            if com_ordens:
                agenda = {c: passivo[c].saidas_previstas(data, [calendario.somar_du(data, k) for k in range(3)])
                          for c in passivo}
                mesa = Mesa(ledger, fundos, meta, data, Pv, volume.iloc[i].dropna(), sigma_di.iloc[i].to_dict(),
                            agenda)
                ids = _Seq()
                # 1) caixa e reenquadramento (vendas antes); 2) ordens regulares a cada `passo_ordens` pregões
                previas = (ordens_gestao_caixa(mesa, caixa_alvo, ids)
                           + ordens_reenquadramento(mesa, ids, i, reducao))
                for o in sorted(previas, key=lambda o: (o.lado.value != "V", o.id)):
                    ordens_estado[mesa.processar(o).estado.value] += 1
                if (i - i0) % passo_ordens == 0:
                    for o in gerar_ordens_exemplo(ledger, fundos, meta, Pv, data, seed, com_override=False, dia=i,
                                                  ids=ids):
                        ordens_estado[mesa.processar(o).estado.value] += 1
                if mesa.ordens:
                    for nome, df in mesa.tabelas().items():
                        if nome in ("ordens", "alocacoes", "auditoria"):
                            tab_ordens.append((nome, df.assign(data_serie=data)))
            # ---- atribuição do dia
            for c in cnpjs:
                for tk, q in q_prev[c].items():
                    if meta.loc[tk, "classe"] == "FUTURO_DI":
                        continue
                    pc = Pant[tk] if meta.loc[tk, "classe"] == "ACAO" else prec.preco_carrego(tk, i)
                    pnl[(c, tk)]["carrego"] += q * (pc - Pant[tk])
                    pnl[(c, tk)]["efeito_preco"] += q * (P[tk] - pc)
            for _, _, c, tipo, tk, delta, preco_l, origem, _ in ledger.linhas_desde(n0):
                if tipo == "CAIXA" and origem == "JUROS_CAIXA":
                    pnl[(c, tk)]["juros_caixa"] += delta
                elif tipo == "CAIXA" and origem.startswith("EVT:"):
                    pnl[(c, tk)]["eventos"] += delta
                elif tipo == "CAIXA" and origem == "AJUSTE_DI_DIA":
                    pnl[(c, tk)]["ajuste_futuros"] += delta
                elif tipo == "ATIVO":  # execução vs MTM do dia = custo de slippage (inclui entrada de DI)
                    pnl[(c, tk)]["custos"] += delta * (P[tk] - preco_l)
        # ---- fechamento do dia
        for f in fundos:
            c = f.cnpj
            fluxo_passivo = 0.0
            if passivo:  # passivo: cotas, taxas, tributos; os fluxos de caixa entram no ledger
                a_ativos = montar_carteira(ledger.quantidades(c), ledger.caixa_projetado(c), meta, Pv).pl
                for valor, tipo in passivo[c].fechar_dia(data, a_ativos, bench[c][i]).fluxos:
                    ledger.creditar(data, c, "CAIXA", valor, f"PASSIVO:{tipo}")
                    fluxo_passivo += valor
            q = ledger.quantidades(c)
            cart = montar_carteira(q, ledger.caixa_projetado(c), meta, Pv)
            for tk, qt in q.items():
                pos_rows.append((data, c, tk, qt, ledger.fator_vne(c, tk)))
                mtm_rows.append((data, c, tk, qt, float(P[tk]), float(cart.posicoes.loc[tk, "valor"]),
                                 float(cart.posicoes.loc[tk, "exposicao"]), fontes[tk]))
            soma = sum(sum(v.values()) for (cc, _), v in pnl.items() if cc == c)
            var_pl = cart.pl - pl_prev[c] if i > i0 else fluxo_passivo  # no 1º dia só entram aplicações do passivo
            pl_rows.append((data, c, nomes[c], ledger.caixa_disponivel(c, data), ledger.caixa_projetado(c),
                            float(cart.posicoes["valor"].sum()), cart.pl, var_pl, soma, soma - (var_pl - fluxo_passivo),
                            fluxo_passivo))
            pl_prev[c] = cart.pl
        for (c, tk), v in pnl.items():
            pnl_rows.append((data, c, tk, *[v[k] for k in COMPONENTES], sum(v.values())))
        q_prev = {f.cnpj: ledger.quantidades(f.cnpj) for f in fundos}
        cx_prev = {f.cnpj: ledger.caixa_projetado(f.cnpj) for f in fundos}

    saidas = {
        "posicoes_diarias": pd.DataFrame(pos_rows, columns=["data", "fundo_cnpj", "ticker", "quantidade", "fator_vne"]),
        "mtm_diario": pd.DataFrame(mtm_rows, columns=["data", "fundo_cnpj", "ticker", "qtd", "preco",
                                                      "valor_mercado", "exposicao", "fonte_preco"]),
        "pnl_atribuicao": pd.DataFrame(pnl_rows, columns=["data", "fundo_cnpj", "ticker", *COMPONENTES, "total"]),
        "pl_diario": pd.DataFrame(pl_rows, columns=[
            "data", "fundo_cnpj", "fundo", "caixa_disponivel", "caixa_projetado", "valor_mercado", "pl",
            "variacao_pl", "pnl_total", "erro_fechamento", "fluxo_passivo"]),
        "ledger": ledger.para_df(),
        **{f"{nome}_serie": pd.concat([df for n, df in tab_ordens if n == nome], ignore_index=True)
           if tab_ordens else pd.DataFrame() for nome in ("ordens", "alocacoes", "auditoria")},
        **_tabelas_passivo(passivo, datas[i1]),
        "ordens_por_estado": pd.DataFrame(sorted(ordens_estado.items()), columns=["estado", "n"]),
    }
    if escrever:
        for nome in TABELAS_SAIDA:
            saidas[nome].to_parquet(destino / f"{nome}.parquet", index=False)
        (destino / "serie_meta.json").write_text(json.dumps(_meta(tabelas, seed, inicio, fim, passo_ordens, com_ordens,
                                                                spread_caixa_negativo)))
    return saidas


def _tabelas_passivo(passivo: dict[str, PassivoFundo], ultima_data) -> dict[str, pd.DataFrame]:
    ps = list(passivo.values())
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return {
        "cotistas_mov": cat([p.mov for p in ps]),
        "lotes_cotistas": cat([p.lotes_df(ultima_data) for p in ps]),
        "provisoes": pd.DataFrame([r for p in ps for r in p.prov_rows]),
        "cota_diaria": pd.DataFrame([r for p in ps for r in p.cota_rows]),
        "tributos": pd.DataFrame([r for p in ps for r in p.tributos]),
    }


def _meta(tabelas, seed, inicio, fim, passo_ordens, com_ordens, spread) -> dict:
    return {"chave": impressao_digital(tabelas, seed), "seed": seed, "inicio": str(inicio), "fim": str(fim),
            "passo_ordens": passo_ordens, "com_ordens": com_ordens, "spread_caixa_negativo": spread}


def carregar_serie(seed: int = 42, destino: Path | str | None = None, inicio=None, fim=None, passo_ordens: int = 5,
                   com_ordens: bool = True, spread_caixa_negativo: float = SPREAD_CAIXA_NEGATIVO,
                   ) -> dict[str, pd.DataFrame] | None:
    """Série gravada em disco se foi gerada com os mesmos parâmetros e universo; senão None (não recalcula)."""
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    meta = destino / "serie_meta.json"
    if not meta.exists() or any(not (destino / f"{n}.parquet").exists() for n in TABELAS_SAIDA):
        return None
    tabelas = gerar_universo(seed=seed, destino=destino)
    if json.loads(meta.read_text()) != _meta(tabelas, seed, inicio, fim, passo_ordens, com_ordens,
                                             spread_caixa_negativo):
        return None
    return {n: pd.read_parquet(destino / f"{n}.parquet") for n in TABELAS_SAIDA}


def serie_pl(fundo: str, inicio=None, fim=None, seed: int = 42, destino: Path | str | None = None,
             resultado: dict[str, pd.DataFrame] | None = None) -> pd.Series:
    """PL diário de um fundo (nome ou CNPJ) de `inicio` a `fim`.

    Ordens agregadas cobrem vários fundos, então a série roda todos juntos; passe `resultado`
    (retorno de `rodar_serie`) para reaproveitar uma simulação já feita.
    """
    res = resultado if resultado is not None else rodar_serie(seed, destino, inicio, fim, escrever=False)
    pl = res["pl_diario"]
    sel = pl[(pl["fundo"] == fundo) | (pl["fundo_cnpj"] == fundo)]
    if sel.empty:
        raise KeyError(f"fundo não encontrado: {fundo}")
    s = sel.set_index("data")["pl"]
    if inicio is not None:
        s = s[s.index >= pd.Timestamp(inicio)]
    if fim is not None:
        s = s[s.index <= pd.Timestamp(fim)]
    return s.rename(sel["fundo"].iloc[0])


def resumo(res: dict[str, pd.DataFrame]) -> str:
    pl = res["pl_diario"]
    linhas = []
    for nome, g in pl.groupby("fundo", sort=False):
        s = g.set_index("data")["pl"]
        r = s.pct_change().dropna()
        pior = r.idxmin()
        linhas.append(f"{nome}: PL inicial={fmt3(s.iloc[0])} final={fmt3(s.iloc[-1])} "
                      f"retorno={fmt3(s.iloc[-1] / s.iloc[0] - 1)} pior dia={pior:%Y-%m-%d} ({fmt3(r.min())})")
    linhas.append(f"maior erro de fechamento do P&L (R$): {fmt3(float(pl['erro_fechamento'].abs().max()))}")
    oe = res["ordens_por_estado"]
    if len(oe):
        linhas.append("ordens: " + ", ".join(f"{e}={n}" for e, n in zip(oe["estado"], oe["n"])))
    return "\n".join(linhas)
