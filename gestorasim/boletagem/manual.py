"""Boletagem manual: ordens digitadas na interface, em um ledger SEPARADO (nunca altera a série oficial).

O estado de partida é o fechamento oficial da data escolhida (posições e caixa projetado gravados pelo ciclo) mais as
execuções manuais já feitas naquela data. O pré-trade usa os mesmos fundos, regulamentos, preços (mtm) e volumes
do ciclo; a liquidação pendente do fechamento não é reconstruída (o caixa disponível parte do projetado).
Tudo que é gravado fica em *_manuais.parquet / ledger_manual.parquet, com a tag MANUAL.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.boletagem.fluxo import Mesa
from gestorasim.boletagem.modelos import EstadoOrdem, OrdemAgregada
from gestorasim.dados import carregar_fundos
from gestorasim.dominio import ClasseAtivo
from gestorasim.enquadramento import avaliar_pre_trade
from gestorasim.enquadramento.pre_trade import calcular_pre
from gestorasim.enquadramento.regras import classe_de
from gestorasim.posicao import Ledger, aplicar_execucao

TABELAS_MANUAIS = ("ordens_manuais", "execucoes_manuais", "alocacoes_manuais", "auditoria_manual", "ledger_manual")
USUARIO_PADRAO = "gestor.manual"


class BoletagemManual:
    def __init__(self, destino: Path | str, seed: int = 42, destino_escrita: Path | str | None = None) -> None:
        """`destino`: dados oficiais (somente leitura); `destino_escrita`: onde ficam as tabelas manuais (default: `destino`)."""
        self.destino = Path(destino)
        self.escrita = Path(destino_escrita) if destino_escrita is not None else self.destino
        self.escrita.mkdir(parents=True, exist_ok=True)
        ler = lambda n: pd.read_parquet(self.destino / f"{n}.parquet")  # noqa: E731
        self.meta = ler("ativos").set_index("ticker")
        self.fundos = carregar_fundos(self.destino)
        self.posicoes, self.pl, self.cota = ler("posicoes_diarias"), ler("pl_diario"), ler("cota_diaria")
        self.painel = ler("mtm_painel")
        self.painel.index = pd.DatetimeIndex(self.painel.index, name="data")
        self.datas = pd.DatetimeIndex(sorted(self.pl["data"].unique()))
        self.volume = gerar_volume(self.meta, pd.DatetimeIndex(ler("calendario")["data"]), seed).pivot(
            index="data", columns="ticker", values="volume")

    # ------------------------------------------------------------------ consultas
    def ativos_negociaveis(self, data) -> list[str]:
        return sorted(self.painel.loc[pd.Timestamp(data)].dropna().index)

    def preco(self, data, ticker: str) -> float:
        return float(self.painel.loc[pd.Timestamp(data), ticker])

    def lote(self, ticker: str) -> int:
        return 100 if classe_de(self.meta, ticker) == "ACAO" else 1

    def _manual(self, nome: str) -> pd.DataFrame:
        arq = self.escrita / f"{nome}.parquet"
        return pd.read_parquet(arq) if arq.exists() else pd.DataFrame()

    # ------------------------------------------------------------------ estado de partida
    def _mesa(self, data) -> Mesa:
        data = pd.Timestamp(data)
        led = Ledger()
        pos = self.posicoes[self.posicoes["data"] == data]
        pl = self.pl[self.pl["data"] == data].set_index("fundo_cnpj")
        precos = self.painel.loc[data].dropna()
        for f in self.fundos:
            led.integralizar(data, f.cnpj, float(pl.loc[f.cnpj, "caixa_projetado"]), "SNAPSHOT")
            for r in pos[pos["fundo_cnpj"] == f.cnpj].itertuples():
                led.lancar_posicao(data, f.cnpj, r.ticker, int(r.quantidade), float(precos[r.ticker]))
        prev = self._manual("alocacoes_manuais")
        if len(prev):  # execuções manuais anteriores na mesma data
            for r in prev[prev["data"] == data].itertuples():
                aplicar_execucao(led, data=data, fundo_cnpj=r.fundo_cnpj, ticker=r.ticker,
                                 classe=classe_de(self.meta, r.ticker), lado=r.lado, quantidade=int(r.quantidade_alocada),
                                 preco=float(r.preco_medio), origem=r.ordem_id, preco_referencia=float(precos[r.ticker]))
        self.n_snapshot = len(led)  # o que vier depois é desta ordem
        cota = self.cota[self.cota["data"] == data].set_index("fundo_cnpj")
        passivos = {c: float(cota.loc[c, "adm_a_pagar"] + cota.loc[c, "perf_provisao"] + cota.loc[c, "resgates_a_pagar"])
                    for c in cota.index}
        return Mesa(led, self.fundos, self.meta, data, precos, self.volume.loc[data].dropna(), None, None, passivos)

    def _ordem(self, data, ticker: str, lado: str, qtds: dict[str, int], usuario: str, override=None) -> OrdemAgregada:
        previas = self._manual("ordens_manuais")
        n = int((previas["data"] == pd.Timestamp(data)).sum()) if len(previas) else 0
        qtds = {c: int(q) for c, q in qtds.items() if q and q > 0}
        if not qtds:
            raise ValueError("informe quantidade > 0 para ao menos um fundo")
        lt = self.lote(ticker)
        if any(q % lt for q in qtds.values()):
            raise ValueError(f"quantidade deve ser múltipla do lote ({lt})")
        return OrdemAgregada(id=f"MAN{n + 1:04d}", data=pd.Timestamp(data).date(), ticker=ticker,
                             classe=ClasseAtivo(classe_de(self.meta, ticker)), lado=lado, alvos=qtds, origem="MANUAL",
                             usuario=usuario or USUARIO_PADRAO, pedido_override=override)

    # ------------------------------------------------------------------ pré-trade ao vivo
    def pre_trade(self, data, ticker: str, lado: str, qtds: dict[str, int]) -> pd.DataFrame:
        """Regras pós-ordem por fundo (sem criar ordem nem gravar nada): fundo, regra, valor, limite, folga, status."""
        mesa = self._mesa(data)
        ordem = self._ordem(data, ticker, lado, qtds, USUARIO_PADRAO)
        linhas = []
        for cnpj, q in ordem.alvos.items():
            f = mesa.fundos[cnpj]
            qs, caixa = mesa.ledger.quantidades(cnpj), mesa.ledger.caixa_projetado(cnpj)
            perfil = mesa.perfil_caixa(cnpj)
            pre = calcular_pre(qs, caixa, f.regulamento, mesa.meta, mesa.precos, mesa.volume, perfil, None)
            for r in avaliar_pre_trade(qs, caixa, f.regulamento, mesa.meta, mesa.precos, ticker=ticker, lado=lado,
                                       quantidade=q, volume=mesa.volume, perfil_caixa=perfil, pre=pre):
                linhas.append({"fundo": f.nome, "fundo_cnpj": cnpj, "regra": r.regra, "detalhe": r.detalhe,
                               "valor_pos_ordem": r.valor_pos_ordem, "limite": r.limite, "folga": r.folga,
                               "status": r.status})
        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------ envio
    def enviar(self, data, ticker: str, lado: str, qtds: dict[str, int], usuario: str = USUARIO_PADRAO,
               override: tuple[str, str] | None = None) -> dict:
        """Cria, passa pelo pré-trade (ou override com usuário e justificativa) e executa a ordem no ledger manual.
        Devolve estado, execuções, alocações e trilha; grava nas tabelas *_manuais."""
        data = pd.Timestamp(data)
        mesa = self._mesa(data)
        ordem = self._ordem(data, ticker, lado, qtds, usuario, override)
        mesa.processar(ordem)
        t = mesa.tabelas()
        out = {"estado": ordem.estado.value, "ordem": t["ordens"], "execucoes": t["execucoes"],
               "alocacoes": t["alocacoes"], "auditoria": t["auditoria"], "pretrade": t["pretrade"]}
        etiqueta = {"data": data, "tag": "MANUAL", "usuario": usuario}
        novos = {
            "ordens_manuais": t["ordens"].assign(**etiqueta),
            "execucoes_manuais": t["execucoes"].assign(**{"tag": "MANUAL"}),
            "alocacoes_manuais": t["alocacoes"].assign(**etiqueta),
            "auditoria_manual": t["auditoria"].assign(data=data, tag="MANUAL"),
            "ledger_manual": mesa.ledger.para_df().iloc[self.n_snapshot:].assign(tag="MANUAL"),
        }
        for nome, df in novos.items():
            if len(df):
                antigo = self._manual(nome)
                pd.concat([antigo, df], ignore_index=True).to_parquet(self.escrita / f"{nome}.parquet", index=False)
        out["liberada"] = ordem.estado in (EstadoOrdem.ALOCADA,)
        return out
