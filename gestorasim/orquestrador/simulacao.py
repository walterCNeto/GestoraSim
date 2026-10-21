"""Simulação do ciclo diário: o estado do gestor, avançado um pregão por vez em etapas nomeadas.

Ordem das etapas (cada uma grava status, duração e hash de entrada):
abertura → mercado → eventos → passivo → caixa_reenquadramento → boletagem → liquidacao → mtm →
provisoes_cota → conciliacao → pos_trade → risco → relatorios.

- abertura: juros do caixa sobre o saldo projetado do fechamento anterior.
- mercado: ajuste diário dos futuros de DI (liquida em D+1).
- eventos: dividendos, cupons e amortizações com pagamento no dia.
- passivo: pagamentos já devidos (resgates convertidos antes, taxa de administração do mês anterior).
- caixa_reenquadramento: gera as ordens de gestão de caixa e de reenquadramento.
- boletagem: pré-trade, execução e rateio dessas ordens e das regulares (rebalanceamento, aleatórias, violadoras).
- liquidacao: caixa liquidado × a liquidar (D+2 ações, D+0 títulos, D+1 ajuste de DI).
- mtm: posições, valor de mercado e atribuição de P&L do dia.
- provisoes_cota: conversões de aplicações/resgates, taxas, tributos e cota (PL líquido).
- conciliacao: arquivo XML do administrador (com divergências injetadas) e quebras do dia.
- pos_trade: regras do regulamento e da política de risco na carteira de fechamento; episódios.
- risco: VaR/ES, sensibilidades, stress, liquidez e backtest do dia.
- relatorios: linha do relatório diário por fundo.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.boletagem.fluxo import Mesa
from gestorasim.boletagem.geradores import _Seq, gerar_ordens_exemplo
from gestorasim.boletagem.gestao_caixa import ordens_gestao_caixa, ordens_reenquadramento
from gestorasim.conciliacao.diaria import ConciliacaoDiaria
from gestorasim.cota.benchmark import indice_acoes, indice_cdi
from gestorasim.cota.engine import PassivoFundo
from gestorasim.cota.parametros import ParametrosPassivo
from gestorasim.dados import calendario, carregar_fundos, gerar_universo
from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.enquadramento import montar_carteira
from gestorasim.enquadramento.pos_trade import PosTradeDiario, _ordens_do_dia
from gestorasim.enquadramento.regras import classe_de
from gestorasim.mtm.cache import painel_oficial
from gestorasim.mtm.estatistica import ewma_vol
from gestorasim.mtm.precificacao import Precificador, ajuste_di
from gestorasim.orquestrador.config import ETAPAS, ConfigCiclo
from gestorasim.posicao import PL_INICIAL, Ledger, ProcessadorEventos, gerar_carteira_inicial
from gestorasim.posicao.caixa import juros_caixa
from gestorasim.posicao.ledger import COLUNAS as COLUNAS_LEDGER

COMPONENTES = ["efeito_preco", "carrego", "eventos", "ajuste_futuros", "custos", "juros_caixa"]
_ORDEM = re.compile(r"^ORD\d+$")
COLUNAS = {
    "posicoes_diarias": ["data", "fundo_cnpj", "ticker", "quantidade", "fator_vne"],
    "mtm_diario": ["data", "fundo_cnpj", "ticker", "qtd", "preco", "valor_mercado", "exposicao", "fonte_preco"],
    "pnl_atribuicao": ["data", "fundo_cnpj", "ticker", *COMPONENTES, "total"],
    "pl_diario": ["data", "fundo_cnpj", "fundo", "caixa_disponivel", "caixa_projetado", "valor_mercado", "pl",
                  "variacao_pl", "pnl_total", "erro_fechamento", "fluxo_passivo"],
    "liquidacao_diaria": ["data", "fundo_cnpj", "caixa_disponivel", "caixa_projetado", "a_liquidar", "liquidado_no_dia"],
    "sensibilidades": ["data", "fundo_cnpj", "fator", "delta", "gama"],
    "stress": ["data", "fundo_cnpj", "cenario", "pnl", "pnl_pct_pl"],
    "liquidez": ["data", "fundo_cnpj", "ticker", "valor_mercado", "dias_para_zerar", "bucket"],
}
COLUNAS_QUEBRA = ["data", "fundo_cnpj", "tipo", "item", "valor_gestor", "valor_adm", "diferenca", "status", "diagnostico"]
COLUNAS_INJECAO = ["id_injecao", "data", "fundo_cnpj", "tipo", "item", "valor_original", "valor_injetado"]
COLUNAS_PARALELA = ["data", "fundo_cnpj", "cota_gestor", "cota_adm", "diferenca_bp"]
COLUNAS_ORDENS = {"ordens": "ordens_serie", "alocacoes": "alocacoes_serie", "auditoria": "auditoria_serie"}


class Dia:
    """Contexto de um pregão: o que uma etapa deixa para as seguintes."""

    def __init__(self, i: int, data: pd.Timestamp, P: pd.Series, Pant: pd.Series | None, q_prev: dict) -> None:
        self.i, self.data, self.P, self.Pant, self.q_prev = i, data, P, Pant, q_prev
        self.Pv = P.dropna()
        self.n0 = 0
        self.pnl = defaultdict(lambda: dict.fromkeys(COMPONENTES, 0.0))
        self.fluxo: dict[str, float] = defaultdict(float)  # fluxos de caixa do passivo no dia, por fundo
        self.mesa: Mesa | None = None
        self.ids: _Seq | None = None
        self.previas: list = []
        self.saidas: dict = {k: [] for k in ("pos", "mtm", "pnl", "pl", "liq", "ordens", "risco", "sens", "stress",
                                              "liqpos", "pt_linhas", "pt_alertas", "quebras", "injecoes", "paralela", "rel")}
        self.cota_rows: dict[str, dict] = {}
        self.risco_linhas: dict[str, dict] = {}
        self._serie_dia: dict | None = None
        self.log: list[dict] = []


class Simulacao:
    def __init__(self, cfg: ConfigCiclo, de=None, ate=None) -> None:
        self.cfg = cfg
        self.destino = Path(cfg.destino) if cfg.destino is not None else DESTINO_PADRAO
        seed = cfg.seed
        self.tabelas = tabelas = gerar_universo(seed=seed, destino=self.destino)
        self.fundos = carregar_fundos(self.destino)
        self.nomes = {f.cnpj: f.nome for f in self.fundos}
        self.cnpjs = [f.cnpj for f in self.fundos]
        self.prec = prec = Precificador(tabelas, seed)
        self.meta, self.datas, self.estados = prec.meta, prec.datas, prec.estados
        self.painel = painel_oficial(tabelas, seed, self.destino)
        self.fontes = {tk: prec.fonte(tk) for tk in self.meta.index}
        self.volume = gerar_volume(self.meta, self.datas, seed).pivot(index="data", columns="ticker", values="volume")
        self.i0 = 0 if de is None else int(self.datas.searchsorted(pd.Timestamp(de)))
        self.i1 = len(self.datas) - 1 if ate is None else int(self.datas.searchsorted(pd.Timestamp(ate), side="right")) - 1
        di = list(self.meta.index[self.meta["classe"] == "FUTURO_DI"])
        self.sigma_di = pd.DataFrame({tk: ewma_vol(self.painel[tk].pct_change().to_numpy()) for tk in di}, index=self.datas)
        # dividendos por data ex (pregão seguinte à data com): para o P&L hipotético do risco
        self.ex_div: dict[int, list[tuple[str, float, int]]] = defaultdict(list)
        for _, e in tabelas["eventos"].query("tipo == 'DIVIDENDO'").iterrows():
            i_com = prec.i_de[pd.Timestamp(e["data_com"])]
            self.ex_div[i_com + 1].append((e["ticker"], float(e["valor_por_unidade"]), i_com))

        # ---- risco (fatores e sensibilidades são puros: ficam fora do estado)
        self.calc = self.fatores = self.reducao = self.motor_risco = self.pos_trade = self.conc = None
        precisa_risco = cfg.com_risco or cfg.com_pos_trade or (cfg.com_ordens and cfg.com_politica_reenquadramento)
        if precisa_risco:
            from gestorasim.risco.fatores import construir_fatores
            from gestorasim.risco.motor import MotorRiscoDiario
            from gestorasim.risco.reenquadramento import ReducaoPolitica
            from gestorasim.risco.sensibilidades import CalculadoraSens

            self.fatores = construir_fatores(tabelas, prec, self.painel)
            self.calc = CalculadoraSens(prec, self.fatores, self.painel)
            if cfg.com_ordens and cfg.com_politica_reenquadramento:
                self.reducao = ReducaoPolitica(self.calc, self.fatores)
            if cfg.com_risco:
                self.motor_risco = MotorRiscoDiario(prec, self.painel, self.volume, self.calc, cfg.janela_hist, cfg.burn_in)
            if cfg.com_pos_trade:
                self.pos_trade = PosTradeDiario(prec, self.painel, self.volume, self.fundos, self.calc,
                                                prazo_reenquadramento_du=cfg.prazo_reenquadramento_du)
        if cfg.com_conciliacao and cfg.com_passivo:
            self.conc = ConciliacaoDiaria(seed, self.meta, self.destino, cfg.taxa_injecao, cfg.escrever_xml)

        # ---- estado (tudo que precisa sobreviver de um pregão ao seguinte) -> checkpoints
        self.ledger = Ledger()
        gerar_carteira_inicial(self.ledger, self.fundos, self.meta, self.painel.iloc[self.i0].dropna(),
                               self.datas[self.i0], seed)
        self.proc = ProcessadorEventos(self.ledger, tabelas["eventos"], prec.fracoes)
        self.caixa_alvo = {f.cnpj: self.ledger.caixa_projetado(f.cnpj) / PL_INICIAL for f in self.fundos}
        self.passivo: dict[str, PassivoFundo] = {}
        self.bench: dict[str, np.ndarray] = {}
        if cfg.com_passivo:
            params = cfg.params_passivo or ParametrosPassivo()
            cdi_idx, acoes_idx = indice_cdi(tabelas), indice_acoes(tabelas, self.painel, seed)
            for f in self.fundos:
                regras = params.regras[f.classe]
                self.passivo[f.cnpj] = PassivoFundo(f, regras, params, tabelas["fluxo_cotistas"].query("fundo_cnpj == @f.cnpj"))
                self.bench[f.cnpj] = acoes_idx if regras.benchmark == "ACOES" else cdi_idx
        self.q_prev: dict[str, dict[str, int]] = {}
        self.cx_prev: dict[str, float] = {}
        self.pl_prev: dict[str, float] = {}
        self.cota_prev: dict[str, float] = {}
        self.ordens_estado: Counter = Counter()
        self.persistencia: dict = {}  # histerese do reenquadramento
        self.dias: dict[int, dict] = {}  # saídas dos pregões executados nesta instância

    # ================================================================== estado / checkpoints
    def estado(self) -> dict:
        return {"ledger": self.ledger, "proc": self.proc, "passivo": self.passivo, "q_prev": self.q_prev,
                "cx_prev": self.cx_prev, "pl_prev": self.pl_prev, "cota_prev": self.cota_prev,
                "ordens_estado": self.ordens_estado, "persistencia": self.persistencia,
                "risco_anterior": None if self.motor_risco is None else (self.motor_risco.anterior, self.motor_risco.i_primeira),
                "pos_trade": None if self.pos_trade is None else (self.pos_trade.abertos, self.pos_trade.todos)}

    def restaurar(self, e: dict) -> None:
        for k in ("ledger", "proc", "passivo", "q_prev", "cx_prev", "pl_prev", "cota_prev", "ordens_estado", "persistencia"):
            setattr(self, k, e[k])
        if self.motor_risco is not None and e["risco_anterior"] is not None:
            self.motor_risco.anterior, self.motor_risco.i_primeira = e["risco_anterior"]
        if self.pos_trade is not None and e["pos_trade"] is not None:
            self.pos_trade.abertos, self.pos_trade.todos = e["pos_trade"]

    def salvar_checkpoint(self, i: int, pasta: Path) -> Path:
        pasta.mkdir(parents=True, exist_ok=True)
        caminho = pasta / f"ckpt_{i:04d}.pkl"
        with open(caminho, "wb") as f:
            pickle.dump({"i": i, "estado": self.estado()}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return caminho

    def carregar_checkpoint(self, caminho: Path) -> int:
        with open(caminho, "rb") as f:
            dados = pickle.load(f)
        self.restaurar(dados["estado"])
        return int(dados["i"])

    # ================================================================== execução
    def executar_dia(self, i: int, registrar: bool = True) -> Dia:
        """Roda as 13 etapas do pregão `i`. Se uma etapa falha (ERRO), as seguintes ficam PULADAS e o erro sobe
        depois de gravado o log."""
        data = self.datas[i]
        d = Dia(i, data, self.painel.iloc[i], self.painel.iloc[i - 1] if i > 0 else None,
                {c: dict(q) for c, q in self.q_prev.items()})
        d.n0 = len(self.ledger)
        falhou = None
        for ordem, etapa in enumerate(ETAPAS, 1):
            reg = {"data": data, "etapa": etapa, "ordem": ordem, "hash_entrada": "", "duracao_s": 0.0, "mensagem": ""}
            if falhou is not None:
                reg["status"] = "PULADA"
                reg["mensagem"] = f"etapa anterior com erro ({falhou[0]})"
            elif not self._aplica(etapa, i):
                reg["status"] = "PULADA"
                reg["mensagem"] = "não se aplica neste pregão / desativada na configuração"
            else:
                reg["hash_entrada"] = self._digest(etapa, d)
                t0 = time.perf_counter()
                try:
                    getattr(self, f"_etapa_{etapa}")(d)
                    reg["status"] = "OK"
                except Exception as exc:  # noqa: BLE001 - qualquer falha de etapa é registrada
                    reg["status"] = "ERRO"
                    reg["mensagem"] = f"{type(exc).__name__}: {exc}"
                    falhou = (etapa, traceback.format_exc())
                reg["duracao_s"] = time.perf_counter() - t0
            d.log.append(reg)
        if falhou is not None:
            self.dias[i] = self._saidas(d, registrar)
            raise RuntimeError(f"erro na etapa '{falhou[0]}' em {data:%Y-%m-%d}\n{falhou[1]}")
        self.q_prev = {c: self.ledger.quantidades(c) for c in self.cnpjs}
        self.cx_prev = {c: self.ledger.caixa_projetado(c) for c in self.cnpjs}
        self.dias[i] = self._saidas(d, registrar)
        return d

    def _saidas(self, d: Dia, registrar: bool) -> dict:
        s = d.saidas
        s["log"] = d.log if registrar else []
        return s

    def _aplica(self, etapa: str, i: int) -> bool:
        cfg, inicio = self.cfg, i == self.i0
        return {
            "abertura": not inicio and cfg.com_juros_caixa,
            "mercado": not inicio,
            "eventos": not inicio,
            "passivo": cfg.com_passivo,
            "caixa_reenquadramento": not inicio and cfg.com_ordens and (cfg.com_gestao_caixa or cfg.com_reenquadramento),
            "boletagem": not inicio and cfg.com_ordens,
            "liquidacao": True,
            "mtm": True,
            "provisoes_cota": True,  # sem passivo, apenas fecha o PL dos ativos
            "conciliacao": self.conc is not None,
            "pos_trade": self.pos_trade is not None and cfg.com_passivo,
            "risco": self.motor_risco is not None and cfg.com_passivo,
            "relatorios": True,
        }[etapa]

    def _digest(self, etapa: str, d: Dia) -> str:
        """Hash do que a etapa recebe: posições, caixa, preços do dia e nº de lançamentos."""
        h = hashlib.sha1(f"{etapa}|{d.data:%Y-%m-%d}|{len(self.ledger)}|{self.cfg.seed}".encode())
        for c in self.cnpjs:
            h.update(repr(sorted(self.ledger.quantidades(c).items())).encode())
            h.update(f"{self.ledger.caixa_projetado(c):.4f}".encode())
        h.update(f"{float(d.Pv.sum()):.6f}".encode())
        return h.hexdigest()[:16]

    # ================================================================== etapas
    def _etapa_abertura(self, d: Dia) -> None:
        for c in self.cnpjs:  # juros do caixa sobre o saldo projetado do fechamento anterior
            j = juros_caixa(self.cx_prev[c], self.estados[d.i - 1].cdi_aa, self.cfg.spread_caixa_negativo)
            self.ledger.creditar(d.data, c, "CAIXA", j, "JUROS_CAIXA")

    def _etapa_mercado(self, d: Dia) -> None:
        for c in self.cnpjs:  # ajuste diário dos futuros de DI, liquidado em D+1
            for tk, q in d.q_prev[c].items():
                if classe_de(self.meta, tk) == "FUTURO_DI":
                    aj = ajuste_di(d.P[tk], d.Pant[tk], self.estados[d.i - 1].cdi_aa, q)
                    self.ledger.creditar(d.data, c, tk, aj, "AJUSTE_DI_DIA", calendario.somar_du(d.data, 1))

    def _etapa_eventos(self, d: Dia) -> None:
        self.proc.processar(d.data, self.cnpjs)

    def _etapa_passivo(self, d: Dia) -> None:
        for c in self.cnpjs:
            d.fluxo[c] = 0.0
            for valor, tipo in self.passivo[c].abrir_dia(d.data):
                self.ledger.creditar(d.data, c, "CAIXA", valor, f"PASSIVO:{tipo}")
                d.fluxo[c] += valor

    def _mesa(self, d: Dia) -> Mesa:
        agenda = {c: self.passivo[c].saidas_previstas(d.data, [calendario.somar_du(d.data, k) for k in range(3)])
                  for c in self.passivo}
        passivos = {c: self.passivo[c].passivos_total for c in self.passivo}
        return Mesa(self.ledger, self.fundos, self.meta, d.data, d.Pv, self.volume.iloc[d.i].dropna(),
                    self.sigma_di.iloc[d.i].to_dict(), agenda, passivos)

    def _etapa_caixa_reenquadramento(self, d: Dia) -> None:
        cfg = self.cfg
        d.mesa, d.ids = self._mesa(d), _Seq()
        previas = []
        if cfg.com_gestao_caixa:
            previas += ordens_gestao_caixa(d.mesa, self.caixa_alvo, d.ids, tolerancia_excesso=cfg.tolerancia_reinvestimento)
        if cfg.com_reenquadramento:
            previas += ordens_reenquadramento(d.mesa, d.ids, d.i, self.reducao, self.persistencia,
                                              cfg.histerese_reenquadramento_du)
        d.previas = previas

    def _etapa_boletagem(self, d: Dia) -> None:
        cfg = self.cfg
        if d.mesa is None:
            d.mesa, d.ids = self._mesa(d), _Seq()
        for o in sorted(d.previas, key=lambda o: (o.lado.value != "V", o.id)):
            self.ordens_estado[d.mesa.processar(o).estado.value] += 1
        if (d.i - self.i0) % cfg.passo_ordens == 0:
            for o in gerar_ordens_exemplo(self.ledger, self.fundos, self.meta, d.Pv, d.data, cfg.seed,
                                          com_override=False, dia=d.i, ids=d.ids):
                self.ordens_estado[d.mesa.processar(o).estado.value] += 1
        if d.mesa.ordens:
            for nome, df in d.mesa.tabelas().items():
                if nome in COLUNAS_ORDENS:
                    d.saidas["ordens"].append((COLUNAS_ORDENS[nome], df.assign(data_serie=d.data)))

    def _etapa_liquidacao(self, d: Dia) -> None:
        for c in self.cnpjs:
            proj, disp = self.ledger.caixa_projetado(c), self.ledger.caixa_disponivel(c, d.data)
            d.saidas["liq"].append((d.data, c, disp, proj, proj - disp, self.ledger.liquidado_em(c, d.data)))

    def _etapa_mtm(self, d: Dia) -> None:
        meta, i = self.meta, d.i
        if i > self.i0:  # atribuição do dia
            for c in self.cnpjs:
                for tk, q in d.q_prev[c].items():
                    if classe_de(meta, tk) == "FUTURO_DI":
                        continue
                    pc = d.Pant[tk] if classe_de(meta, tk) == "ACAO" else self.prec.preco_carrego(tk, i)
                    d.pnl[(c, tk)]["carrego"] += q * (pc - d.Pant[tk])
                    d.pnl[(c, tk)]["efeito_preco"] += q * (d.P[tk] - pc)
            for _, _, c, tipo, tk, delta, preco_l, origem, _ in self.ledger.linhas_desde(d.n0):
                if tipo == "CAIXA" and origem == "JUROS_CAIXA":
                    d.pnl[(c, tk)]["juros_caixa"] += delta
                elif tipo == "CAIXA" and origem.startswith("EVT:"):
                    d.pnl[(c, tk)]["eventos"] += delta
                elif tipo == "CAIXA" and origem == "AJUSTE_DI_DIA":
                    d.pnl[(c, tk)]["ajuste_futuros"] += delta
                elif tipo == "ATIVO":  # execução vs MTM do dia = custo de slippage (inclui entrada de DI)
                    d.pnl[(c, tk)]["custos"] += delta * (d.P[tk] - preco_l)
        for c in self.cnpjs:
            q = self.ledger.quantidades(c)
            cart = montar_carteira(q, self.ledger.caixa_projetado(c), meta, d.Pv)
            for tk, qt in q.items():
                lin = cart.linha(tk)
                d.saidas["pos"].append((d.data, c, tk, qt, self.ledger.fator_vne(c, tk)))
                d.saidas["mtm"].append((d.data, c, tk, qt, float(d.P[tk]), lin.valor, lin.exposicao, self.fontes[tk]))
        for (c, tk), v in d.pnl.items():
            d.saidas["pnl"].append((d.data, c, tk, *[v[k] for k in COMPONENTES], sum(v.values())))

    def _etapa_provisoes_cota(self, d: Dia) -> None:
        i, meta = d.i, self.meta
        for c in self.cnpjs:
            if self.passivo:
                a_ativos = montar_carteira(self.ledger.quantidades(c), self.ledger.caixa_projetado(c), meta, d.Pv).pl
                r = self.passivo[c].fechar_dia(d.data, a_ativos, self.bench[c][i], abertura_lancada=True)
                for valor, tipo in r.fluxos:
                    self.ledger.creditar(d.data, c, "CAIXA", valor, f"PASSIVO:{tipo}")
                    d.fluxo[c] += valor
                d.cota_rows[c] = r.cota
            cart = montar_carteira(self.ledger.quantidades(c), self.ledger.caixa_projetado(c), meta, d.Pv)
            soma = sum(sum(v.values()) for (cc, _), v in d.pnl.items() if cc == c)
            var_pl = cart.pl - self.pl_prev[c] if i > self.i0 else d.fluxo[c]  # 1º dia: só aplicações do passivo
            d.saidas["pl"].append((d.data, c, self.nomes[c], self.ledger.caixa_disponivel(c, d.data),
                                   self.ledger.caixa_projetado(c), cart.valor_mercado, cart.pl, var_pl, soma,
                                   soma - (var_pl - d.fluxo[c]), d.fluxo[c]))
            self.pl_prev[c] = cart.pl

    def _serie_dia(self, d: Dia) -> dict[str, pd.DataFrame]:
        """Tabelas do pregão, no formato que os módulos de conciliação e pós-trade esperam."""
        if d._serie_dia is None:
            ordens = [df for n, df in d.saidas["ordens"] if n == "alocacoes_serie"]
            auds = [df for n, df in d.saidas["ordens"] if n == "auditoria_serie"]
            d._serie_dia = {
                "cota_diaria": pd.DataFrame(list(d.cota_rows.values())),
                "pl_diario": pd.DataFrame(d.saidas["pl"], columns=COLUNAS["pl_diario"]),
                "mtm_diario": pd.DataFrame(d.saidas["mtm"], columns=COLUNAS["mtm_diario"]),
                "ledger": pd.DataFrame(self.ledger.linhas_desde(d.n0), columns=COLUNAS_LEDGER),
                "alocacoes_serie": pd.concat(ordens, ignore_index=True) if ordens else pd.DataFrame(),
                "auditoria_serie": pd.concat(auds, ignore_index=True) if auds else pd.DataFrame(),
            }
        return d._serie_dia

    def _etapa_conciliacao(self, d: Dia) -> None:
        r = self.conc.dia(d.i, self._serie_dia(d))
        d.saidas["quebras"], d.saidas["injecoes"], d.saidas["paralela"] = r["quebras"], r["injecoes"], r["cota_paralela"]

    def _passivos(self, d: Dia, c: str) -> float:
        row = d.cota_rows[c]
        return row["adm_a_pagar"] + row["perf_provisao"] + row["resgates_a_pagar"]

    def _etapa_pos_trade(self, d: Dia) -> None:
        from gestorasim.risco.motor import metricas_politica

        sd = self._serie_dia(d)
        alocs = _ordens_do_dia(sd) if len(sd["alocacoes_serie"]) else {}
        led = sd["ledger"]
        cx = led[(led["tipo"] == "CAIXA") & led["origem"].str.match(_ORDEM)]
        for c in self.cnpjs:
            q = self.ledger.quantidades(c)
            pl_liq = d.cota_rows[c]["pl_liquido"]
            m = metricas_politica(self.calc, self.fatores, q, d.i, pl_liq)
            cxf = cx[cx["fundo_cnpj"] == c]
            tr = (float(cxf["delta"].sum()), float(cxf.loc[cxf["data_liquidacao"] <= cxf["data"], "delta"].sum()))
            linhas, alertas = self.pos_trade.dia(
                d.i, d.data, c, self.ledger.caixa_projetado(c), self.ledger.caixa_disponivel(c, d.data), q,
                d.q_prev.get(c, {}), m["var_pct"], m["stress_pct"], self._passivos(d, c),
                alocs.get((d.data, c), []), tr)
            d.saidas["pt_linhas"] += linhas
            d.saidas["pt_alertas"] += alertas

    def _hip(self, d: Dia, c: str) -> tuple[float | None, float | None]:
        """P&L hipotético do dia: limpo (preço + ajuste de futuros + dividendo na data ex) e sujo (+ carrego e
        eventos de renda fixa)."""
        if d.i == self.i0:
            return None, None
        limpo = sujo_extra = 0.0
        for (cc, tk), v in d.pnl.items():
            classe = self.prec._classe.get(tk)
            if cc != c or classe is None:  # "CAIXA" (juros) fica fora do P&L de fatores
                continue
            limpo += v["efeito_preco"] + v["ajuste_futuros"]
            sujo_extra += v["carrego"] + (v["eventos"] if classe.value != "ACAO" else 0.0)
        for tk, valor, i_com in self.ex_div.get(d.i, []):
            limpo += self.ledger.quantidade_em(c, tk, self.datas[i_com]) * valor
        return limpo, limpo + sujo_extra

    def _etapa_risco(self, d: Dia) -> None:
        for c in self.cnpjs:
            q = self.ledger.quantidades(c)
            limpo, sujo = self._hip(d, c)
            r = self.motor_risco.dia(d.i, d.data, c, self.nomes[c], q, d.cota_rows[c]["pl_liquido"],
                                     self.ledger.caixa_projetado(c), limpo, sujo)
            d.risco_linhas[c] = r["risco"]
            d.saidas["risco"].append(r["risco"])
            d.saidas["sens"] += r["sens"]
            d.saidas["stress"] += r["stress"]
            d.saidas["liqpos"] += r["liq"]

    def _etapa_relatorios(self, d: Dia) -> None:
        alocadas = bloqueadas = 0
        for nome, df in d.saidas["ordens"]:
            if nome == "ordens_serie":
                alocadas += int((df["estado"] == "ALOCADA").sum())
                bloqueadas += int((df["estado"] == "BLOQUEADA").sum())
        for c in self.cnpjs:
            cr = d.cota_rows.get(c)
            alertas = [a for a in d.saidas["pt_alertas"] if a["fundo_cnpj"] == c]
            linha = {
                "data": d.data, "fundo_cnpj": c, "fundo": self.nomes[c],
                "pl_liquido": cr["pl_liquido"] if cr else np.nan, "cota_liquida": cr["cota_liquida"] if cr else np.nan,
                "retorno_dia_cota": (cr["cota_liquida"] / self.cota_prev[c] - 1) if cr and c in self.cota_prev else np.nan,
                "var_param_pct": d.risco_linhas.get(c, {}).get("var_param_pct", np.nan),
                "alertas_hard": sum(a["nivel"] == "HARD" for a in alertas),
                "alertas_soft": sum(a["nivel"] == "SOFT" for a in alertas),
                "quebras": sum(q["fundo_cnpj"] == c for q in d.saidas["quebras"]),
                "ordens_alocadas": alocadas, "ordens_bloqueadas": bloqueadas,
            }
            d.saidas["rel"].append(linha)
            if cr:
                self.cota_prev[c] = cr["cota_liquida"]

    # ================================================================== tabelas
    def tabelas_dias(self) -> dict[str, pd.DataFrame]:
        """Tabelas por pregão (com coluna de data) dos dias executados nesta instância."""
        ordem = sorted(self.dias)
        junta = lambda chave: [x for i in ordem for x in self.dias[i][chave]]  # noqa: E731
        out = {
            "posicoes_diarias": pd.DataFrame(junta("pos"), columns=COLUNAS["posicoes_diarias"]),
            "mtm_diario": pd.DataFrame(junta("mtm"), columns=COLUNAS["mtm_diario"]),
            "pnl_atribuicao": pd.DataFrame(junta("pnl"), columns=COLUNAS["pnl_atribuicao"]),
            "pl_diario": pd.DataFrame(junta("pl"), columns=COLUNAS["pl_diario"]),
            "liquidacao_diaria": pd.DataFrame(junta("liq"), columns=COLUNAS["liquidacao_diaria"]),
            "risco_diario": pd.DataFrame(junta("risco")),
            "sensibilidades": pd.DataFrame(junta("sens"), columns=COLUNAS["sensibilidades"]),
            "stress": pd.DataFrame(junta("stress"), columns=COLUNAS["stress"]),
            "liquidez": pd.DataFrame(junta("liqpos"), columns=COLUNAS["liquidez"]),
            "enquadramento_diario": pd.DataFrame(junta("pt_linhas")),
            "alertas": pd.DataFrame(junta("pt_alertas")),
            "quebras_conciliacao": pd.DataFrame(junta("quebras"), columns=COLUNAS_QUEBRA),
            "injecoes": pd.DataFrame(junta("injecoes"), columns=COLUNAS_INJECAO),
            "cota_paralela": pd.DataFrame(junta("paralela"), columns=COLUNAS_PARALELA),
            "relatorio_diario": pd.DataFrame(junta("rel")),
        }
        for destino_nome in COLUNAS_ORDENS.values():
            partes = [df for i in ordem for n, df in self.dias[i]["ordens"] if n == destino_nome]
            out[destino_nome] = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
        return out

    def tabelas_estado(self) -> dict[str, pd.DataFrame]:
        """Tabelas que vêm do estado acumulado (ledger, passivo, episódios), sempre completas."""
        ps = list(self.passivo.values())
        cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
        out = {
            "ledger": self.ledger.para_df(),
            "cotistas_mov": cat([p.mov for p in ps]),
            "lotes_cotistas": cat([p.lotes_df(self.datas[self.i1]) for p in ps]),
            "provisoes": pd.DataFrame([r for p in ps for r in p.prov_rows]),
            "cota_diaria": pd.DataFrame([r for p in ps for r in p.cota_rows]),
            "tributos": pd.DataFrame([r for p in ps for r in p.tributos]),
        }
        if self.pos_trade is not None:
            out["episodios"] = self.pos_trade.episodios()
        return out

    def log(self) -> pd.DataFrame:
        linhas = [r for i in sorted(self.dias) for r in self.dias[i].get("log", [])]
        return pd.DataFrame(linhas, columns=["data", "etapa", "ordem", "status", "duracao_s", "hash_entrada", "mensagem"])
