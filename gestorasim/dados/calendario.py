"""Calendário de dias úteis aproximado da B3 (feriados nacionais + móveis)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _pascoa(ano: int) -> date:
    """Domingo de Páscoa (algoritmo gregoriano anônimo)."""
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return date(ano, mes, dia)


def _feriados_ano(ano: int) -> list[date]:
    fixos = [(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 24), (12, 25), (12, 31)]
    if ano >= 2024:
        fixos.append((11, 20))  # Consciência Negra
    p = _pascoa(ano)
    moveis = [p - timedelta(48), p - timedelta(47), p - timedelta(2), p + timedelta(60)]
    return [date(ano, m, d) for m, d in fixos] + moveis


FERIADOS = np.array(
    sorted({f for ano in range(2020, 2046) for f in _feriados_ano(ano)}), dtype="datetime64[D]"
)


def _d(d) -> np.datetime64:
    return np.datetime64(pd.Timestamp(d).date(), "D")


def eh_dia_util(d) -> bool:
    return bool(np.is_busday(_d(d), holidays=FERIADOS))


def proximo_dia_util(d) -> pd.Timestamp:
    """Mesma data se for dia útil; senão o dia útil seguinte."""
    return pd.Timestamp(np.busday_offset(_d(d), 0, roll="forward", holidays=FERIADOS))


def somar_du(d, n: int) -> pd.Timestamp:
    return pd.Timestamp(np.busday_offset(_d(d), n, roll="forward", holidays=FERIADOS))


def du(d0, d1) -> np.ndarray | int:
    """Dias úteis em [d0, d1). Aceita escalares ou arrays."""
    d0 = np.asarray(d0, dtype="datetime64[D]") if not np.isscalar(d0) and not isinstance(d0, (date, pd.Timestamp)) else _d(d0)
    d1 = np.asarray(d1, dtype="datetime64[D]") if not np.isscalar(d1) and not isinstance(d1, (date, pd.Timestamp)) else _d(d1)
    return np.busday_count(d0, d1, holidays=FERIADOS)


def calendario_b3(inicio, n_dias: int) -> pd.DatetimeIndex:
    """Os `n_dias` primeiros dias úteis a partir de `inicio` (inclusive)."""
    primeiro = proximo_dia_util(inicio)
    offsets = np.arange(n_dias)
    datas = np.busday_offset(_d(primeiro), offsets, roll="forward", holidays=FERIADOS)
    return pd.DatetimeIndex(datas.astype("datetime64[ns]"), name="data")
