"""Formatação: números com três casas decimais."""

from __future__ import annotations

import pandas as pd


def tres_casas(df: pd.DataFrame):
    """Styler que mostra colunas numéricas (exceto inteiros) com 3 casas decimais; ordenação continua numérica."""
    fmt = {c: "{:.3f}" for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    return df.style.format(fmt, na_rep="")
