import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import dados  # noqa: E402
from app.paginas import lamina  # noqa: E402

dados.configurar("Lâmina mensal")
lamina.render()
