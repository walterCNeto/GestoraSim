"""Gera capturas de cada página do app em docs/screens/<pagina>.png (largura 1400).

Requer playwright (`pip install playwright && playwright install chromium`). Sobe o Streamlit sobre `data/demo`,
navega pelo menu lateral (assim os filtros globais persistem) e fotografa a página inteira.

    python scripts/capturar_telas.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "docs" / "screens"
PORTA = 8599
# (rótulo no menu, arquivo de saída, filtros opcionais (fundo, data dd/mm/aaaa), ação extra)
PAGINAS = [
    ("Comece aqui", "00_comece_aqui", None),
    ("Visão geral", "01_visao_geral", None),
    ("Carteira", "02_carteira", None),
    ("Boletagem", "03_boletagem", ("Sim Ações FIA", "09/01/2023")),
    ("Risco", "04_risco", ("Sim Macro Multimercado", "31/08/2023")),
    ("Enquadramento", "05_enquadramento", ("Sim Macro Multimercado", "31/08/2023")),
    ("Passivo e cota", "06_passivo_e_cota", ("Sim Macro Multimercado", "31/05/2023")),
    ("Conciliação", "07_conciliacao", ("Sim Macro Multimercado", "31/08/2023")),
    ("Trilha de auditoria", "08_trilha_de_auditoria", ("Sim Ações FIA", "09/01/2023")),
    ("Lâmina mensal", "09_lamina_mensal", ("Sim Ações FIA", "31/08/2023")),
]


def _esperar_servidor() -> None:
    for _ in range(90):
        try:
            if urllib.request.urlopen(f"http://localhost:{PORTA}/_stcore/health", timeout=2).read() == b"ok":
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("servidor Streamlit não subiu")


def _escolher(page, rotulo: str, valor: str) -> None:
    caixa = page.locator("[data-testid='stSidebar'] [data-testid='stSelectbox']").filter(has_text=rotulo).locator("input").first
    caixa.click()
    caixa.fill(valor)
    page.get_by_role("option", name=valor).first.click()
    page.wait_for_timeout(1500)


def _quieto(page, ms: int = 2500) -> None:
    page.wait_for_timeout(ms)
    for _ in range(60):
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(800)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright não instalado: pip install playwright && playwright install chromium")
        return 1
    SAIDA.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GESTORASIM_DADOS": str(RAIZ / "data" / "demo"), "GESTORASIM_ESCRITA": str(RAIZ / "data" / "_tmp_app")}
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(RAIZ / "app" / "Comece_aqui.py"), "--server.port", str(PORTA),
                            "--server.headless", "true"], cwd=RAIZ, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _esperar_servidor()
        with sync_playwright() as p:
            nav = p.chromium.launch()
            page = nav.new_page(viewport={"width": 1400, "height": 1000})
            page.goto(f"http://localhost:{PORTA}/")
            _quieto(page, 4000)
            for rotulo, arquivo, filtros in PAGINAS:
                if rotulo != "Comece aqui":
                    page.locator("[data-testid='stSidebarNav'] a").filter(has_text=rotulo).first.click()
                    _quieto(page)
                if filtros:
                    _escolher(page, "Fundo", filtros[0])
                    _escolher(page, "Data", filtros[1])
                    _quieto(page)
                if rotulo == "Boletagem":  # ordem violadora do roteiro: compra de debênture no FIA
                    page.get_by_label("Ativo").first.click()
                    page.get_by_role("option", name="DLOG14").first.click()
                    page.get_by_label("Sim Ações FIA").fill("4888")
                    page.keyboard.press("Enter")
                    _quieto(page, 3500)
                altura = page.evaluate("document.querySelector(\"[data-testid='stMain']\").scrollHeight")
                page.set_viewport_size({"width": 1400, "height": min(max(altura + 40, 900), 7000)})
                _quieto(page, 1500)
                page.screenshot(path=str(SAIDA / f"{arquivo}.png"))
                page.set_viewport_size({"width": 1400, "height": 1000})
                print("ok", arquivo)
            nav.close()
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
