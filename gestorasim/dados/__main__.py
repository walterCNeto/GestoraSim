from gestorasim.dados import gerar_universo, resumo

if __name__ == "__main__":
    print(resumo(gerar_universo(seed=42)))
