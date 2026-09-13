"""
Análise exploratória de dados (AED): impactos socioeconômicos da COVID-19 no Brasil.

Fontes:
- Our World in Data (COVID-19) — recorte do Brasil
- Google COVID-19 Community Mobility Reports — agregado nacional do Brasil

Execução:
    .venv/bin/python analise.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DIR_DADOS = Path("data")
DIR_FIGURAS = Path("figures")
DIR_FIGURAS.mkdir(exist_ok=True)


def carregar_e_limpar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os recortes do Brasil, trata valores ausentes e une pela data."""
    owid = pd.read_csv(DIR_DADOS / "brazil_owid.csv", parse_dates=["date"])
    mobilidade = pd.read_csv(DIR_DADOS / "brazil_mobility.csv", parse_dates=["date"])

    colunas_owid = [
        "date",
        "new_cases_smoothed",
        "new_deaths_smoothed",
        "stringency_index",
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "total_deaths_per_million",
    ]
    covid = owid[colunas_owid].copy()

    # Prefere séries suavizadas (a notificação diária ficou irregular/semanal depois)
    covid["new_cases_smoothed"] = covid["new_cases_smoothed"].interpolate(limit_direction="both")
    covid["new_deaths_smoothed"] = covid["new_deaths_smoothed"].interpolate(limit_direction="both")
    # O índice de restrição termina no meio da série; preenche só lacunas curtas
    covid["stringency_index"] = covid["stringency_index"].ffill(limit=7)

    colunas_mob = [
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    mob = mobilidade[colunas_mob].copy()
    mob = mob.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Média móvel de 7 dias para reduzir sazonalidade semanal da mobilidade
    for col in colunas_mob[1:]:
        mob[f"{col}_ma7"] = mob[col].rolling(7, min_periods=1).mean()

    unido = covid.merge(mob, on="date", how="inner")
    unido = unido.sort_values("date").reset_index(drop=True)

    print("=== Resumo da limpeza ===")
    print(
        f"Linhas OWID (Brasil): {len(covid):,} | "
        f"Linhas de mobilidade: {len(mob):,} | "
        f"Após união: {len(unido):,}"
    )
    print(f"Período: {unido['date'].min().date()} → {unido['date'].max().date()}")
    print(f"Restrição ausente após limpeza: {unido['stringency_index'].isna().mean():.1%}")
    return unido, covid


def imprimir_insights(df: pd.DataFrame) -> None:
    """Responde perguntas orientadas por dados com base na série unida."""
    pior_idx = df["workplaces_percent_change_from_baseline"].idxmin()
    pior = df.loc[pior_idx]

    def estatisticas_periodo(inicio: str, fim: str) -> pd.Series:
        recorte = df[(df["date"] >= inicio) & (df["date"] <= fim)]
        return pd.Series(
            {
                "trabalho_media": recorte["workplaces_percent_change_from_baseline"].mean(),
                "obitos_media": recorte["new_deaths_smoothed"].mean(),
                "restricao_media": recorte["stringency_index"].mean(),
                "varejo_media": recorte[
                    "retail_and_recreation_percent_change_from_baseline"
                ].mean(),
            }
        )

    abr2020 = estatisticas_periodo("2020-04-01", "2020-04-30")
    abr2021 = estatisticas_periodo("2021-04-01", "2021-04-30")
    abr2022 = estatisticas_periodo("2022-04-01", "2022-04-30")

    corr = df[
        [
            "workplaces_percent_change_from_baseline",
            "new_deaths_smoothed",
            "stringency_index",
        ]
    ].corr()

    print("\n=== Perguntas orientadas por dados ===")
    print(
        "P1: Quando a queda da mobilidade no trabalho foi mais intensa?\n"
        f"    R: {pior['date'].date()} "
        f"({pior['workplaces_percent_change_from_baseline']:.0f}% em relação à linha de base)."
    )
    print(
        "P2: Voltar ao trabalho coincidiu com menor mortalidade?\n"
        f"    R: Abr/2020 trabalho {abr2020['trabalho_media']:.1f}% | "
        f"óbitos/dia {abr2020['obitos_media']:.0f} | restrição {abr2020['restricao_media']:.1f}\n"
        f"       Abr/2021 trabalho {abr2021['trabalho_media']:.1f}% | "
        f"óbitos/dia {abr2021['obitos_media']:.0f} | restrição {abr2021['restricao_media']:.1f}\n"
        "       → A mobilidade se recuperou enquanto o pico de óbitos piorou (onda Gama)."
    )
    print(
        "P3: Qual a relação entre restrições e presença no trabalho?\n"
        f"    R: Correlação = "
        f"{corr.loc['workplaces_percent_change_from_baseline', 'stringency_index']:.2f} "
        "(mais restrição ↔ menos presença no trabalho)."
    )
    print(
        "P4: Em abr/2022 a atividade já havia se recuperado?\n"
        f"    R: Trabalho {abr2022['trabalho_media']:+.1f}% vs linha de base; "
        f"óbitos/dia cerca de {abr2022['obitos_media']:.0f}."
    )


def grafico_mobilidade_vs_obitos(df: pd.DataFrame) -> None:
    """Gráfico de linhas: mobilidade (trabalho/varejo) x óbitos diários."""
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(
        df["date"],
        df["workplaces_percent_change_from_baseline_ma7"],
        color="#1f4e79",
        label="Locais de trabalho (MM 7 dias, % vs base)",
        linewidth=1.8,
    )
    ax1.plot(
        df["date"],
        df["retail_and_recreation_percent_change_from_baseline_ma7"],
        color="#6b8f71",
        label="Varejo e lazer (MM 7 dias, % vs base)",
        linewidth=1.5,
        alpha=0.9,
    )
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Variação da mobilidade vs linha de base (%)")
    ax1.set_xlabel("Data")

    ax2 = ax1.twinx()
    ax2.fill_between(
        df["date"],
        df["new_deaths_smoothed"],
        color="#c0392b",
        alpha=0.25,
        label="Óbitos diários (suavizados)",
    )
    ax2.set_ylabel("Óbitos diários (suavizados)")

    linhas1, rotulos1 = ax1.get_legend_handles_labels()
    linhas2, rotulos2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc="upper right", fontsize=8)
    ax1.set_title("Brasil: recuperação da mobilidade x óbitos por COVID-19 (2020–2022)")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "01_mobilidade_vs_obitos.png", dpi=150)
    plt.close(fig)


def grafico_restricao_boxplot(df: pd.DataFrame) -> None:
    """Boxplot: mobilidade no trabalho por faixa do índice de restrição."""
    plot_df = df.dropna(
        subset=["stringency_index", "workplaces_percent_change_from_baseline"]
    ).copy()
    plot_df["faixa_restricao"] = pd.cut(
        plot_df["stringency_index"],
        bins=[0, 40, 60, 80, 100],
        labels=["Baixa (0–40)", "Média (40–60)", "Alta (60–80)", "Muito alta (80–100)"],
        include_lowest=True,
    )

    faixas = ["Baixa (0–40)", "Média (40–60)", "Alta (60–80)", "Muito alta (80–100)"]
    dados = [
        plot_df.loc[
            plot_df["faixa_restricao"] == faixa,
            "workplaces_percent_change_from_baseline",
        ].dropna()
        for faixa in faixas
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(dados, tick_labels=faixas, patch_artist=True)
    cores = ["#a8d5a2", "#f0c674", "#e67e22", "#c0392b"]
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mobilidade no trabalho (% vs linha de base)")
    ax.set_xlabel("Faixa do Índice de Restrição de Oxford")
    ax.set_title("A presença no trabalho cai conforme aumentam as restrições")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "02_restricao_vs_trabalho_box.png", dpi=150)
    plt.close(fig)


def grafico_comparacao_abril(df: pd.DataFrame) -> None:
    """Barras agrupadas comparando abril de 2020, 2021 e 2022."""
    periodos = [
        ("Abr 2020\n(1ª onda)", "2020-04-01", "2020-04-30"),
        ("Abr 2021\n(pico Gama)", "2021-04-01", "2021-04-30"),
        ("Abr 2022\n(pós-vacina)", "2022-04-01", "2022-04-30"),
    ]
    trabalho, obitos, rotulos = [], [], []
    for rotulo, inicio, fim in periodos:
        recorte = df[(df["date"] >= inicio) & (df["date"] <= fim)]
        trabalho.append(recorte["workplaces_percent_change_from_baseline"].mean())
        obitos.append(recorte["new_deaths_smoothed"].mean())
        rotulos.append(rotulo)

    x = range(len(rotulos))
    largura = 0.36
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(
        [i - largura / 2 for i in x],
        trabalho,
        largura,
        color="#1f4e79",
        label="Média da mobilidade no trabalho (%)",
    )
    ax1.set_ylabel("Mobilidade no trabalho (% vs linha de base)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(rotulos)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax2 = ax1.twinx()
    ax2.bar(
        [i + largura / 2 for i in x],
        obitos,
        largura,
        color="#c0392b",
        label="Média de óbitos diários",
    )
    ax2.set_ylabel("Média de óbitos diários (suavizados)")

    ax1.set_title("Mesmo mês do calendário, realidades diferentes: mobilidade x mortalidade")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "03_comparacao_abril_barras.png", dpi=150)
    plt.close(fig)

    for rotulo, w, d in zip(rotulos, trabalho, obitos):
        print(
            f"Insight das barras [{rotulo.replace(chr(10), ' ')}]: "
            f"trabalho={w:.1f}%, óbitos/dia={d:.0f}"
        )


def grafico_vacinacao_e_obitos(df: pd.DataFrame) -> None:
    """Cobertura vacinal completa x óbitos a partir de janeiro de 2021."""
    v = df[df["date"] >= "2021-01-01"].copy()
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(
        v["date"],
        v["new_deaths_smoothed"],
        color="#c0392b",
        label="Óbitos diários (suavizados)",
    )
    ax1.set_ylabel("Óbitos diários (suavizados)")
    ax1.set_xlabel("Data")

    ax2 = ax1.twinx()
    ax2.plot(
        v["date"],
        v["people_fully_vaccinated_per_hundred"],
        color="#2ecc71",
        label="Totalmente vacinados (%)",
        linewidth=2,
    )
    ax2.set_ylabel("Totalmente vacinados (% da população)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title("Ampliação da vacinação e queda dos óbitos diários (Brasil, 2021–2022)")
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "04_vacinacao_vs_obitos.png", dpi=150)
    plt.close(fig)


def main() -> None:
    unido, _ = carregar_e_limpar()
    imprimir_insights(unido)
    grafico_mobilidade_vs_obitos(unido)
    grafico_restricao_boxplot(unido)
    grafico_comparacao_abril(unido)
    grafico_vacinacao_e_obitos(unido)
    print(f"\nFiguras salvas em {DIR_FIGURAS.resolve()}")


if __name__ == "__main__":
    main()
