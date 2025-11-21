import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from _3_split_pre_proc import split_multi, years_to_float, map_devtype_to_track

COLUNAS_PRINCIPAIS = [
    "DevType",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "YearsCodePro",
    "EdLevel",
    "Employment",
    "Country",
    "JobSat",
]


def carregar_dataset() -> pd.DataFrame:
    """Carrega o dataset original do StackOverflow."""
    print("carregar_dataset")
    caminho = "../data/survey_results_public.csv"
    return pd.read_csv(caminho)


def mostrar_visao_geral(df: pd.DataFrame):
    """Mostra visão geral de dimensões e colunas principais."""
    cols_existentes = [c for c in COLUNAS_PRINCIPAIS if c in df.columns]
    texto = []
    texto.append("mostrar_visao_geral")
    texto.append(f"Dimensões: {df.shape}")
    texto.append(f"Colunas principais encontradas ({len(cols_existentes)}):")
    for c in cols_existentes:
        texto.append(f"  - {c}")
    if cols_existentes:
        texto.append("\nPrimeiras linhas das colunas principais:")
        texto.append(str(df[cols_existentes].head()))
    print("\n".join(texto))


def analisar_tipos(df: pd.DataFrame):
    """Mostra tipos e porcentagem de missing nas colunas principais."""
    cols_existentes = [c for c in COLUNAS_PRINCIPAIS if c in df.columns]
    tipos = df[cols_existentes].dtypes
    missing = df[cols_existentes].isna().mean()

    tabela = pd.DataFrame({"tipo": tipos, "missing": missing})

    texto = ["analisar_tipos", str(tabela)]
    print("\n".join(texto))


def estatisticas_numericas(df: pd.DataFrame):
    """Mostra estatísticas básicas de colunas numéricas principais."""
    cols_existentes = [c for c in COLUNAS_PRINCIPAIS if c in df.columns]
    num = df[cols_existentes].select_dtypes(include="number")

    if num.empty:
        print("estatisticas_numericas\nNenhuma coluna numérica entre as principais.")
        return

    desc = num.describe().T
    texto = ["estatisticas_numericas", str(desc)]
    print("\n".join(texto))


def resumo_categoricas(df: pd.DataFrame):
    """Mostra contagem básica das colunas categóricas principais."""
    cols_existentes = [c for c in COLUNAS_PRINCIPAIS if c in df.columns]
    cat = df[cols_existentes].select_dtypes(include="object")

    linhas = ["resumo_categoricas"]

    for col in cat.columns:
        valores = cat[col].value_counts(dropna=False)
        linhas.append("\n------------------------------")
        linhas.append(f"Coluna: {col}")
        linhas.append(f"Níveis únicos: {len(valores)}")
        linhas.append("Top 5 valores mais comuns:")
        linhas.append(str(valores.head(5)))
        linhas.append(f"Missing: {cat[col].isna().sum()}")

    print("\n".join(linhas))


def plotar_top10_devtype(df: pd.DataFrame):
    """Plota o top 10 das categorias de DevType."""
    print("plotar_top10_devtype")

    if "DevType" not in df.columns:
        print("Coluna DevType não encontrada no dataset.")
        return

    contagem = df["DevType"].value_counts().head(10)

    plt.figure(figsize=(8, 3))
    contagem.plot(kind="bar")
    plt.title("Top 10 categorias - DevType", fontsize=10)
    plt.ylabel("Frequência", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig("../data/top10_devtype.png", dpi=300)
    plt.show()


def preprocess_career_view(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas para EDA focada em carreiras."""
    print("preprocess_career_view")

    df = df.copy()

    if "DevType" in df.columns:
        df["track"] = df["DevType"].apply(map_devtype_to_track)
        df = df[~df["track"].isna()]
    else:
        df["track"] = "unknown"

    if "LanguageHaveWorkedWith" in df.columns:
        df["langs_list"] = df["LanguageHaveWorkedWith"].apply(split_multi)
    else:
        df["langs_list"] = [[] for _ in range(len(df))]

    if "DatabaseHaveWorkedWith" in df.columns:
        df["dbs_list"] = df["DatabaseHaveWorkedWith"].apply(split_multi)
    else:
        df["dbs_list"] = [[] for _ in range(len(df))]

    if "YearsCodePro" in df.columns:
        df["YearsCodePro_num"] = df["YearsCodePro"].apply(years_to_float)
    elif "YearsCode" in df.columns:
        df["YearsCodePro_num"] = df["YearsCode"].apply(years_to_float)
    else:
        df["YearsCodePro_num"] = np.nan

    df["all_skills"] = df.apply(
        lambda row: (row.get("langs_list") or []) + (row.get("dbs_list") or []),
        axis=1,
    )

    return df


def plot_career_distributions(df: pd.DataFrame):
    """Plota gráficos principais de distribuição por trilha de carreira."""
    print("\nplot_career_distributions")

    if "track" not in df.columns:
        print("Coluna 'track' não encontrada; pulando gráficos de carreira.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        "Análise Exploratória das Carreiras em TI (dados StackOverflow)",
        fontsize=12,
        fontweight="bold",
    )

    career_counts = df["track"].value_counts()
    bars = axes[0, 0].bar(
        career_counts.index,
        career_counts.values,
        color="#3B82F6",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    axes[0, 0].set_title("Distribuição das Trilhas de Carreira", fontweight="bold", fontsize=10)
    axes[0, 0].set_xlabel("Trilha", fontsize=8)
    axes[0, 0].set_ylabel("Número de profissionais", fontsize=8)
    axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0, 0].tick_params(axis="y", labelsize=7)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    for bar, count in zip(bars, career_counts.values):
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{count}\n({count/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    if df["YearsCodePro_num"].notna().any():
        exp_column = "YearsCodePro_num"
        career_exp_data = []
        career_labels = []

        for career in df["track"].unique():
            exp_data = df[df["track"] == career][exp_column].dropna()
            if len(exp_data) > 0:
                career_exp_data.append(exp_data)
                career_labels.append(career)

        if career_exp_data:
            box_plot = axes[0, 1].boxplot(
                career_exp_data, labels=career_labels, patch_artist=True
            )

            for patch in box_plot["boxes"]:
                patch.set_facecolor("#10B981")
                patch.set_alpha(0.7)

            axes[0, 1].set_title(
                "Anos de Experiência por Trilha", fontweight="bold", fontsize=10
            )
            axes[0, 1].set_xlabel("Trilha", fontsize=8)
            axes[0, 1].set_ylabel("Anos de experiência", fontsize=8)
            axes[0, 1].tick_params(axis="x", rotation=45, labelsize=7)
            axes[0, 1].tick_params(axis="y", labelsize=7)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Dados de experiência\nnão disponíveis",
                ha="center",
                va="center",
                fontsize=8,
                transform=axes[0, 1].transAxes,
            )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "Dados de experiência\nnão disponíveis",
            ha="center",
            va="center",
            fontsize=8,
            transform=axes[0, 1].transAxes,
        )

    axes[0, 1].grid(True, alpha=0.3, axis="y")

    all_skills_counter = Counter()
    for skills_list in df.get("all_skills", []):
        if isinstance(skills_list, list):
            all_skills_counter.update(skills_list)

    if all_skills_counter:
        top_skills = [skill for skill, _ in all_skills_counter.most_common(12)]

        skill_heatmap_data = []
        valid_careers = []

        for career in df["track"].unique():
            career_data = df[df["track"] == career]
            skill_freq = []
            for skill in top_skills:
                count = 0
                for skills in career_data["all_skills"]:
                    if isinstance(skills, list) and skill in skills:
                        count += 1
                skill_freq.append(
                    count / len(career_data) * 100 if len(career_data) > 0 else 0
                )
            skill_heatmap_data.append(skill_freq)
            valid_careers.append(career)

        if skill_heatmap_data:
            im = axes[1, 0].imshow(
                skill_heatmap_data,
                cmap="YlOrRd",
                aspect="auto",
                vmin=0,
                vmax=100,
            )

            axes[1, 0].set_xticks(range(len(top_skills)))
            axes[1, 0].set_xticklabels(
                top_skills, rotation=45, ha="right", fontsize=6
            )
            axes[1, 0].set_yticks(range(len(valid_careers)))
            axes[1, 0].set_yticklabels(valid_careers, fontsize=7)

            axes[1, 0].set_title(
                "Top Skills por Trilha (%)", fontweight="bold", fontsize=10
            )
            axes[1, 0].set_xlabel("Skills", fontsize=8)
            axes[1, 0].set_ylabel("Trilha", fontsize=8)

            for i in range(len(valid_careers)):
                for j in range(len(top_skills)):
                    axes[1, 0].text(
                        j,
                        i,
                        f"{skill_heatmap_data[i][j]:.0f}%",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=6,
                    )

            cbar = plt.colorbar(im, ax=axes[1, 0])
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("Frequência (%)", fontsize=8)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Dados de skills\nnão disponíveis",
            ha="center",
            va="center",
            fontsize=8,
            transform=axes[1, 0].transAxes,
        )

    if "EdLevel" in df.columns:
        education_column = "EdLevel"
        education_by_career = (
            pd.crosstab(df["track"], df[education_column], normalize="index") * 100
        )

        bottom_vals = np.zeros(len(education_by_career))
        for education_level in education_by_career.columns:
            values = education_by_career[education_level].values
            axes[1, 1].bar(
                education_by_career.index,
                values,
                bottom=bottom_vals,
                label=education_level,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom_vals += values

        axes[1, 1].set_title(
            "Distribuição Educacional por Trilha (%)", fontweight="bold", fontsize=10
        )
        axes[1, 1].set_xlabel("Trilha", fontsize=8)
        axes[1, 1].set_ylabel("Percentual", fontsize=8)
        axes[1, 1].tick_params(axis="x", rotation=45, labelsize=7)
        axes[1, 1].tick_params(axis="y", labelsize=7)
        leg = axes[1, 1].legend(title="Nível Educacional", fontsize=7)
        leg.get_title().set_fontsize(8)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Dados educacionais\nnão disponíveis",
            ha="center",
            va="center",
            fontsize=8,
            transform=axes[1, 1].transAxes,
        )

    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("../data/career_distributions_real_data.png", dpi=300, bbox_inches="tight")
    plt.show()


df = carregar_dataset()
mostrar_visao_geral(df)
analisar_tipos(df)
estatisticas_numericas(df)
resumo_categoricas(df)
plotar_top10_devtype(df)

df_career = preprocess_career_view(df)
if not df_career.empty:
    plot_career_distributions(df_career)

df.to_csv("../data/eda_output.csv", index=False)
