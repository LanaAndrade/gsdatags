import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import joblib


class ModelAnalysis:
    """Faz análise gráfica do modelo treinado."""

    def __init__(self, model_path: str = "model.joblib", data_path: str = None):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.COLORS = {}
        self.setup_plot_style()

    def setup_plot_style(self):
        """Configura o estilo dos gráficos."""
        plt.rcParams.update({
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
        })

        self.COLORS = {
            "backend": "#3B82F6",
            "frontend": "#EF4444",
            "data_ml": "#10B981",
            "devops_sre": "#8B5CF6",
            "mobile": "#F59E0B",
            "qa": "#6B7280",
            "ux": "#EC4899",
            "train": "#3B82F6",
            "test": "#EF4444",
            "heatmap": "coolwarm",
        }

    def load_model_and_data(self):
        """Carrega o modelo salvo para análise."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Modelo carregado com sucesso de: {self.model_path}")

            if hasattr(self.model, "named_steps"):
                print("Modelo é um pipeline sklearn")

            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False

    def plot_feature_importance(self, top_n: int = 15):
        """Plota importância das principais features."""
        print("\nGerando gráfico de importância das features")

        try:
            if hasattr(self.model.named_steps["clf"], "feature_importances_"):
                importances = self.model.named_steps["clf"].feature_importances_
                feature_names = self.model.named_steps["pre"].get_feature_names_out()

                cleaned_feature_names = []
                for name in feature_names:
                    if "_" in name:
                        cleaned_name = name.split("_")[-1]
                        if cleaned_name == "lavaScript":
                            cleaned_name = "JavaScript"
                        cleaned_feature_names.append(cleaned_name)
                    else:
                        cleaned_feature_names.append(name)

                fi_df = pd.DataFrame(
                    {"feature": cleaned_feature_names, "importance": importances}
                ).sort_values("importance", ascending=False).head(top_n)

                plt.figure(figsize=(10, 6))
                bars = plt.barh(
                    fi_df["feature"],
                    fi_df["importance"],
                    color=plt.cm.Blues(np.linspace(0.4, 1, len(fi_df))),
                )

                plt.xlabel("Importância")
                plt.title(f"Top {top_n} Features Mais Importantes (Random Forest)")
                plt.gca().invert_yaxis()

                for bar, importance in zip(bars, fi_df["importance"]):
                    plt.text(
                        bar.get_width() + 0.001,
                        bar.get_y() + bar.get_height() / 2,
                        f"{importance:.4f}",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

                plt.tight_layout()
                plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
                plt.show()

                print("Gráfico de importância salvo como feature_importance.png")
            else:
                print("Modelo não possui atributo feature_importances_")
        except Exception as e:
            print(f"Não foi possível gerar importância de features: {e}")

    def plot_learning_curves(self, X_train, y_train, cv=5):
        """Plota curvas de aprendizado para treino e validação."""
        print("\nGerando curvas de aprendizado")

        try:
            train_sizes, train_scores, test_scores = learning_curve(
                self.model,
                X_train,
                y_train,
                cv=cv,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring="f1_macro",
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(8, 5))
            plt.grid(True, alpha=0.3)

            plt.fill_between(
                train_sizes,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1,
                color=self.COLORS["train"],
            )
            plt.fill_between(
                train_sizes,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
                color=self.COLORS["test"],
            )

            plt.plot(
                train_sizes,
                train_scores_mean,
                "o-",
                color=self.COLORS["train"],
                label="Score de treino",
                linewidth=2,
            )
            plt.plot(
                train_sizes,
                test_scores_mean,
                "o-",
                color=self.COLORS["test"],
                label="Score de validação",
                linewidth=2,
            )

            plt.xlabel("Tamanho do conjunto de treino")
            plt.ylabel("Score F1 (macro)")
            plt.title("Curvas de Aprendizado")
            plt.legend(loc="best")

            plt.tight_layout()
            plt.savefig("learning_curves.png", dpi=300, bbox_inches="tight")
            plt.show()

            print("Curvas de aprendizado salvas como learning_curves.png")
        except Exception as e:
            print(f"Não foi possível gerar curvas de aprendizado: {e}")

    def plot_class_performance(self, X_test, y_test):
        """Plota métricas por classe (precisão, recall e F1)."""
        print("\nGerando desempenho por classe")

        try:
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            classes = []
            precision = []
            recall = []
            f1 = []

            for class_name in self.model.classes_:
                if class_name in report:
                    classes.append(class_name)
                    precision.append(report[class_name]["precision"])
                    recall.append(report[class_name]["recall"])
                    f1.append(report[class_name]["f1-score"])

            metrics_df = pd.DataFrame(
                {
                    "Classe": classes,
                    "Precisão": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                }
            )

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))

            metrics = ["Precisão", "Recall", "F1-Score"]
            colors = [self.COLORS.get(cls, "#6B7280") for cls in classes]

            for idx, metric in enumerate(metrics):
                bars = axes[idx].bar(
                    metrics_df["Classe"],
                    metrics_df[metric],
                    color=colors,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )
                axes[idx].set_title(f"{metric} por Classe")
                axes[idx].set_ylabel(metric)
                axes[idx].tick_params(axis="x", rotation=45)
                axes[idx].set_ylim(0, 1)
                axes[idx].grid(True, alpha=0.3, axis="y")

                for bar, value in zip(bars, metrics_df[metric]):
                    height = bar.get_height()
                    axes[idx].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            fig.suptitle(
                "Desempenho do Modelo por Tipo de Carreira",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig("class_performance.png", dpi=300, bbox_inches="tight")
            plt.show()

            print("Desempenho por classe salvo como class_performance.png")
        except Exception as e:
            print(f"Não foi possível gerar desempenho por classe: {e}")

    def plot_error_analysis(self, X_test, y_test):
        """Plota um resumo da distribuição de acertos e erros."""
        print("\nGerando análise de erros")

        try:
            y_pred = self.model.predict(X_test)

            correct = y_pred == y_test
            error_mask = y_pred != y_test

            error_types = {}
            for true_class in self.model.classes_:
                for pred_class in self.model.classes_:
                    if true_class != pred_class:
                        count = ((y_test == true_class) & (y_pred == pred_class)).sum()
                        if count > 0:
                            error_types[f"{true_class}→{pred_class}"] = count

            top_errors = dict(
                sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            correct_count = correct.sum()
            error_count = error_mask.sum()

            ax1.pie(
                [correct_count, error_count],
                labels=["Acertos", "Erros"],
                autopct="%1.1f%%",
                colors=["#10B981", "#EF4444"],
                startangle=90,
            )
            ax1.set_title("Distribuição de Acertos vs Erros")

            if top_errors:
                bars = ax2.barh(
                    list(top_errors.keys()),
                    list(top_errors.values()),
                    color="#F59E0B",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax2.set_title("Top 10 Erros Mais Comuns")
                ax2.set_xlabel("Número de Ocorrências")

                for bar, count in zip(bars, top_errors.values()):
                    ax2.text(
                        bar.get_width() + 0.1,
                        bar.get_y() + bar.get_height() / 2,
                        f"{count}",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Poucos erros para analisar",
                    ha="center",
                    va="center",
                )

            fig.suptitle(
                "Análise de Erros do Modelo", fontsize=12, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig("error_analysis.png", dpi=300, bbox_inches="tight")
            plt.show()

            print("Análise de erros salva como error_analysis.png")
        except Exception as e:
            print(f"Não foi possível gerar análise de erros: {e}")

    def run_all(self, X_train, y_train, X_test, y_test):
        """Roda as principais análises do modelo em sequência."""
        if not self.load_model_and_data():
            return
        self.plot_feature_importance(top_n=15)
        self.plot_learning_curves(X_train, y_train, cv=5)
        self.plot_class_performance(X_test, y_test)
        self.plot_error_analysis(X_test, y_test)
