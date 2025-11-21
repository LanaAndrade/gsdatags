# 4_train_compare.py
import argparse
import json
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump

from _3_split_pre_proc import load_and_split, make_preprocessor, majority_baseline


class AlgorithmComparison:
    """Plota um gráfico comparando as métricas dos algoritmos."""

    def plot_algorithm_comparison(self, results: Dict[str, Dict[str, float]]):
        names = list(results.keys())
        acc_test = [results[m]["accuracy_test"] for m in names]
        f1_test = [results[m]["f1_test"] for m in names]

        x = np.arange(len(names))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.grid(True, axis="y", alpha=0.3)

        plt.bar(x - width / 2, acc_test, width, label="Accuracy (teste)")
        plt.bar(x + width / 2, f1_test, width, label="F1-macro (teste)")

        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Comparação de algoritmos no conjunto de teste")
        plt.legend()

        plt.tight_layout()
        plt.savefig("model_scores.png", dpi=300, bbox_inches="tight")
        plt.show()


ap = argparse.ArgumentParser()
ap.add_argument(
    "--data",
    required=True,
    help="caminho para survey_results_public.csv",
)
ap.add_argument(
    "--out",
    required=True,
    help="caminho para salvar o melhor modelo (.joblib)",
)
args = ap.parse_args()

print("Carregando dados e fazendo split")
X_train, X_test, y_train, y_test = load_and_split(args.data)
pre = make_preprocessor()

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42,
    ),
    "LinearSVC": LinearSVC(
        C=1.0,
        random_state=42,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=15,
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results: Dict[str, Dict[str, float]] = {}
best_name = None
best_f1_cv = -np.inf
best_pipeline = None

print("Treinando e avaliando modelos")
for name, clf in models.items():
    print(f"\nModelo: {name}")
    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    f1_cv = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    acc_cv = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average="macro")
    prec_test = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_test = recall_score(y_test, y_pred, average="macro", zero_division=0)

    results[name] = {
        "accuracy_cv_mean": float(acc_cv.mean()),
        "f1_cv_mean": float(f1_cv.mean()),
        "accuracy_test": float(acc_test),
        "f1_test": float(f1_test),
        "precision_test": float(prec_test),
        "recall_test": float(rec_test),
    }

    print(
        f"CV: acc={acc_cv.mean():.3f}, f1={f1_cv.mean():.3f} | "
        f"Teste: acc={acc_test:.3f}, f1={f1_test:.3f}, "
        f"prec={prec_test:.3f}, rec={rec_test:.3f}"
    )

    if f1_cv.mean() > best_f1_cv:
        best_f1_cv = f1_cv.mean()
        best_name = name
        best_pipeline = pipe

baseline_class = majority_baseline(y_train)
baseline_acc = (y_test == baseline_class).mean()

print("\nBaseline (maior classe):")
print(f"Classe mais frequente: {baseline_class}")
print(f"Accuracy baseline no teste: {baseline_acc:.3f}")

dump(best_pipeline, args.out)

meta = {
    "best_model": best_name,
    "best_macroF1_cv": float(best_f1_cv),
    "baseline_class": baseline_class,
    "baseline_accuracy_test": float(baseline_acc),
    "metrics_test": results.get(best_name, {}),
    "all_models": results,
}
with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\nModelo salvo em: {args.out}")
print("Resumo do melhor modelo:")
print(json.dumps(meta, indent=2))

comp = AlgorithmComparison()
comp.plot_algorithm_comparison(results)
