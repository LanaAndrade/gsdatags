import re
import numpy as np
import pandas as pd
from typing import Tuple
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from _2_eng_at import DEVTYPE_TO_TRACK, TRACK_PRIORITY

PRIORIDADE = TRACK_PRIORITY


def split_multi(text: str):
    """Divide campos com ';' em lista, tratando nulos."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [t.strip() for t in text.split(";") if t.strip()]


def years_to_float(x):
    """Converte descrições de anos de experiência em float."""
    if not isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return None

    x_low = x.strip().lower()

    if "less than" in x_low:
        return 0.5
    if "more than" in x_low:
        m = re.search(r"(\d+)", x_low)
        return float(m.group(1)) if m else None

    try:
        return float(x_low)
    except Exception:
        return None


class MultiHotter(BaseEstimator, TransformerMixin):
    """Transforma lista de skills em vetor multi-hot com top-N itens."""

    def __init__(self, top_n: int = 12):
        self.top_n = top_n
        self.vocab_ = None
        self.mlb_ = None
        self.colname_ = None

    def _as_series(self, X):
        """Garante que a entrada seja tratada como Series."""
        if hasattr(X, "ndim") and getattr(X, "ndim", 2) == 2:
            if hasattr(X, "columns") and X.shape[1] == 1:
                return X.iloc[:, 0]
        if hasattr(X, "to_numpy") and not hasattr(X, "columns"):
            return X
        return pd.Series(
            [row[0] if isinstance(row, (list, tuple)) else row for row in X]
        )

    def fit(self, X, y=None):
        """Aprende o vocabulário de skills mais frequentes."""
        from sklearn.preprocessing import MultiLabelBinarizer

        s = self._as_series(X)
        counter = Counter()
        for lst in s:
            lst = lst or []
            counter.update([str(z).strip() for z in lst])

        most = [w for w, _ in counter.most_common(self.top_n)]
        self.vocab_ = most
        self.mlb_ = MultiLabelBinarizer(classes=self.vocab_)
        self.mlb_.fit([self.vocab_])
        return self

    def transform(self, X):
        """Transforma listas de skills em matriz multi-hot."""
        s = self._as_series(X)
        rows = []
        for lst in s:
            lst = lst or []
            rows.append(
                [
                    str(z).strip()
                    for z in lst
                    if str(z).strip() in (self.vocab_ or [])
                ]
            )
        return self.mlb_.transform(rows)

    def get_feature_names_out(self, input_features=None):
        """Retorna nomes das colunas geradas."""
        return [f"{self.colname_}__{w}" for w in (self.vocab_ or [])]


def map_devtype_to_track(devtype_cell: str) -> str:
    """Mapeia DevType textual para trilha de carreira."""
    types = split_multi(devtype_cell)
    tracks = []
    for t in types:
        label = DEVTYPE_TO_TRACK.get(t)
        if label:
            tracks.append(label)

    for p in PRIORIDADE:
        if p in tracks:
            return p
    return tracks[0] if tracks else None


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Cria X e y a partir do CSV bruto da pesquisa."""
    cols = [
        "DevType",
        "LanguageHaveWorkedWith",
        "DatabaseHaveWorkedWith",
        "YearsCodePro",
        "YearsCode",
        "EdLevel",
        "Employment",
    ]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df["track"] = df["DevType"].apply(map_devtype_to_track)
    df = df[~df["track"].isna()]

    df["langs_list"] = df["LanguageHaveWorkedWith"].apply(split_multi)
    df["dbs_list"] = df["DatabaseHaveWorkedWith"].apply(split_multi)

    if df["YearsCodePro"].notna().any():
        year_col = "YearsCodePro"
    else:
        year_col = "YearsCode"
    df["YearsCodePro_num"] = df[year_col].apply(years_to_float)

    X = df[["langs_list", "dbs_list", "YearsCodePro_num", "EdLevel", "Employment"]].copy()
    y = df["track"].astype(str)
    return X, y


def make_preprocessor() -> ColumnTransformer:
    """Cria o pré-processador de colunas para o pipeline."""
    langs_pipe = Pipeline([
        ("mh", MultiHotter(top_n=15)),
    ])
    langs_pipe.named_steps["mh"].colname_ = "langs"

    dbs_pipe = Pipeline([
        ("mh", MultiHotter(top_n=8)),
    ])
    dbs_pipe.named_steps["mh"].colname_ = "dbs"

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("sc", StandardScaler()),
    ])

    pre = ColumnTransformer(
        [
            ("langs", langs_pipe, ["langs_list"]),
            ("dbs", dbs_pipe, ["dbs_list"]),
            ("cat", cat_pipe, ["EdLevel", "Employment"]),
            ("num", num_pipe, ["YearsCodePro_num"]),
        ],
        remainder="drop",
    )
    return pre


def majority_baseline(y):
    """Retorna a classe mais frequente como baseline simples."""
    return Counter(y).most_common(1)[0][0]


def load_and_split(data_path: str):
    """Carrega o CSV, monta X e y e faz o split treino/teste."""
    df = pd.read_csv(data_path)
    X, y = build_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
