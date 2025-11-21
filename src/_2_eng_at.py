from typing import List, Dict, Optional
import re
import pandas as pd
import argparse

DEVTYPE_TO_TRACK: Dict[str, str] = {
    "Back-end developer": "backend",
    "Front-end developer": "frontend",
    "Full-stack developer": "backend",
    "Mobile developer": "mobile",
    "Data scientist or machine learning specialist": "data_ml",
    "Data or business analyst": "data_ml",
    "DevOps specialist": "devops_sre",
    "Site reliability engineer": "devops_sre",
    "Database administrator": "data_ml",
    "Cloud infrastructure engineer": "devops_sre",
    "QA or test developer": "qa",
    "Product designer or UX designer": "ux",
    "Game or graphics developer": "frontend",
}

TRACK_PRIORITY = ["devops_sre", "data_ml", "backend", "frontend", "mobile", "qa", "ux"]


def map_devtype_to_track(devtype: str) -> Optional[str]:
    """Mapeia um texto de DevType para uma trilha de carreira."""
    if not isinstance(devtype, str) or not devtype.strip():
        return None

    roles = [t.strip() for t in devtype.split(";") if t.strip()]
    tracks: List[str] = []

    for role in roles:
        track = DEVTYPE_TO_TRACK.get(role)
        if track:
            tracks.append(track)

    if not tracks:
        return None

    for t in TRACK_PRIORITY:
        if t in tracks:
            return t

    return tracks[0]


def split_multi(text: str) -> List[str]:
    """Divide campos com ';' em uma lista limpa de strings."""
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


def preprocess_survey_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas derivadas usadas pelo pipeline."""
    df = df.copy()

    if "DevType" in df.columns:
        df["track"] = df["DevType"].apply(map_devtype_to_track)

    if "LanguageHaveWorkedWith" in df.columns:
        df["langs_list"] = df["LanguageHaveWorkedWith"].apply(split_multi)

    if "DatabaseHaveWorkedWith" in df.columns:
        df["dbs_list"] = df["DatabaseHaveWorkedWith"].apply(split_multi)

    if "YearsCodePro" in df.columns:
        df["YearsCodePro_num"] = df["YearsCodePro"].apply(years_to_float)

    df["all_skills"] = df.apply(
        lambda row: (row.get("langs_list") or []) + (row.get("dbs_list") or []),
        axis=1,
    )

    return df


def main(data_path: str, out_path: str = "../data/survey_processed.csv") -> None:
    """Roda a engenharia de atributos e salva o CSV processado."""
    print("feature_engineering: carregando dataset...")
    df = pd.read_csv(data_path)
    print("shape original:", df.shape)

    df_proc = preprocess_survey_df(df)
    print("shape processado:", df_proc.shape)

    df_proc.to_csv(out_path, index=False)
    print(f"dataset processado salvo em: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="caminho para survey_results_public.csv",
    )
    parser.add_argument(
        "--out",
        default="../data/survey_processed.csv",
        help="onde salvar o CSV processado",
    )
    args = parser.parse_args()
    main(args.data, args.out)
