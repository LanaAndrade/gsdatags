import subprocess
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from _3_split_pre_proc import load_and_split
from _5_model_anls import ModelAnalysis
from _6_career_rec import print_recommendation_example

DATA_PATH = BASE_DIR / "data" / "survey_results_public.csv"
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.joblib"

print("Rodandoo...")


def run_step(script_name: str, *extra_args: str):
    print(f"Rodando {script_name}")
    subprocess.run(
        ["python", script_name, *[str(a) for a in extra_args]],
        check=True,
        cwd=SRC_DIR,
    )


run_step("_1_eda.py")
run_step("_2_eng_at.py", "--data", DATA_PATH)
run_step("_4_train_compare.py", "--data", DATA_PATH, "--out", MODEL_PATH)

print("Análises detalhadas do modelo")

X_train, X_test, y_train, y_test = load_and_split(str(DATA_PATH))

analysis = ModelAnalysis(
    model_path=str(MODEL_PATH),
    data_path=str(DATA_PATH),
)

analysis.run_all(X_train, y_train, X_test, y_test)

print("Gerando recomendação de carreira")

print_recommendation_example(str(MODEL_PATH))

print("The end")
