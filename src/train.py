import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Bonus 2: dai nguong quyet dinh duoc quet de tim diem F1 cao nhat.
THRESHOLD_GRID = np.arange(0.10, 0.901, 0.05)


def scan_threshold(y_true, probs):
    """
    Quet nguong quyet dinh tren THRESHOLD_GRID va tra ve nguong cho F1 cao nhat.

    model.predict() mac dinh gan nhan 1 khi xac suat > 0.5. Voi du lieu mat can
    bang lop, 0.5 hiem khi la nguong toi uu: ha nguong xuong giup bat duoc nhieu
    truong hop thuoc lop thieu so hon, doi lai precision giam.

    Tra ve:
        (best_threshold, best_f1)
    """
    best_threshold, best_f1 = 0.5, -1.0

    for t in THRESHOLD_GRID:
        t = round(float(t), 2)
        f1_t = float(f1_score(y_true, (probs >= t).astype(int)))
        if f1_t > best_f1:
            best_threshold, best_f1 = t, f1_t

    return best_threshold, best_f1


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # Khoi tao va huan luyen mo hinh
        # random_state=42 de dam bao tinh tai tao giua cac lan chay
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Du doan tren tap holdout va tinh chi so
        # f1_score o day tinh cho LOP DUONG (target = 1), khong dung average:
        # average="weighted" hay "macro" se bi lop da so keo len cao va
        # lam mat y nghia cua nguong 0.65.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: quet nguong quyet dinh thay vi chi dung mac dinh 0.5
        probs = model.predict_proba(X_eval)[:, 1]
        best_threshold, best_f1 = scan_threshold(y_eval, probs)

        # Ghi nhan chi so va mo hinh vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1_at_best_threshold", best_f1)
        mlflow.sklearn.log_model(model, "model")

        # In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(
            f"Nguong tot nhat: {best_threshold:.2f} -> F1 {best_f1:.4f} "
            f"(nguong mac dinh 0.50 -> F1 {f1:.4f}, chenh {best_f1 - f1:+.4f})"
        )

        # Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "best_threshold": best_threshold,
                    "f1_at_best_threshold": best_f1,
                },
                f,
            )

        # Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
