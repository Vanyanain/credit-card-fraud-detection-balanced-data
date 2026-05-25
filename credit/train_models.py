from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib


RANDOM_STATE = 42
DATA_PATH = BASE_DIR / "creditcard_2023.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=["Class"])
    if "id" in features.columns:
        features = features.drop(columns=["id"])
    target = df["Class"]
    return features, target


def build_models() -> dict[str, Pipeline]:
    numeric_features = list(pd.read_csv(DATA_PATH, nrows=1).drop(columns=["Class"]).columns)
    if "id" in numeric_features:
        numeric_features.remove("id")

    scaler = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scale", scaler),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10, min_samples_leaf=10)),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("scale", scaler),
                ("model", LinearSVC(random_state=RANDOM_STATE, dual="auto", max_iter=5000)),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "lightgbm": Pipeline(
            steps=[
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        num_leaves=31,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                )
            ]
        ),
        "neural_network": Pipeline(
            steps=[
                ("scale", clone(scaler)),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        batch_size=512,
                        max_iter=10,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def get_scores(model: Pipeline, features: pd.DataFrame) -> tuple[pd.Series, str]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1], "probability"
    if hasattr(model, "decision_function"):
        return model.decision_function(features), "decision"
    raise ValueError("Model does not support probability or decision scores.")


def plot_confusion_matrix(cm, title: str, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc: float, title: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_curve(recall, precision, pr_auc: float, title: str, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def maybe_downsample(x_train, y_train, sample_size: int | None):
    if sample_size is None or sample_size >= len(x_train):
        return x_train, y_train

    sampled_x, _, sampled_y, _ = train_test_split(
        x_train,
        y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return sampled_x, sampled_y


def evaluate_model(name: str, model: Pipeline, x_train, x_test, y_train, y_test, sample_size: int | None = None) -> dict[str, float]:
    fit_x, fit_y = maybe_downsample(x_train, y_train, sample_size)
    model.fit(fit_x, fit_y)
    predictions = model.predict(x_test)
    scores, score_type = get_scores(model, x_test)

    roc_auc = roc_auc_score(y_test, scores)
    fpr, tpr, _ = roc_curve(y_test, scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(recall_curve, precision_curve)
    cm = confusion_matrix(y_test, predictions)

    plot_roc_curve(fpr, tpr, roc_auc, f"{name} ROC Curve", OUTPUT_DIR / f"{name}_roc_curve.png")
    plot_pr_curve(recall_curve, precision_curve, pr_auc, f"{name} Precision-Recall Curve", OUTPUT_DIR / f"{name}_pr_curve.png")
    plot_confusion_matrix(cm, f"{name} Confusion Matrix", OUTPUT_DIR / f"{name}_confusion_matrix.png")

    print(f"Saving trained model: {name}...")
    joblib.dump(model, OUTPUT_DIR / f"{name}_model.joblib")

    return {
        "model": name,
        "score_type": score_type,
        "train_rows_used": len(fit_x),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main() -> None:
    (BASE_DIR / ".matplotlib").mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results = []
    training_rows = {
        "logistic_regression": None,
        "decision_tree": None,
        "svm": None,
        "xgboost": None,
        "lightgbm": None,
        "neural_network": 50000,
    }

    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        results.append(
            evaluate_model(
                model_name,
                model,
                x_train,
                x_test,
                y_train,
                y_test,
                sample_size=training_rows[model_name],
            )
        )

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print("\nMetrics summary:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved metrics and plots in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
