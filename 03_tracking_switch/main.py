# main.py: the same pipeline, tracking on or off from config.yaml
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tracking

cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
run = tracking.start_run(cfg)  # None when wandb.enabled is false

try:
    df = pd.read_csv("../data/customers.csv")
    X, y = df[["usage_hours", "support_tickets"]], df["churned"]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=y_dev)

    pipe = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=cfg["model"]["C"])),
    ])
    pipe.fit(X_train, y_train)

    recall = recall_score(y_val, pipe.predict(X_val))
    tracking.log_metrics(run, {"val_recall": recall})
    print(f"val_recall {recall:.3f} | gate {cfg['quality_gate']['min_value']}")

    if recall < cfg["quality_gate"]["min_value"]:
        raise ValueError("quality gate FAILED: model not saved, not registered")

    path = Path("models/churn_pipeline.joblib")
    path.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, path)
    tracking.log_file_artifact(run, "churn-model", "model", path)
    print("saved:", path.as_posix())
except Exception:
    tracking.finish_run(run, exit_code=1)  # the failed run stays visible in W&B
    raise

tracking.finish_run(run)
