# Step 2: train from the versioned dataset, pass the gate, then version the model
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import wandb

GATE = 0.60  # minimum validation recall to save and register a model

run = wandb.init(
    project="churn-tracking-101",
    job_type="train",
    name="train-model",
    config={"C": 1.0, "gate_val_recall": GATE},
)

dataset = run.use_artifact("customers:latest")  # new: this run's input, recorded
df = pd.read_csv(Path(dataset.download()) / "customers.csv")  # new: that exact version

X, y = df[["usage_hours", "support_tickets"]], df["churned"]
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=y_dev)

C = run.config["C"]
pipe = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(C=C)),
])
pipe.fit(X_train, y_train)

recall = recall_score(y_val, pipe.predict(X_val))
run.log({"val_recall": recall})
print(f"C={C:g} | val_recall {recall:.3f} | gate {GATE}")

if recall < GATE:  # new: quality gate. A weak model is never saved or registered
    run.finish(exit_code=1)  # new: the run shows as failed in W&B
    raise SystemExit("quality gate FAILED: model not registered")

test_recall = recall_score(y_test, pipe.predict(X_test))  # new: test, used once
run.log({"test_recall": test_recall})
print(f"quality gate passed | test_recall {test_recall:.3f}")

path = Path("models/churn_pipeline.joblib")
path.parent.mkdir(exist_ok=True)
joblib.dump(pipe, path)

model = wandb.Artifact(name="churn-model", type="model")  # new
model.add_file(str(path))  # new
run.log_artifact(model)  # new: this run's output. W&B now links data > run > model
print("logged artifact: churn-model (type model)")
run.finish()
