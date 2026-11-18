# Step 2: start a run and record its settings
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import wandb  # new

df = pd.read_csv("../data/customers.csv")
X, y = df[["usage_hours", "support_tickets"]], df["churned"]
X_dev, X_test, y_dev, y_test = train_test_split(  # 15% test, locked away first
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(  # 70% train, 15% validation
    X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=y_dev)

run = wandb.init(  # new: one run = one recorded execution
    project="churn-tracking-101",
    name="first-run",
    config={"C": 1.0, "features": list(X.columns), "split": "70/15/15"},  # new: settings
)
C = run.config["C"]  # new: read the setting back from the run, one source of truth
pipe = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(C=C)),
])
pipe.fit(X_train, y_train)

accuracy = pipe.score(X_val, y_val)
recall = recall_score(y_val, pipe.predict(X_val))
print(f"C={C:g} | val_accuracy {accuracy:.3f} | val_recall {recall:.3f}")
run.finish()  # new: upload what is pending and close the run
