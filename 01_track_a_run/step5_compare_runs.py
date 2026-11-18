# Step 5: three settings, three runs, one group. Compare them in the workspace
# Built on step 3: the table and chart of step 4 are left out to keep the loop short
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import wandb

df = pd.read_csv("../data/customers.csv")
X, y = df[["usage_hours", "support_tickets"]], df["churned"]
X_dev, X_test, y_dev, y_test = train_test_split(  # 15% test, locked away first
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(  # 70% train, 15% validation
    X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=y_dev)

for C in [0.001, 0.05, 1.0]:  # new: the experiment, one run per setting
    run = wandb.init(
        project="churn-tracking-101",
        group="compare-C",  # new: keeps the three runs together
        name=f"C-{C}",  # new
        config={"C": C, "features": list(X.columns), "split": "70/15/15"},
    )
    pipe = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C)),
    ])
    pipe.fit(X_train, y_train)

    accuracy = pipe.score(X_val, y_val)
    recall = recall_score(y_val, pipe.predict(X_val))
    run.log({"val_accuracy": accuracy, "val_recall": recall,
             "rows_train": len(X_train)})
    run.summary["rows_train"] = len(X_train)
    print(f"C={C:g} | val_accuracy {accuracy:.3f} | val_recall {recall:.3f}")
    run.finish()  # new: close this run before the next one starts
