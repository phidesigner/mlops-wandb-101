# Step 1: train, print two numbers. Nothing is kept when the script ends
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/customers.csv")
X, y = df[["usage_hours", "support_tickets"]], df["churned"]
X_dev, X_test, y_dev, y_test = train_test_split(  # 15% test, locked away first
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(  # 70% train, 15% validation
    X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=y_dev)

C = 1.0  # the setting we will play with: lower C means a simpler model
pipe = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(C=C)),
])
pipe.fit(X_train, y_train)

accuracy = pipe.score(X_val, y_val)
recall = recall_score(y_val, pipe.predict(X_val))  # share of real churners we catch
print(f"C={C:g} | val_accuracy {accuracy:.3f} | val_recall {recall:.3f}")
