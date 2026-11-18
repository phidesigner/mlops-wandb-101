# Step 3: a separate script loads the model that carries the alias "prod"
from pathlib import Path

import joblib
import pandas as pd

import wandb

run = wandb.init(project="churn-tracking-101", job_type="inference", name="serve-prod")

model = run.use_artifact("churn-model:prod")  # new: by alias, never by file path
pipe = joblib.load(Path(model.download()) / "churn_pipeline.joblib")
print("serving:", model.name)

new_customers = pd.DataFrame(
    {"usage_hours": [3.0, 25.0, None], "support_tickets": [4, 0, 2]})
new_customers["prediction"] = pipe.predict(new_customers)
print(new_customers.to_string(index=False))
run.finish()
