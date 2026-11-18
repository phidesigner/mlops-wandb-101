# Step 1: version the dataset. One run logs the file as an artifact
import wandb

run = wandb.init(project="churn-tracking-101", job_type="load-data", name="log-dataset")

dataset = wandb.Artifact(name="customers", type="dataset")  # new: a versioned bundle
dataset.add_file("../data/customers.csv")  # new: put the file in the bundle
run.log_artifact(dataset)  # new: upload it as an output of this run
print("logged artifact: customers (type dataset)")
run.finish()
