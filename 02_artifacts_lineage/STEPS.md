# 101 lineage: data, run, model

Run before slides "Every model has a history" and "Promote with an alias"; slide `s911-demo-lineage` shows the result. About 10 min. Needs an internet connection

Right-click `101/02_artifacts_lineage` in Explorer > **Open in Integrated Terminal**, then:

```text
conda activate tracking-101
```

One question all the way: which data and which run produced the model that is serving predictions?

## Step 1. Version the dataset

Change: `wandb.Artifact(name, type)` creates a bundle; `add_file` puts the file in it; `run.log_artifact` uploads it as an output of this run

```text
python step1_log_dataset.py
```

Expect: `logged artifact: customers (type dataset)`

Open the project > **Artifacts** > `customers`. One version, `v0`, with the alias `latest`

Run the same command again. Still `v0`. W&B prints that the artifact already exists with the same content, so no new version is created

Say: "An artifact is a versioned file. W&B compares the content, not the file name. Same content, same version; changed content, `v1`."

## Step 2. Train from that version, pass the gate, version the model

Change: `run.use_artifact("customers:latest")` records the dataset as this run's input; a quality gate on validation recall; `log_artifact` records the model as this run's output

```text
python step2_train_from_artifact.py
```

Expect:

```text
C=1 | val_recall 0.778 | gate 0.6
quality gate passed | test_recall 0.778
logged artifact: churn-model (type model)
```

Open **Artifacts** > `churn-model` > `v0` > **Lineage**. The graph reads: `customers:v0` > run `train-model` > `churn-model:v0`

Now show the gate. In `step2_train_from_artifact.py` change `"C": 1.0` to `"C": 0.001` and run again

Expect: `C=0.001 | val_recall 0.000 | gate 0.6`, then `quality gate FAILED: model not registered`. In W&B the run shows as failed and `churn-model` still has one version

Set `"C"` back to `1.0`

Say: "use_artifact is the input, log_artifact is the output. Two lines, and W&B draws the history for us. The gate runs before saving: a weak model never becomes a version, so nobody can deploy it by mistake. And the test split is used once, after the gate, only to report."

## Step 3. Promote with an alias, serve by alias

First promote, in the browser: **Artifacts** > `churn-model` > `v0` > next to **Aliases** click **+** and add `prod`

Change: a separate script, with no training code, asks for `churn-model:prod`

```text
python step3_serve_prod.py
```

Expect:

```text
serving: churn-model:prod
 usage_hours  support_tickets  prediction
         3.0                4           1
        25.0                0           0
         NaN                2           0
```

Open **Artifacts** > `churn-model`: the alias `prod` sits on `v0`, so that is the version being served. Open **Lineage**: the run `serve-prod` now hangs from `churn-model:v0`

Run it before adding the alias to show the failure: W&B answers that `churn-model:prod` does not exist. Nothing is served until a person promotes a version

Say: "The serving code never names a version or a file path. It asks for prod. Promotion is moving the alias to a newer version. Rollback is moving it back. No code change, no redeploy of the code. This is how the API in the golden repository finds its model when MODEL_SOURCE=wandb: it reads the alias once, at startup, so after moving the alias you restart the API."

## Link to the slides and the golden repository

| Here | On the slides and in the golden repository |
| --- | --- |
| Step 1, dataset artifact | `log_file_artifact()` with `artifact_type="dataset"`, switched on by `wandb.log_processed_data` in `config.yaml`. Difference: the golden repository logs the processed data as an output of the training run, so its graph reads run > data and run > model |
| Step 2, quality gate | `evaluation.quality_gate` in `config.yaml`; `check_quality_gate()` runs in `src/main.py` before the model is saved |
| Step 2, model artifact | `log_file_artifact()` with `artifact_type="model"`, name from `wandb.model_artifact_name` |
| Step 3, alias `prod` | `_load_model()` in `src/api.py`: `MODEL_SOURCE=wandb`, `WANDB_MODEL_ALIAS=prod`. Difference: the API downloads with `wandb.Api()`, without a run, so no consumer appears in the lineage graph |

**Remember:** version the data and the model, let the run connect them, and serve by alias so promotion and rollback need no code change
