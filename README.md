# 101: from a printed number to a traceable model

Tiny Python files, run before the tracking and lineage slides of Sessions 10-11. They build, from zero, the demo on slide `s911-demo-tracking`

A model learns churn from `data/customers.csv`: 300 synthetic customers, two features (`usage_hours`, `support_tickets`), about one in five churns. The split never changes: 70% train, 15% validation, 15% test. The model is the Sessions 9-11 pipeline: fill, scale, logistic regression
Each step adds one small change to the previous step, marked `# new`. One exception, stated in the file: `01_track_a_run` step 5 builds on step 3

| Run before | Folder | The example grows from |
| --- | --- | --- |
| What a run records | `01_track_a_run` | A printed number > a run with its settings > logged metrics > a table and a chart > three runs compared |
| Lineage: data, run, model | `02_artifacts_lineage` | Version the dataset > train from that version, pass the gate, version the model > serve the model with the alias `prod` |
| Tracking in the golden repository | `03_tracking_switch` | One `tracking.py` module, one switch in `config.yaml`: the same pipeline runs with tracking on or off |
| Share the evidence | `04_report` | A report built in the browser; instructor only: the same report from code |

Each folder has a `STEPS.md` (what to run, what to expect, what to say)

W&B means Weights & Biases, the experiment tracking platform used in this course

## Setup, once

1. Create a free account at https://wandb.ai (students and professors can request the free academic plan with the university email)
2. Right-click `101` in VS Code Explorer > **Open in Integrated Terminal**, then:

```text
conda env create -f environment.yml
conda activate tracking-101
wandb login
```

`wandb login` asks for your API key: copy it from https://wandb.ai/authorize and paste it in the terminal. The key is saved on your computer, never in a code file

## No internet in the classroom

Every step in `01_track_a_run` and `03_tracking_switch` also works offline. The setting is saved per folder, so run the command in the terminal of the step folder you are working in, for example `101/01_track_a_run`:

```text
wandb offline
```

Runs are then stored in a `wandb` folder inside that step folder. Go back with `wandb online`, and upload the stored runs with `wandb sync --yes`, both from the same step folder. W&B 0.30 removed `--sync-all`; `wandb sync --yes` uploads the unsynced offline runs without a confirmation prompt. Folder `02_artifacts_lineage` steps 2 and 3 and folder `04_report` need a connection, because they read from W&B

## What this demo does not show

- No Git commit on the runs: this folder is not a Git repository. In the golden repository each run shows its commit
- `03_tracking_switch` skips the test audit and the plots that the golden repository logs
- Logging predictions is not monitoring: nothing here detects drift or checks fairness
- Switching tracking off is a decision you take in `config.yaml`. The pipeline does not detect a W&B outage by itself
