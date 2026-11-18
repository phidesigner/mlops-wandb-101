# 101 tracking switch: the pipeline never depends on W&B

Run before slide "Tracking in the golden repository". About 5 min

Right-click `101/03_tracking_switch` in Explorer > **Open in Integrated Terminal**, then:

```text
conda activate tracking-101
```

Three files: `config.yaml` (the settings), `tracking.py` (every W&B call), `main.py` (the pipeline). Open `tracking.py` first: every function starts with `if run is None: return`

## Step 1. Tracking on

Change: none. `wandb.enabled: true` in `config.yaml`

```text
python main.py
```

Expect: `val_recall 0.778 | gate 0.6`, then `saved: models/churn_pipeline.joblib`, plus the W&B run link

Open the run `pipeline-run`. **Overview > Config** holds the whole `config.yaml`: model, quality gate and W&B settings

Say: "main.py has no `import wandb`. It calls four helpers. The whole config file is stored on the run, so the settings that produced this model can never be lost."

## Step 2. Tracking off

Change: in `config.yaml` set `wandb.enabled: false`

```text
python main.py
```

Expect: `tracking off: continuing without W&B`, then the same two lines. No W&B output, no new run

Say: "Same pipeline, same model, no account needed. Tracking is optional infrastructure: switch it off to run without W&B, for example when the service is down or in the automatic checks on GitHub. The switch is ours; the pipeline does not detect an outage by itself. The golden repository does the same with one more switch, WANDB_MODE=disabled, for its tests."

## Step 3. A failed gate is evidence too

Change: set `wandb.enabled: true` again and `model.C: 0.001`

```text
python main.py
```

Expect: `val_recall 0.000 | gate 0.6`, then a traceback ending in `ValueError: quality gate FAILED: model not saved, not registered`

In W&B the run shows as **failed**, with its config and its `val_recall`

Set `model.C` back to `1.0`

Say: "The except block calls finish_run with exit_code=1 before raising again. The failed attempt stays on record: what we tried, with which settings, and why it stopped."

## Link to the slides and the golden repository

| Here | In the golden repository |
| --- | --- |
| `tracking.py`, `run=None` does nothing | `src/tracking.py`, the same contract with tables and plots added |
| `wandb` section in `config.yaml` | `wandb` section in `config.yaml`: project, name, group, tags and the switches for each kind of evidence |
| `finish_run(run, exit_code=1)` | The `except` block at the end of `src/main.py` |
| Tracking off | `wandb.enabled: false`, or `WANDB_MODE=disabled` in `tests/conftest.py` and `.github/workflows/ci.yml` |

**Remember:** keep every tracking call in one module with one switch, so the pipeline never fails because tracking is off
