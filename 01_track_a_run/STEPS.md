# 101 tracking: what a run records

Run before slides "What a run records" and "W&B in five calls"; slide `s911-demo-tracking` shows the result. About 10 min

Right-click `101/01_track_a_run` in Explorer > **Open in Integrated Terminal**, then:

```text
conda activate tracking-101
```

One question all the way: which value of `C` do we ship? `C` controls how simple the logistic regression is.
Each step file is the previous one plus one small change, marked `# new`. Open it, point to the change, run it

## Step 1. No tracking: a number on the screen

Change: none yet. Train on 70%, score on the 15% validation split, print

```text
python step1_no_tracking.py
```

Expect: `C=1 | val_accuracy 0.911 | val_recall 0.778`

Say: "Close the terminal and this number is gone. Next week nobody knows which setting, which data or which code produced 0.911. That is the problem tracking solves."

## Step 2. Start a run and record its settings

Change: `wandb.init(project, name, config)` starts a run; `run.config["C"]` reads the setting back; `run.finish()` closes it

```text
python step2_init_config.py
```

Expect: the same line, plus W&B lines with a link `View run first-run at: https://wandb.ai/...`

Open the link. Show the **Overview** tab: Config holds `C`, `features`, `split`. Show what W&B recorded without being asked: start time, duration, Python version, operating system and the command. In a Git repository, such as the golden repository, it also records the commit; this demo folder is not one

Say: "A run is one recorded execution. We wrote three lines. W&B stored the settings we gave it and the context we did not think of."

## Step 3. Log the metrics

Change: `run.log({...})` stores the metrics by name; `run.summary[...]` stores one final fact

```text
python step3_log_metrics.py
```

Expect: the same printed line. In W&B, the run `with-metrics` shows `val_accuracy` and `val_recall` in **Overview > Summary**

Say: "config is what we decided before the run. log is what we measured during the run. summary is what we want to find later without opening anything."

## Step 4. Log evidence you can open

Change: a `wandb.Table` with the 45 validation rows and their predictions; `wandb.plot.confusion_matrix`

```text
python step4_table_and_plot.py
```

Expect: `logged: val_predictions table, 45 rows; confusion_matrix`

Open the run `with-evidence`. In the workspace, sort the table by `actual`, then filter `actual = 1` and `predicted = 0`: these are the churners we miss

Say: "A metric says how good. A table says where it fails. The golden repository logs the same things: metrics, a predictions table, curves and a confusion matrix."

## Step 5. Three settings, three runs

Change: step 3 inside a loop over `C` (the table and chart of step 4 are left out to keep the loop short); `group="compare-C"` keeps the runs together; one `finish()` per run

```text
python step5_compare_runs.py
```

Expect:

```text
C=0.001 | val_accuracy 0.800 | val_recall 0.000
C=0.05 | val_accuracy 0.889 | val_recall 0.444
C=1 | val_accuracy 0.911 | val_recall 0.778
```

Open the project workspace. In the runs table, add the columns `C`, `val_accuracy` and `val_recall`, then sort by `val_recall`. Group by `Group` to fold the three runs together

Say: "Look at C=0.001: 80% accuracy and it catches zero churners. It predicts 'stay' for everyone, and four in five customers do stay. Accuracy alone would have shipped a useless model. Now the comparison is a table everyone on the team can open, not a memory."

## Link to the slides and the golden repository

| Here | On the slides and in the golden repository |
| --- | --- |
| Step 2, `wandb.init(config=...)` | `start_run()` in `src/tracking.py`: the whole `config.yaml` is stored on the run |
| Step 3, `run.log` | `log_metrics()` in `src/tracking.py`, called by `src/main.py` after validation and after the test audit |
| Step 4, table and chart | `log_table()` and `log_classification_plots()` in `src/tracking.py` |
| Step 5, `group` and `name` | `wandb.group`, `wandb.name` and `wandb.tags` in `config.yaml` |

**Remember:** config records the settings, log records the measurements, and the run keeps them together with the code version
