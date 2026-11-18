# Instructor only: build the report from code, so every release gets the same page
# Needs one extra package: pip install wandb-workspaces
import wandb
import wandb_workspaces.reports.v2 as wr

PROJECT = "churn-tracking-101"
api = wandb.Api()
ENTITY = api.default_entity  # your W&B username or team

# Keep the latest finished run for each C, even after repeating the demo.
names = {"C-0.001", "C-0.05", "C-1.0"}
latest = {}
for run in api.runs(
    f"{ENTITY}/{PROJECT}",
    filters={"group": "compare-C", "state": "finished"},
    order="-created_at",
):
    if run.name in names:
        latest.setdefault(run.name, run.id)
    if len(latest) == len(names):
        break
if set(latest) != names:
    raise RuntimeError("Run 01_track_a_run/step5_compare_runs.py first")
run_ids = list(latest.values())

report = wr.Report(
    project=PROJECT,
    entity=ENTITY,
    title="Churn model: which C do we ship?",
    description="Three runs compared on validation. Test is used once, after the choice",
)
report.blocks = [
    wr.H1("Decision"),
    wr.P("C=1.0 catches the most churners on validation and passes the 0.60 recall gate"),
    wr.H1("Evidence"),
    wr.PanelGrid(
        runsets=[wr.Runset(  # filter the charts and table to these exact three runs
            entity=ENTITY, project=PROJECT, name="compare-C",
            filters=f"ID in {run_ids!r}")],
        panels=[
            wr.BarPlot(title="Validation recall", metrics=["val_recall"]),
            wr.BarPlot(title="Validation accuracy", metrics=["val_accuracy"]),
            wr.RunComparer(diff_only="split"),
        ],
    ),
]
report.save()  # nothing is uploaded until save()
print(f"report saved: https://wandb.ai/{ENTITY}/{PROJECT}/reportlist")
