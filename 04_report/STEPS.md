# 101 report: share the evidence

Run after the demo slide "Demo: track runs, gate the model". About 5 min. Needs an internet connection and the three runs from `01_track_a_run` step 5

## Step 1. Build the report in the browser

1. Open the project `churn-tracking-101` in W&B
2. Click **Create report** in the upper-right corner of the workspace
3. Keep the charts for `val_recall` and `val_accuracy`, then click **Create report**
4. Type a title: `Churn model: which C do we ship?`
5. Above the charts, write the decision in one line: `C=1.0 catches the most churners on validation and passes the 0.60 recall gate`
6. Click **Publish to project**, then **Share** to copy the link

Say: "A workspace is where I explore. A report is what I send: the decision in one line, the evidence under it, live from the runs. The reader does not need to run any code."

## Step 2 (instructor only). The same report from code

Change: the report as a Python script, so every model release gets the same page

Right-click `101/04_report` in Explorer > **Open in Integrated Terminal**, then:

```text
conda activate tracking-101
pip install wandb-workspaces
python step1_report.py
```

Expect: `report saved: https://wandb.ai/<your-user>/churn-tracking-101/reportlist`. Open that page: the new report is at the top of the list. It compares the latest finished run for each of `C-0.001`, `C-0.05` and `C-1.0`, filtered by their run IDs. Repeating the demo does not add older runs to this report

Say: "When a report is code, it can run after every training pipeline. Same structure every time, no copy and paste."

**Remember:** a report puts the decision first and links it to the runs that support it
