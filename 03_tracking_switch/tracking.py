# tracking.py: every W&B call lives here, so main.py reads like a recipe
# Contract: every function accepts run=None and then does nothing
import wandb


def start_run(cfg):
    """Return an active W&B run, or None when tracking is switched off."""
    wb = cfg["wandb"]
    if not wb["enabled"]:
        print("tracking off: continuing without W&B")
        return None
    return wandb.init(
        project=wb["project"], name=wb["name"], job_type=wb["job_type"],
        tags=wb["tags"], config=cfg)  # the whole config is stored on the run


def log_metrics(run, metrics):
    if run is None:
        return
    run.log(metrics)


def log_file_artifact(run, name, artifact_type, file_path):
    if run is None:
        return
    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)


def finish_run(run, exit_code=0):
    if run is None:
        return
    run.finish(exit_code=exit_code)  # exit_code=1 marks the run as failed
