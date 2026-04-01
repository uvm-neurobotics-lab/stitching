"""
A script to perform hyperparameter tuning using W&B sweeps. Launches Slurm jobs via launch_multidataset_training.py and
waits for them to finish before reporting the final average test accuracy over all datasets.

To test this script standalone, try:
    python src/hypertrain.py -c moe-stitch/baseline-resnet-cifar-resisc.yml -n -vv
    
To run the sweep via W&B:
    wandb sweep tests/hypersearch.yml
    CONFIG=moe-stitch/baseline-resnet-cifar-resisc.yml HARDWARE=v100 wandb agent <sweep-id> --count 25
"""
import time
import pandas as pd
import wandb
import subprocess
from pathlib import Path

import launch_multidataset_training as launch_utils
from stitch_train import validate_config


def get_job_status(job_ids):
    """Check the status of Slurm jobs using squeue."""
    if not job_ids:
        return {}
    
    # Filter out None job_ids (e.g. from dry runs or failed launches)
    valid_job_ids = [str(jid) for jid in job_ids if jid is not None]
    if not valid_job_ids:
        return {}

    try:
        # Run squeue to get status of our jobs.
        # -h: no header, -j: job IDs, -o %i %t: output job ID and state.
        cmd = ["squeue", "-h", "-j", ",".join(valid_job_ids), "-o", "%i %t"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        status_dict = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split()
                if len(parts) == 2:
                    jid, state = parts
                    status_dict[int(jid)] = state
        return status_dict
    except subprocess.CalledProcessError:
        # squeue might fail if some jobs are already finished and not in the queue anymore.
        # In that case, we can't easily distinguish between "finished" and "failed/never existed"
        # without sacct, but for simplicity, we'll assume they are finished if they are not in squeue.
        return {}


def main():
    # 1. Initialize W&B
    # This will pick up the sweep configuration
    run = wandb.init()
    sweep_config = run.config

    # 2. Prepare the base configuration
    parser = launch_utils.create_arg_parser("Hyperparameter sweep wrapper")
    # We expect the user to pass the base config via -c/--config as they would for launch_multidataset_training.py
    args, launcher_args = parser.parse_known_args()
    
    # Add our custom flag to return job IDs
    args.return_job_ids = True

    config = launch_utils.prep_config(parser, args)
    
    # 3. Override config with sweep parameters
    # The requirement says assume config["optimizer_args"] always has "lr" and "weight_decay"
    if "lr" in sweep_config:
        config["train_config"]["optimizer_args"]["lr"] = sweep_config["lr"]
    if "weight_decay" in sweep_config:
        config["train_config"]["optimizer_args"]["weight_decay"] = sweep_config["weight_decay"]
    
    # Handle other possible sweep params
    if "batch_size" in sweep_config:
        config["train_config"]["batch_size"] = sweep_config["batch_size"]
    if "epochs" in sweep_config:
        config["train_config"]["epochs"] = sweep_config["epochs"]

    # Re-validate after overrides
    config = validate_config(config, dataset_required=False)

    # 4. Launch jobs
    print("Launching multi-dataset training jobs...")
    job_ids = launch_utils.setup_and_launch_jobs(config, args, launcher_args)
    print(f"Launched jobs: {job_ids}")

    # 5. Wait for completion
    # Track which jobs we are still waiting for
    active_job_ids = [jid for jid in job_ids if jid is not None]
    
    # We need to know where the results will be saved to extract accuracy later.
    expname = config["config"].stem
    rootdir = Path(config["save_dir"]).resolve() / config["project"] / expname
    res_fname = launch_utils.result_filename(config)

    print("Waiting for jobs to complete...")
    while active_job_ids:
        # Check status
        statuses = get_job_status(active_job_ids)
        
        # A job is finished if it's no longer in squeue
        # (Note: this is a simplification; ideally we'd use sacct to check for successful completion)
        active_job_ids = [jid for jid in active_job_ids if jid in statuses]
        
        if active_job_ids:
            print(f"Still waiting for {len(active_job_ids)} jobs: {active_job_ids}")
            # Report dummy value to keep wandb alive
            wandb.log({"Avg Test Accuracy": 0.0})
            time.sleep(60) # Check every minute

    print("All jobs finished. Collecting results...")

    # 6. Collect results and report to W&B
    test_accuracies = []
    for dataset in config["datasets"]:
        result_path = rootdir / dataset / res_fname
        if result_path.exists():
            try:
                df = pd.read_pickle(result_path)
                if not df.empty:
                    # Extract "Overall/Test Accuracy" from the last row
                    acc = df["Overall/Test Accuracy"].iloc[-1]
                    test_accuracies.append(acc)
                    print(f"Dataset {dataset}: Test Accuracy = {acc}")
                else:
                    print(f"WARNING: Result file for {dataset} is empty.")
            except Exception as e:
                print(f"ERROR: Could not read result file for {dataset}: {e}")
        else:
            print(f"WARNING: Result file for {dataset} not found at {result_path}")

    if test_accuracies:
        avg_acc = sum(test_accuracies) / len(test_accuracies)
        print(f"Final Average Test Accuracy: {avg_acc}")
        wandb.log({"Avg Test Accuracy": avg_acc})
    else:
        print("ERROR: No test accuracies were collected.")
        wandb.log({"Avg Test Accuracy": 0.0})

    run.finish()


if __name__ == "__main__":
    main()
