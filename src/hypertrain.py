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
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb

import launch_multidataset_training as launch_utils
from stitch_train import validate_config


SEEDS = [12345, 67890, 111213]


def get_job_status(job_ids):
    """Check the status of Slurm jobs using squeue."""
    if not job_ids:
        return {}
    
    # Filter out None job_ids (e.g. from dry runs or failed launches).
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
    # Initialize W&B. This will pick up the sweep configuration.
    run = wandb.init()
    sweep_config = run.config

    # Prepare the base configuration. We accept the same args as launch_multidataset_training.py.
    parser = launch_utils.create_arg_parser(__doc__)
    args, launcher_args = parser.parse_known_args()
    
    # Add our custom flag to return job IDs.
    # TODO: Do this differently.
    args.return_job_ids = True

    config = launch_utils.prep_config(parser, args)
    config["run_name"] = run.name
    
    # Override config with sweep parameters.
    # The requirement says assume config["optimizer_args"] always has "lr" and "weight_decay".
    if "lr" in sweep_config:
        config["train_config"]["optimizer_args"]["lr"] = sweep_config["lr"]
    if "weight_decay" in sweep_config:
        config["train_config"]["optimizer_args"]["weight_decay"] = sweep_config["weight_decay"]
    
    # Handle other possible sweep params.
    if "batch_size" in sweep_config:
        config["train_config"]["batch_size"] = sweep_config["batch_size"]
    if "epochs" in sweep_config:
        config["train_config"]["epochs"] = sweep_config["epochs"]

    # Re-validate after overrides.
    config = validate_config(config, dataset_required=False)

    # Launch jobs for multiple seeds.
    print(f"Launching multi-dataset training jobs for seeds {SEEDS}...")
    all_job_ids = []
    for seed in SEEDS:
        print(f"\n\n---------- SEED {seed} ----------")
        config["train_config"]["seed"] = seed
        job_ids = launch_utils.setup_and_launch_jobs(config, args, launcher_args)
        all_job_ids.extend(job_ids)
    
    print(f"All launched jobs: {all_job_ids}")

    # Wait for completion.
    # Track which jobs we are still waiting for.
    active_job_ids = [jid for jid in all_job_ids if jid is not None]
    
    # We need to know where the results will be saved to extract accuracy later.
    rootdir = launch_utils.result_rootdir(config)

    print("Waiting for jobs to complete...")
    while active_job_ids:
        # Check status.
        statuses = get_job_status(active_job_ids)
        
        # A job is finished if it's no longer in squeue.
        # (Note: this is a simplification; ideally we'd use sacct to check for successful completion.)
        active_job_ids = [jid for jid in active_job_ids if jid in statuses]
        
        if active_job_ids:
            print(f"Still waiting for {len(active_job_ids)} jobs: {active_job_ids}")
            # Report dummy value to keep wandb alive
            wandb.log({"Avg Test Accuracy": 0.0})
            time.sleep(60)  # Check every minute

    print("All jobs finished. Collecting results...")

    # Collect results.
    dataset_accuracies = {dataset: defaultdict(list) for dataset in config["datasets"]}
    
    for seed in SEEDS:
        config["train_config"]["seed"] = seed
        res_fname = launch_utils.result_filename(config)
        
        for dataset in config["datasets"]:
            result_path = rootdir / dataset / res_fname
            if result_path.exists():
                df = pd.read_pickle(result_path)
                if not df.empty:
                    # Extract final accuracy from the last row.
                    for metric in ("Train Accuracy", "Test Accuracy"):
                        acc = df[f"Overall/{metric}"].iloc[-1]
                        dataset_accuracies[dataset][f"{metric}"].append(acc)
                else:
                    raise RuntimeError(f"Result file for {dataset} (seed {seed}) is empty.")
            else:
                raise RuntimeError(f"Result file for {dataset} (seed {seed}) not found at {result_path}")

    # Calculate averages and report to W&B.
    final_accuracies = defaultdict(list)
    for dataset, metrics in dataset_accuracies.items():
        for metric, accs in metrics.items():
            if accs:
                avg = np.mean(accs)
                std = np.std(accs)
                print(f"{avg:.2%} ({std:.2%}) = {dataset} {metric}")
                wandb.log({f"{dataset}/{metric}": avg, f"{dataset}/{metric} Std": std})
                final_accuracies[metric].append(avg)
            else:
                raise RuntimeError(f"No results collected for dataset {dataset}.")
    if not final_accuracies:
        raise RuntimeError("No test accuracies were collected.")

    for metric, accs in final_accuracies.items():
        avg = np.mean(accs)
        std = np.std(accs)
        print(f"\nAverage {metric} across all datasets: {avg:.2%} ({std:.2%})")
        wandb.log({f"Avg {metric}": avg, f"Std {metric}": std})

    run.finish()


if __name__ == "__main__":
    main()
