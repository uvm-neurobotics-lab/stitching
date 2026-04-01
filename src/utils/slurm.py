"""
Utilities for launching Slurm jobs.
"""
import os
import re
import subprocess


def from_cfg_to_cmd(to_copy, from_config, dest_args):
    for arg in to_copy:
        val = from_config.get(arg)
        if val:
            dest_args.append("--" + arg.replace("_", "-"))
            if isinstance(val, (list, tuple)):
                dest_args.extend(val)
            else:
                dest_args.append(val)


def call_sbatch(cmd, verbose=False, dry_run=False, env=None, return_job_id=False):
    """
    Run the given command, which is assumed to be a call to `sbatch` or similar. We will expect that the console output
    contains the ID of a job which was launched.
        - In the case of success, this will output the ID of the job which was launched.
        - In the case of failure, this will print the full console log for debugging.

    Args:
        cmd: A command for `sbatch` or an `sbatch` wrapper like Neuromanager's `launcher`.
        verbose: Whether to print the console output of `cmd`, rather than swallowing it.
        dry_run: Do not actually call the command, instead just print it to the console.
        env: If not None, modify the environment with the provided dict entries.
        return_job_id: If True, return the Slurm job ID (int) instead of the process exit code.

    Returns:
        int: The exit code of the called process, or 0 in the case of a dry run (if return_job_id is False).
             Or the Slurm job ID if return_job_id is True.
    """
    if dry_run:
        print("Command that would be run:")
        print("    " + " ".join(cmd))
        return 0

    try:
        envvars = [f"{k}={v}" for k, v in env.items()] if env else []
        print("Running command: " + " ".join(envvars + cmd))
        newenv = None
        if env:
            newenv = os.environ.copy()
            newenv.update(env)
        
        # If we need to return the job ID, we MUST capture stdout regardless of verbose setting.
        # But if verbose is also True, we still want to see the output.
        if return_job_id or not verbose:
            stderr = subprocess.STDOUT
            stdout = subprocess.PIPE
        else:
            stderr = None
            stdout = None

        res = subprocess.run(cmd, text=True, check=True, env=newenv, stdout=stdout, stderr=stderr)
        
        job_id = None
        if stdout is not None:
            if verbose:
                print(res.stdout)
            # Find the Slurm job ID in the output and print it, if we captured the output.
            match = re.search(r"Submitted batch job (\d+)", res.stdout)
            if not match:
                print("WARNING: Could not find Slurm job ID in launcher output. Output of launcher:")
                print(res.stdout)
            else:
                job_id = int(match.group(1))
                print(f"Submitted batch job {job_id}")

        if return_job_id:
            return job_id
        return res.returncode
    except subprocess.CalledProcessError as e:
        # Print the output if we captured it, to allow for debugging.
        if not verbose or return_job_id:
            print("LAUNCH FAILED. Launcher output:")
            print("-" * 80)
            print(e.stdout)
            print("-" * 80)
        raise
