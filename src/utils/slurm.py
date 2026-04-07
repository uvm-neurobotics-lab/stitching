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


def call_sbatch(cmd, verbose=False, dry_run=False, return_job_id=False, env=None):
    """
    Run the given command, which is assumed to be a call to `sbatch` or similar. We will expect that the console output
    contains the ID of a job which was launched.
        - In the case of success, this will output the ID of the job which was launched.
        - In the case of failure, this will print the full console log for debugging.

    Args:
        cmd: A command for `sbatch` or an `sbatch` wrapper like Neuromanager's `launcher`.
        verbose: Whether to print the console output of `cmd`, rather than swallowing it.
        dry_run: Do not actually call the command, instead just print it to the console.
        return_job_id: If True, return both the process exit code (int) and the Slurm job ID (int).
        env: If not None, modify the environment with the provided dict entries.

    Returns:
        int: The exit code of the called process, or 0 in the case of a dry run.
        int: (Optional) The Slurm job ID, if `return_job_id` is True.
    """
    if dry_run:
        print("Command that would be run:")
        print("    " + " ".join(cmd))
        if return_job_id:
            return os.EX_OK, None
        else:
            return os.EX_OK

    try:
        envvars = [f"{k}={v}" for k, v in env.items()] if env else []
        print("Running command: " + " ".join(envvars + cmd))
        newenv = None
        if env:
            newenv = os.environ.copy()
            newenv.update(env)

        # If verbose, just let the launcher output directly to console.
        # But if we need to return the job ID, we MUST capture stdout regardless of verbose setting.
        if not return_job_id and verbose:
            stderr = None
            stdout = None
        else:  # Normally, redirect stderr -> stdout and capture them both into stdout.
            stderr = subprocess.STDOUT
            stdout = subprocess.PIPE

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
                print(match.group(0))

        if return_job_id:
            if job_id is None:
                raise RuntimeError("Unable to find Slurm job ID.")
            return res.returncode, job_id
        else:
            return res.returncode
    except subprocess.CalledProcessError as e:
        # Print the output if we captured it, to allow for debugging.
        if not verbose:
            print("LAUNCH FAILED. Launcher output:")
            print("-" * 80)
            print(e.stdout)
            print("-" * 80)
        raise
