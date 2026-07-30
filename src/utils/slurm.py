"""
Utilities for launching Slurm jobs.
"""
import os
import re
import subprocess

from utils import as_strings


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


def build_command(script_path, hardware, conda_env, config_path, seed, result_file, verbosity, launcher_args):
    """
    Builds an `sbatch` call suitable for launching the given script on a Slurm cluster. Once built, the command can be
    passed to `utils.slurm.call_sbatch()`. Assumes the script is in the `src/` folder.
    Args:
        script_path: The path to the script to launch.
        hardware: The type of hardware to launch on (actually this just maps to the pre-baked sbatch scripts in the
                 same directory as this script, and is specifically based on UVM's Slurm cluster).
        conda_env: The name of the conda environment to activate before running the script.
        config_path: The path of the config to pass to --config.
        seed: The seed to use for --seed.
        result_file: The path or filename to use for --metrics-output.
        verbosity: The verbosity level to run at.
        launcher_args: Arguments to be passed on to `sbatch`.

    Returns:
        A list of strings which can be used as an argument to `subprocess.run()`.
    """
    # Find the script to run, relative to this file.
    assert script_path.is_file(), f"Script file ({script_path}) not found or is not a file."
    if hardware == "nvgpu":
        sbatch_filename = "train.sbatch"
    elif hardware == "nvgpu2":
        sbatch_filename = "train-2gpu.sbatch"
    elif hardware == "preempt":
        sbatch_filename = "preempt-train.sbatch"
    elif hardware == "general":
        sbatch_filename = "train-cpu.sbatch"
    else:
        raise RuntimeError(f"Unrecognized hardware: {hardware}")
    sbatch_script = script_path.parent.parent / sbatch_filename
    assert sbatch_script.is_file(), f"SBATCH file ({sbatch_script}) not found or is not a file."

    # NOTE: We allow launching multiple different seeds from the same config, so supply these on the command line.
    train_cmd = [script_path, "--config", config_path, "--seed", seed, "--metrics-output", result_file]
    if verbosity:
        train_cmd.append("-" + ("v" * verbosity))

    # Add launcher wrapper.
    launch_cmd = ["sbatch"] + launcher_args + [sbatch_script, conda_env] + train_cmd
    launch_cmd = as_strings(launch_cmd)

    return launch_cmd
