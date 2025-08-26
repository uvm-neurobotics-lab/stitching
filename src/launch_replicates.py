"""
Scan a given folder for jobs which previously ran successfully, and launch new instances of the existing jobs but with
new random seeds.

To test this script, try:
    python src/launch_replicates.py -d <experiment-folder> --seed 67890 -n -vv
"""
import os
import sys
from pathlib import Path

import utils.argparsing as argutils
from stitch_train import build_command
from utils.slurm import call_sbatch

CFG_FILENAME = "config.yml"


def create_arg_parser(desc, allow_abbrev=True):
    """
    Creates the argument parser for this program.

    Args:
        desc (str): The human-readable description for the arg parser.
        allow_abbrev (bool): The `allow_abbrev` argument to `argparse.ArgumentParser()`.

    Returns:
        argutils.ArgParser: The parser.
    """
    parser = argutils.create_parser(desc, allow_abbrev=allow_abbrev)

    # Main Arguments
    parser.add_argument("-d", "--dir", metavar="FOLDER", type=argutils.existing_path, required=True,
                        help="Location to scan for job configs.")
    parser.add_argument("--seed", type=int, required=True, help="The new seed to run.")
    parser.add_argument("--orig-seed", type=int, default=12345, help="The seed of the first replicate.")

    # Launcher Arguments
    parser.add_argument("--cluster", metavar="NAME", default="nvgpu", choices=["nvgpu", "hgnodes", "gpu-test"],
                        help="The Slurm partition on which to launch eval jobs.")
    parser.add_argument("--conda-env", "--conda", "--env", metavar="NAME", default="stitch",
                        help="The Conda environment to activate before running the job.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Run even if results already exist. Overwrite config files if they exist.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Do not actually launch jobs, but only print out the equivalent commands that would be"
                             " launched.")
    argutils.add_verbose_arg(parser)
    parser.add_argument("--lv", "--launch-verbose", dest="launch_verbose", action="store_true",
                        help="Be verbose when launching the job (output all the launcher print statements).")
    return parser


def scan_and_launch(args, launcher_args):
    """ Scan for job configs and launch each one if the given results don't already exist. """
    result = 0
    launched = 0
    root_path = Path(".").resolve()

    # Launch a job for each config we find.
    existing_results = list(sorted(args.dir.resolve().rglob(f"result-{args.orig_seed}.pkl")))
    for existing_file in existing_results:
        # Get relative paths, for cleaner printing.
        try:
            name = str(existing_file.parent.relative_to(root_path))
        except ValueError:
            # Paths are not relative, so just use the full path.
            name = str(existing_file.parent)
        print(f"\n---- LAUNCHING {name} ----\n")

        # Ensure config exists, navigate to parent folder.
        cfgfile = existing_file.parent.resolve() / CFG_FILENAME
        assert cfgfile.is_file(), f"Missing config file: {cfgfile}"
        outdir = existing_file.parent
        if not args.dry_run:
            os.chdir(outdir.resolve())
            print(f"Run folder: {outdir}")
        else:
            print(f"Would change directories to run folder: {outdir}")

        # Check if results already exist.
        res_fname = f"result-{args.seed}.pkl"
        result_file = existing_file.parent / res_fname
        if result_file.is_file():
            if args.force:
                print(f"WARNING: Overwriting {result_file} due to --force.")
            else:
                print(f"Skipping due to result already existing.")
                continue

        # Get the launch command.
        command = build_command(args.cluster, args.conda_env, CFG_FILENAME, args.seed, res_fname, args.verbose,
                                launcher_args)

        # Launch the job.
        # NOTE: The MKL_THREADING_LAYER variable is a workaround for an issue I was experiencing on the VACC while
        #       using torchrun.
        result += call_sbatch(command, args.launch_verbose, args.dry_run, env={"MKL_THREADING_LAYER": "GNU"})
        if result == 0:
            launched += 1

    print(f"\nFound a total of {len(existing_results)} configs and launched {launched} jobs.")
    return result


def main(argv=None):
    parser = create_arg_parser(__doc__)
    args, launcher_args = parser.parse_known_args(argv)
    if args.seed == args.orig_seed:
        parser.error("Seed should not be the same as the original seed.")
    return scan_and_launch(args, launcher_args)


if __name__ == "__main__":
    sys.exit(main())
