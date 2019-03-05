"""Script for joining a new competition."""

import zipfile
import argparse
from subprocess import run
from os import listdir, makedirs
from os.path import abspath, join, pardir

project_dir = join(abspath(__file__), pardir)

parser = argparse.ArgumentParser(description="Get set up for a new kaggle competition.")

parser.add_argument(
    "name", nargs="+", type=str, help="Competition name as in kaggle API."
)

args = parser.parse_args()
comp_name = args.name[0]
comp_underscore = comp_name.replace("-", "_")

def main():
    
    print("\n --> Setting up for competition:", comp_name, "\n")

    res = run(f"kaggle competitions files {comp_name}", capture_output=True)
    
    print(res.stdout.decode("utf-8"))

    if res.returncode:
        print("Stopping.")
        return None

    run(f"kaggle competitions leaderboard {comp_name} --show")

    approve = input("\nWant to set up this competition? [y/n]: ")

    if approve.lower() != "y":
        print("\nStopping.")
        return None

    print("\nCreating folders in data + notebooks:")
    raw_path = join(project_dir, "data", comp_underscore, "raw")
    notebook_path = join(project_dir, "notebooks", comp_underscore)
    print(raw_path)
    makedirs(raw_path, exist_ok=True)
    open(join(raw_path, ".gitkeep"), "a").close()
    print(notebook_path)
    makedirs(notebook_path, exist_ok=True)
    open(join(notebook_path, ".gitkeep"), "a").close()

    print("\nDownloading data to new folder:\n")
    run(f"kaggle competitions download -p {raw_path} {comp_name}")

    zip_files = [fn for fn in listdir(raw_path) if fn.endswith(".zip")]
    if len(zip_files):
        print("\nUnzipping data:")
        for f in zip_files:
            print(f, end=" ")
            zip_ref = zipfile.ZipFile(join(raw_path, f), 'r')
            zip_ref.extractall(raw_path)
            zip_ref.close()

    print("\n\nDone.")


if __name__ == "__main__":
    main()
