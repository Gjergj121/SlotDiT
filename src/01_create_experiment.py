import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("-d", "--exp-directory", required=True)
    args = parser.parse_args()
    source = Path(args.config)
    if not source.is_file():
        parser.error(f"Config not found: {source}")
    destination = Path(args.exp_directory)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "experiment_params.json"
    with target.open("x") as stream:
        stream.write(source.read_text())
    print(destination)


if __name__ == "__main__":
    main()
