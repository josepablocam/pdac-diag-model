#!/usr/bin/env python3
from argparse import ArgumentParser
import pickle

import nn


def load_models(files):
    models = []
    for _file in files:
        with open(_file, "rb") as fin:
            file_models = pickle.load(fin)
            models.extend(file_models)
    return models


def get_args():
    parser = ArgumentParser(description="Combine lists of trained models")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="List of paths to pickled lists of models",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save combined list",
    )
    return parser.parse_args()


def main():
    args = get_args()
    models = load_models(args.input)
    with open(args.output, "wb") as fout:
        pickle.dump(models, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
