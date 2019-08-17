import os
import sys
import argparse

from src.utils.labeler import Labeler

def main(input_file, output_file, to_lower=False):
    labeler = Labeler(input_file, output_file, to_lower)
    labeler.match()
    labeler.write(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--to_lower", action='store_true')
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.to_lower)
