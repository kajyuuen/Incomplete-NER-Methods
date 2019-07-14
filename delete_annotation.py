import os
import sys
import argparse

from src.utils.label_deleter import LabelDeleter

def main(input_file, output_file, p):
    label_deleter = LabelDeleter(input_file)
    label_deleter.delete_label(float(p))
    label_deleter.write(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--entity_keep_ratio", type=float, default=0.5)
    args = parser.parse_args()
    main(args.input_file, args.output_file, 1 - args.entity_keep_ratio)