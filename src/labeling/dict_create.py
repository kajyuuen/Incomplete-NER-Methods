import os
import sys
import argparse

from src.utils.dictionary_creator import DictionaryCreator

def main(input_file, output_file):
    label_deleter = DictionaryCreator(input_file)
    label_deleter.write(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
