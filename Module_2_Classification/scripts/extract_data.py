from sys import path

path.append(".")

import argparse

from src.data_extractor import process_json_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and output CSV.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input folder path containing JSON files.",
        default="../dataset_doc/train",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV file path.",
        default="../dataset_doc/text_classification_train.csv",
    )

    args = parser.parse_args()

    process_json_files(folder_path=args.input, output_path=args.output)
