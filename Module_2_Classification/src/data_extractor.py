import csv
import json
from pathlib import Path
from collections import defaultdict
from operator import itemgetter

import pandas as pd
from tqdm import tqdm


def ranking_labels(doc_labels: list) -> list[int]:
    """Marks label changes in the list
       [a, a, b, c] -> [0, 0, 1, 2]

    Args:
        doc_labels (list[str]): list of labels

    Returns:
        list[int]: list with labels 'ranks'
    """
    rank = []
    label_change_counter = 0
    for n in range(len(doc_labels)):
        if n != 0:
            if doc_labels[n - 1] != doc_labels[n]:
                label_change_counter += 1
        rank.append(label_change_counter)
    return rank


def process_json_files(folder_path: str, output_path: str):
    """Given a root path to the dataset in .json format, generates .csv file with the following structure:
       label, span_id (position of the span in text), text, filename

    Args:
        folder_path (str): root folder path
        output_path (str): output .csv fullpath

    Returns:
        None
    """
    counter_sucess_files = 0
    for file_path in tqdm(Path(folder_path).rglob("*.json")):
        with open(file_path, "r", encoding="utf-8") as json_file:
            json_data = pd.DataFrame(json.load(json_file))
            try:
                json_data["text"] = json_data["text"].fillna("")
                
#                 # Sort by filename to handle text segmentation
                json_data.sort_values(by=["label"], inplace=True)
#                 json_data.sort_values(by=["filename", "start"], inplace=True)
                
                # Combine pieces of text into spans
                json_data["span_id"] = ranking_labels(json_data["label"].tolist())
            
                json_data["filename"] = file_path.stem
                json_data.sort_values(by=["filename", "span_id"], inplace=True)
                
                grouped_data = json_data.groupby(["filename", "span_id"]).agg({
                    'label': 'first', 
                    'text': ' '.join
                }).reset_index()

                # Create 'filename' column
                grouped_data["filename"] = file_path.stem
                
                # Sort the data by filename and span_id
                grouped_data.sort_values(by=["filename", "span_id"], inplace=True)

                with open(output_path, "a", newline="", encoding='utf-8') as csv_file:
                    fieldnames = grouped_data.columns
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if counter_sucess_files == 0:
                        writer.writeheader()
                    for index, row in grouped_data.iterrows():
                        writer.writerow(row.to_dict())
                counter_sucess_files += 1

            except Exception as e:
                print(f"Encountered exception {e} while processing {file_path}")
                continue

    return None