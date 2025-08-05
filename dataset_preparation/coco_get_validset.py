#! /bin/env python

import sys
from utils import load_json, write_json

def filter_data(data: list[dict]):
    valid_data = [entry for entry in data if entry["split"] == "valid"]
    return valid_data

if __name__ == "__main__":
    coco_path = sys.argv[1]
    data = load_json(coco_path)
    valid_data = filter_data(data)
    write_json(valid_data, "datasets/COCO_search18/fixations_val_subset.json")