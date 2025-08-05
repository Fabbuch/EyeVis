#! /bin/env python

from utils import load_json, write_json
import sys

def write_index(data: list[dict], outpath: str):
    index_data = {}
    for trial in data:
        task_name = trial["task"]
        image_name = trial["name"]
        if image_name not in index_data.keys():
            index_data[image_name] = task_name
    write_json([index_data], outpath)

if __name__ == "__main__":
    json_path = sys.argv[1]
    outpath = sys.argv[2]
    data = load_json(json_path)
    write_index(data, outpath)