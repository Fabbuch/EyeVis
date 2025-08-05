#! /bin/env python
import json

def load_json(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(data: list[dict], outpath: str):
    with open(outpath, "w") as outfile:
        json.dump(data, outfile, indent=4)