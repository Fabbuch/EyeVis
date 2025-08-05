#! /bin/env python

import sys
import zipfile
from coco_get_validset import load_json

def extract_images(outpath: str, zip_path: str, image_paths: list[str]):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(outpath, image_paths)

def get_valid_image_paths(fixations_path: str):
    data = load_json(fixations_path)
    paths = []
    for trial in data:
        path_name = "coco_search18_images_TA/"+str(trial["task"])+"/"+str(trial["name"])
        paths.append(path_name)
    return paths

if __name__ == "__main__":
    zip_path = sys.argv[1]
    fixations_path = sys.argv[2]
    outpath = sys.argv[3]
    image_paths = get_valid_image_paths(fixations_path)
    extract_images(outpath, zip_path, image_paths)