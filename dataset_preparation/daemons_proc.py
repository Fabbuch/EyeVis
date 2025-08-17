#! /bin/env python

import os
import requests
import zipfile
import pandas as pd
from math import pi, tan

def download_data(url: str, extract: list[str], zip_path: str):
    if not os.path.exists(zip_path):
        movements = requests.get(url).content
        with open(zip_path, "wb") as f:
            f.write(movements)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("datasets", extract)

def create_subset_of_df():
    csv_path = "datasets/DAEMONS/fixations.csv"
    df = pd.read_csv(csv_path, index_col=0)
    subset_of_ids_df = df[df["VP"] < 10]
    subset_df = subset_of_ids_df[subset_of_ids_df["Img"].str.contains("DAEMONS_corpus_potsdam")]
    subset_images = ["DAEMONS_potsdam_corpus"+"/"+path for path in set(subset_df["Img"])]
    subset_df["image_name"] = subset_df["Img"].apply(lambda x: x[-8:])
    subset_df = subset_df.drop(columns=["Img"])
    subset_df = subset_df.rename({"VP": "subject_id", "fixdur": "fixation_duration"})
    subset_df.to_csv(csv_path)
    return subset_images

def add_xy_px():
    csv_path = "datasets/DAEMONS/fixations.csv"
    df = pd.read_csv(csv_path, index_col=0)
    df[["x_px", "y_px"]] = df.apply(visual_angle_to_px, axis=1)
    df.drop(["x", "y"])
    df.rename({"x_px": "x", "y_px": "y"})
    df.to_csv(csv_path)

def visual_angle_to_px(row) -> pd.Series:
    angle_x, angle_y = row["x"], row["y"]
    distance = 95
    res_x, res_y = 1920, 1200
    max_angle_x, max_angle_y = 32, 18
    width = 2 * distance * tan(max_angle_x/180 * pi/2)
    height = 2 * distance * tan(max_angle_y/180 * pi/2)
    mm_per_px_x = width/res_x
    mm_per_px_y = height/res_y
    displace_x = 2 * distance * tan(angle_x/180 * pi/2)
    displace_y = 2 * distance * tan(angle_y/180 * pi/2)
    x_px = displace_x/mm_per_px_x
    y_px = displace_y/mm_per_px_y
    return pd.Series([x_px, y_px], index=['x_px', 'y_px'])

def rename_images():
    corpus_path = "datasets/DAEMONS_potsdam_corpus"
    for path in os.listdir(corpus_path):
        if path.startswith("DAEMONS_corpus_potsdam"):
            abs_path = corpus_path + "/" + path
            new_path = corpus_path + "/" + path[-8:]
            os.rename(abs_path, new_path)

if __name__ == "__main__":
    # download movements
    download_data("https://osf.io/download/ztgna/", 
                  ["eye_movement/SAC_val.csv", "eye_movement/readme.md"],
                  "datasets/daemons.zip")
    subset_images = create_subset_of_df()
    # download images
    download_data("https://osf.io/download/x5jb2/", 
                  subset_images,
                  "datasets/images.zip")
    add_xy_px()
    rename_images()
    
    
