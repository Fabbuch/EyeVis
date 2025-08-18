#! /bin/env python

import os
import requests
import zipfile
import pandas as pd
from math import pi, tan

def download_data(url: str, extract: list[str], zip_path: str):
    """This downloads a zip file from a url, extracts all files in extract and writes them to a 'datasets' subfolder.
    The download is skipped if a zip at zip_path already exists."""
    if not os.path.exists(zip_path):
        print(f"Downloading data from: {url}\n...")
        movements = requests.get(url).content
        with open(zip_path, "wb") as f:
            f.write(movements)
    print(f"Extracting into 'datasets' folder: \n{extract}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("datasets", extract)

def create_subset_of_df():
    """Creates subset of dataset, with only subjects with ids 1-10 and moves it together with the readme to a subfolder 'datasets/DAEMONS'."""
    csv_path = "datasets/eye_movement/SAC_val.csv"
    df = pd.read_csv(csv_path, index_col=0)
    subset_of_ids_df = df[df["VP"] < 10]
    subset_df = subset_of_ids_df[subset_of_ids_df["Img"].str.contains("DAEMONS_corpus_potsdam")]
    subset_images = ["DAEMONS_potsdam_corpus"+"/"+path for path in set(subset_df["Img"])]
    image_name_series = subset_df["Img"].apply(lambda x: x[-8:])
    subset_df.loc[:,"image_name"] = image_name_series
    subset_df = subset_df.drop(columns=["Img"])
    subset_df = subset_df.rename(columns={"VP": "subject_id", "fixdur": "fixation_duration"})
    subset_df.to_csv(csv_path)
    print("Writing subset of data to 'datasets/DAEMONS'\n...")
    os.rename(csv_path, "datasets/DAEMONS/fixations.csv")
    os.rename("datasets/eye_movement/readme.md", "datasets/DAEMONS/readme.md")
    return subset_images

def add_xy_px():
    """Converts x and y row in original df (which are in degree of visual angle) to screen space coordinates replacing the original data columns."""
    csv_path = "datasets/DAEMONS/fixations.csv"
    df = pd.read_csv(csv_path, index_col=0)
    df[["x_px", "y_px"]] = df.apply(visual_angle_to_px, axis=1)
    df = df.drop(columns=["x", "y"])
    df = df.rename(columns={"x_px": "x", "y_px": "y"})
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
    """Removes the 'DAEMONS_potsdam_corpus' prefix from image filenames."""
    corpus_path = "datasets/DAEMONS_potsdam_corpus"
    for path in os.listdir(corpus_path):
        if path.startswith("DAEMONS_corpus_potsdam"):
            abs_path = corpus_path + "/" + path
            new_path = "datasets/DAEMONS" + "/" + "img" + "/" + path[-8:]
            os.rename(abs_path, new_path)
    

if __name__ == "__main__":
    # create directory structure for resulting dataset:
    if not os.path.exists("datasets/DAEMONS"):
        os.makedirs("datasets/DAEMONS/img")
    # download movements
    download_data("https://osf.io/download/ztgna/", 
                  ["eye_movement/SAC_val.csv", "eye_movement/readme.md"],
                  "datasets/daemons.zip")
    subset_images = create_subset_of_df()
    # download subset of images
    download_data("https://osf.io/download/x5jb2/", 
                  subset_images,
                  "datasets/images.zip")
    # convert degrees of visual angle to screen coordinates
    add_xy_px()
    rename_images()
    # remove emtpy DAEMONS_potsdam_corpus and eye_movement directories
    os.rmdir("datasets/DAEMONS_potsdam_corpus")
    os.rmdir("datasets/eye_movement")
    
    
