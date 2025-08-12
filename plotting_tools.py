#! /bin/env python

import plotly.graph_objects as go
from PIL import Image
from utils import load_json
import plotly.express as px
import os
import pandas as pd
import re

import duckdb

from langchain_core.tools import tool

from typing import Tuple


# Fixed paths for default dataset "COCO"
# INDEX_PATH = "datasets/COCO_search18/task_image_index.json"
# FIXATIONS_PATH = "datasets/COCO_search18/fixations_val_subset.json"
# IMAGES_DIR = "datasets/COCO_search18/coco_search18_images_TA"

# Fixed paths for default dataset "DAEMONS"
FIXATIONS_PATH = "datasets/eye_movement/SAC_val.csv"
IMAGES_DIR = "datasets/DAEMONS_potsdam_corpus"

# valid_image_names = list(load_json(INDEX_PATH)[0].keys())
valid_image_names = [path for path in os.listdir("datasets/DAEMONS_potsdam_corpus") if not path.startswith(".")]
valid_col_names = list(pd.read_csv(FIXATIONS_PATH).keys())

@tool("fixation_heat_map", response_format="content_and_artifact", parse_docstring=True)
def tool_fixation_heat_map(subject_ids: list[int], image_name: str) ->  Tuple[str, go.Figure]:
    """Generate a heat map of fixation times for one or more subjects.

    Args:
        subject_ids: list of ids of subjects from which eye-tracking data was collected
        image_name: file name of the viewed image
    """
    # Get fixation data and image_path
    fixations_df = get_fixations(subject_ids, image_name)
    # Get all x, y coordinates and fixation durations
    x, y, t = fixations_df["x_px"].to_list(), fixations_df["y_px"].to_list(), fixations_df["fixation_duration"].to_list()
    image_path = get_image_path(image_name)

    fig = get_layout_image_fig(image_path, 1920, 1080)
    max_time = max(t)

    fig.add_trace(
        go.Histogram2d(x=x, y=y, z=t, zsmooth="best", histfunc="avg", zmin=max_time*0.05, zmax=max_time*0.55,
            hovertemplate="%{z}ms (avg)<extra></extra>",
            xbins={
                "start": 0, 
                "end": 1920, 
                "size": 120
            }, 
            ybins={
                "start": 0,
                "end": 1080,
                "size": 120
            }, 
            # based on preset 'thermal' with added alpha
            colorscale=[
                'rgba(0, 0, 0, 0)',
                'rgba(60, 0, 210, 0.6)',
                'rgba(80, 20, 195, 0.8)',
                'rgba(126, 40, 160, 1)',
                'rgba(158, 60, 145, 1)',
                'rgba(193, 100, 125, 1)',
                'rgba(225, 113, 100, 1)',
                'rgba(255, 139, 75, 1)',
                'rgba(255, 173, 60, 1)',
                'rgba(255, 211, 30, 1)',
                'rgba(255, 255, 0, 1)'
            ],
            opacity=0.8
        )
    )

    content = f"Generated heat map plot for subjects with ids {subject_ids} and image {image_name}."
    return content, fig

@tool("scan_path_plot", response_format="content_and_artifact", parse_docstring=True)
def tool_scan_path_plot(subject_ids: list[int], image_name: str) ->  Tuple[str, go.Figure]:
    """Generate a plot showing the scan path of the fixations of one or more subjects.

    Args:
        subject_ids: list of ids of subjects from which eye-tracking data was collected
        image_name: file name of the viewed image
    """
    # Get fixation data and image_path
    df_fixations = get_fixations(subject_ids, image_name)

    image_path = get_image_path(image_name)

    fig = get_layout_image_fig(image_path, 1920, 1080)

    # Since there are 10 subjects in the default dataset, a discrete color scale with 10 different colors
    # is enough to generate 10 different colors. For custom datasets, a larger number of different colors
    # might need to be generated.
    colors = px.colors.qualitative.G10

    for i, id in enumerate(df_fixations["subject_id"].unique()):
        color = colors[i]
        fixations_for_subject = df_fixations[df_fixations["subject_id"] == id]
        x, y, t = fixations_for_subject["x_px"].to_list(), fixations_for_subject["y_px"].to_list(), fixations_for_subject["fixation_duration"].to_list()
        fig.add_scatter(x=x, y=y, mode="lines+markers", text=t,
                        hovertemplate="%{text}ms<extra></extra>",
                        marker={
                            "size": 12,
                            "symbol": "arrow",
                            "angleref": "previous"
                        },
                        name=f"ID {id}",
                        line={
                            "color": color,
                            "width": 2,
                        }
                    )
    # # Add circle marker at start:
    # fig.add_scatter(x=x[:1], y=y[:1], mode="markers",
    #                 hovertemplate="Start<extra></extra>",
    #                 marker={
    #                     "size": 12,
    #                     "symbol": "circle",
    #                     "color": "red"
    #                 },
    #             )
    content = f"Generated scan path plot for subjects with ids {subject_ids} and image {image_name}."
    return content, fig

def get_fixations(subject_id_list: list[int], image_name: str) -> pd.DataFrame:
    """Select rows with fixation data for the given subjects and image."""
    df = pd.read_csv(FIXATIONS_PATH)
    
    # Adding file extension if it is missing
    if not image_name.endswith(".jpg"):
        image_name = image_name+".jpg"
    if image_name not in valid_image_names:
        raise ValueError(f"Image '{image_name}' does not exist. Has to be one of {valid_image_names}")

    df_fixations = df[(df["subject_id"].isin(subject_id_list)) & (df["image_name"] == image_name)]
    
    if len(df_fixations) == 0:
        raise ValueError(f"No available data for image '{image_name}' and subject ids: {subject_id_list}")
    return df_fixations
    
def get_image_path(image_name: str) -> str:
    # Adding file extension if it is missing
    if not image_name.endswith(".jpg"):
        image_name = image_name+".jpg"
    if image_name not in valid_image_names:
        raise ValueError(f"Image '{image_name}' does not exist. Has to be one of {valid_image_names}")
    return IMAGES_DIR + "/" + image_name

@tool("query_dataset", response_format="content", parse_docstring=True)
def tool_query_dataset(query: str) -> list:
    """Query the dataset using an SQL query (duckdb dialect).

    Args:
        query: an SQL query string. The dataset is daemons, with each entry being one fixation. use single quotes to escape strings, for example: "select distinct subject_id from daemons where image_name == 'DAEMONS_corpus_potsdam_0061.jpg'"
    """
    df = pd.read_csv(FIXATIONS_PATH)

    query = query.replace("daemons", "tool_query_dataset.df")
    query = query.replace("“", "'")
    query = query.replace("”", "'")
    query = query.replace("\"", "'")

    valid_col_names = ["subject_id", "fixation_duration", "image_name"]
        
    try:
        result_df = duckdb.sql(query).df()
    except duckdb.duckdb.BinderException as exc:
        exc_msg = str(exc)
        invalid_column_math = re.search('Referenced column "(.+)"', exc_msg)
        if invalid_column_math == None:
            raise ValueError(f"Query '{query}' could not be processed: {exc_msg}")
        raise ValueError(f"Column '{invalid_column_math.group(1)}' does not exist. Has to be in {valid_col_names}")

    if result_df.empty:
        raise ValueError(f"No results for query '{query}'")
    
    header = [list(result_df.keys())]
    data = result_df.drop_duplicates().values.tolist()
    return header + data

def get_layout_image_fig(image_path: str, width: int, height: int) -> go.Figure:
    fig = go.Figure()
    image = Image.open(image_path)
    fig.add_layout_image(
            x=0,
            sizex=width,
            y=0,
            sizey=height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            source=image,
            yanchor="bottom"
    )
    fig.update_xaxes(showgrid=False, visible=False, range=(0, width))
    fig.update_yaxes(showgrid=False, visible=False, scaleanchor='x', range=(0, height))
    return fig


tools = {
    name[5:]: func for (name, func) in globals().items() if name.startswith("tool_")
}

