#! /bin/env python

import plotly.graph_objects as go
from PIL import Image
import plotly.express as px
import os
import pandas as pd
import re
import io
from io import BytesIO
import base64

import duckdb

from langchain_core.tools import tool, InjectedToolArg
from typing_extensions import Annotated

from typing import Tuple, Any

DEFAULT_DATASETS = ['DAEMONS']

@tool("fixation_heat_map", response_format="content_and_artifact", parse_docstring=True)
def tool_fixation_heat_map(
        subject_ids: list[int] | int,
        image_name: str,
        dataset_name: Annotated[str, InjectedToolArg],
        fixation_df: Annotated[Any, InjectedToolArg],
        img_files: Annotated[dict | None, InjectedToolArg]
        ) ->  Tuple[str, go.Figure]:
    """Generate a heat map of fixation times for one or more subjects.

    Args:
        subject_ids: list of ids of subjects from which eye-tracking data was collected
        image_name: file name of the viewed image
        dataset_name: name of the dataset
        fixation_df: dataframe of fixation data from uploaded dataset. None if the dataset is one of the predefined default datasets
        img_files: dict of base64 encoded binary image files, with filenames as keys. None if the dataset is one of the predefined default datasets
    """
    # input validation: assure subject_ids is a list
    if isinstance(subject_ids, int):
        subject_ids = [subject_ids]
    
    if dataset_name in DEFAULT_DATASETS:
        # Get fixation data and image_path
        fixations_path = get_fixations_path(dataset_name)
        img_dir = get_images_dir(dataset_name)
        fixation_df = pd.read_csv(fixations_path)
        image_data = get_image_path(image_name, img_dir, dataset_name)
    else:
        # custom dataset
        if image_name not in img_files.keys():
            if not image_name.endswith(".jpg"):
                image_name = image_name+".jpg"
            raise ValueError(f"Image '{image_name}' does not exist in dataset {dataset_name}. Has to be one of {list(img_files.keys())}")
        img_string = img_files[image_name]
        _, content_string = img_string.split(',')
        decoded = base64.b64decode(content_string)
        image_data = BytesIO(decoded)
    
    fig, width, height = get_layout_image_fig(image_data)
    
    fixation_df = get_fixations(subject_ids, image_name, fixation_df)

    # Get all x, y coordinates and fixation durations
    x, y, t = fixation_df["x"].to_list(), fixation_df["y"].to_list(), fixation_df["fixation_duration"].to_list()
    
    fig.update_layout({"title": f"Heat map: {image_name}"})
    max_time = max(t)

    fig.add_trace(
        go.Histogram2d(x=x, y=y, z=t, zsmooth="best", histfunc="avg", zmin=max_time*0.05, zmax=max_time*0.55,
            hovertemplate="%{z}ms (avg)<extra></extra>",
            xbins={
                "start": 0, 
                "end": width, 
                "size": 120
            }, 
            ybins={
                "start": 0,
                "end": height,
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
def tool_scan_path_plot(
        subject_ids: list[int] | int,
        image_name: str,
        dataset_name: Annotated[str, InjectedToolArg],
        fixation_df: Annotated[Any, InjectedToolArg],
        img_files: Annotated[dict | None, InjectedToolArg]
        ) ->  Tuple[str, go.Figure]:
    """Generate a plot showing the scan path of the fixations of one or more subjects.

    Args:
        subject_ids: list of ids of subjects from which eye-tracking data was collected
        image_name: file name of the viewed image
        dataset_name: name of the dataset
        fixation_df: dataframe of fixation data from uploaded dataset. None if the dataset is one of the predefined default datasets
        img_files: dict of base64 encoded binary image files, with filenames as keys. None if the dataset is one of the predefined default datasets
    """
    # input validation:
    if isinstance(subject_ids, int):
        subject_ids = [subject_ids]

    if dataset_name in DEFAULT_DATASETS:
        # Get fixation data and image_path
        fixations_path = get_fixations_path(dataset_name)
        fixation_df = pd.read_csv(fixations_path)
        img_dir = get_images_dir(dataset_name)
        image_data = get_image_path(image_name, img_dir, dataset_name)
    else:
        # custom dataset
        img_string = img_files[image_name]
        _, content_string = img_string.split(',')
        decoded = base64.b64decode(content_string)
        image_data = BytesIO(decoded)

    fig, _, _ = get_layout_image_fig(image_data)

    fixation_df = get_fixations(subject_ids, image_name, fixation_df)

    fig.update_layout({"title": f"Scan path: {image_name}"})

    # Since there are 10 subjects in the default dataset, a discrete color scale with 10 different colors
    # is enough to generate 10 different colors. For custom datasets, a larger number of different colors
    # might need to be generated.
    colors = px.colors.qualitative.G10

    for i, id in enumerate(fixation_df["subject_id"].unique()):
        color = colors[i]
        fixations_for_subject = fixation_df[fixation_df["subject_id"] == id]
        x, y, t = fixations_for_subject["x"].to_list(), fixations_for_subject["y"].to_list(), fixations_for_subject["fixation_duration"].to_list()
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

def get_fixations(subject_id_list: list[int], image_name: str, fixation_df: pd.DataFrame) -> pd.DataFrame:
    """Select rows with fixation data for the given subjects and image from a pandas dataframe."""

    fixation_rows = fixation_df[(fixation_df["subject_id"].isin(subject_id_list)) & (fixation_df["image_name"] == image_name)]
    
    if len(fixation_rows) == 0:
        raise ValueError(f"No available data for image '{image_name}' and subject ids: {subject_id_list}")
    return fixation_rows
    
def get_image_path(image_name: str, img_dir: str, dataset_name: str) -> str:
    # Adding file extension if it is missing
    if not image_name.endswith(".jpg"):
        image_name = image_name+".jpg"
    valid_image_names = get_valid_image_names(dataset_name)
    if image_name not in valid_image_names:
        raise ValueError(f"Image '{image_name}' does not exist. Has to be one of {valid_image_names}")
    return img_dir + "/" + image_name

@tool("query_dataset", response_format="content", parse_docstring=True)
def tool_query_dataset(
    query: str, 
    dataset_name: Annotated[str, InjectedToolArg],
    fixation_df: Annotated[Any, InjectedToolArg]
    ) -> list:
    """Query the dataset using an SQL query (duckdb dialect).

    Args:
        query: an SQL query string. The dataset is daemons, with each entry being one fixation. use single quotes to escape strings, for example: "select distinct subject_id from daemons where image_name == '0061.jpg'"
        dataset_name: name of the dataset
        fixation_df: dataframe of fixation data from uploaded dataset. None if the dataset is one of the predefined default datasets
    """
    if dataset_name in DEFAULT_DATASETS:
        fixations_path = get_fixations_path(dataset_name)
        fixation_df = pd.read_csv(fixations_path)

    query = query.replace("daemons", "tool_query_dataset.fixation_df")
    query = query.replace("“", "'")
    query = query.replace("”", "'")
    query = query.replace("\"", "'")

    valid_col_names = list(fixation_df.columns)
        
    try:
        result_df = duckdb.sql(query).df()
    except duckdb.duckdb.BinderException as exc:
        exc_msg = str(exc)
        invalid_column_match = re.search('Referenced column "(.+)"', exc_msg)
        if invalid_column_match == None:
            raise ValueError(f"Query '{query}' could not be processed: {exc_msg}")
        raise ValueError(f"Column '{invalid_column_match.group(1)}' does not exist. Has to be in {valid_col_names}")

    if result_df.empty:
        raise ValueError(f"No results for query '{query}'")
    
    header = [list(result_df.keys())]
    data = result_df.drop_duplicates().values.tolist()
    return header + data

def get_layout_image_fig(image_path: str | BytesIO) -> tuple[go.Figure, int, int]:
    fig = default_fig_factory()
    image = Image.open(image_path)
    width, height = image.size
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
    return fig, width, height

def default_fig_factory() -> go.Figure:
    default_figure = go.Figure(
        layout={
            "scene": {"aspectratio": {"x": 1920, "y": 1080}},
            "xaxis": {"visible": True},
            "yaxis": {"visible": True},
            "title": "Generate a figure",
            "margin": {"b": 16, "l": 16, "r": 16, "t": 70}
            }
        )
    return default_figure

def get_dataset_info(dataset_name: str) -> str:
    fixations_path = get_fixations_path(dataset_name)
    df = pd.read_csv(fixations_path)
    
    # Get info and save it as a string
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    return info_string

def get_valid_col_names(dataset_name: str) -> list[str]:
    fixations_path = get_fixations_path(dataset_name)
    df = pd.read_csv(fixations_path)
    col_names = list(df.columns)
    return col_names

def get_valid_image_names(dataset_name: str) -> list[str]:
    img_dir = get_images_dir(dataset_name)
    valid_image_names = [path for path in os.listdir(img_dir) if not path.startswith(".")]
    return valid_image_names

def get_fixations_path(dataset_name: str) -> str:
    return "datasets" + "/" + dataset_name + "/" + "fixations.csv"

def get_images_dir(dataset_name: str) -> str:
    return "datasets" + "/" + dataset_name + "/" + "img"

def find_missing_columns(fixation_df: pd.DataFrame) -> list[str]:
    necessary_columns = ["subject_id", "image_name", "x", "y", "fixation_duration"]
    missing_columns = []
    for col in necessary_columns:
        if not col in fixation_df.columns:
            missing_columns.append(col)
    return missing_columns

tools = {
    name[5:]: func for (name, func) in globals().items() if name.startswith("tool_")
}

