# EyeVis: LLM-assisted Data Analysis for Eye-tracking Data
This is a data analysis tool that makes the visualisation of eye-tracking data easier by using an LLM-agent that listens to your instructions and generates plots for you. You can use this too to generate scan path plots, fixation heat maps or query the dataset through natural language.
You can explore some publically available datasets included in this repository under ```/datasets``` or you can upload your own dataset to visualise.

## Installation Instructions
This is a data app made with [Dash](https://dash.plotly.com/).
This app uses llama3.2:3b a small, quantized llama language model (run locally through [ollama](https://ollama.com/)) by default, but can be configured to use other language models that support tool calling.
Before running this app, make sure to download ollama and run
```bash
ollama pull llama3.2:3b
```
To download the llama3.2:3b model (this requires 2GB of disk space).
Also make sure to install all other required python dependencies with
```bash
pip install -r requirements.txt
```
Start the app like this:
```bash
python app.py
```

## Datasources
The default dataset included in this repository is a subset of The Potsdam dataset of eye movement on natural scenes (DAEMONS) from [Schwetlick et al. 2024](https://doi.org/10.3389/fpsyg.2024.1389609), available for download through OSF (Open science framework) at [this link](https://osf.io/ewr5u/). Eye-tracking data was collected for this dataset with still images as stimuli. And this app is made with this form of eye-tracking data in mind. The data is recorded in the form of screen coordinates of fixations and saccades as well as their durations. If you want to upload your own data, make sure your data is also processed in this way.

## Uploading a Dataset
This tool operates on datasets consisting of a csv file with fixation data and a set of images over which fixations were collected. If you want to upload your own dataset, make sure your data follows the same structure as the DAEMONS dataset (check the datset's README included here in ```datasets/DAEMONS/readme.md``` if you want to know more about the structure). Specifically, make sure that your csv file contains the following columns:
- 'subject_id': contains numerical subject identifiers
- 'fixation_duration': contains fixation durations
- 'image_name': contains filenames of the uploaded images
- 'x': contains x coordinates of the corresponding fixations
- 'y': contains y coordinate of the corresponding fixations

## Structure of this Repository
This repository contains the source code for the EyeVis app. The ```/dataset_preparation``` directory also contains any scripts that were used for preprocessing of the pubically available datasets included in the app.

## References
- Schwetlick, L., Kümmerer, M., Bethge, M., & Engbert, R. (2024). Potsdam data set of eye movement on natural scenes (Daemons). Frontiers in Psychology, 15, 1389609. https://doi.org/10.3389/fpsyg.2024.1389609