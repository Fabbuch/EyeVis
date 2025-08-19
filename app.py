# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import plotting_tools as ptools
from dash import Dash, html, dcc, Input, Output, State, callback, Patch, no_update
import dash_bootstrap_components as dbc
import os
import base64
import pandas as pd
import io
import json

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages
)

datasets_path = os.getcwd() + "/" + "datasets"

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, title="EyeVis")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434"
).bind_tools(list(ptools.tools.values()))

# initialize message history:
initial_messages = [
    {"content":
        "You are a data analysis assistant for eye-tracking data. " \
        "Use the provided tools to answer the user's questions about the dataset or generate visualisations. " \
        "The dataset is tabular data, where each row represents a fixation. " \
        f"The table has the following column names: {ptools.get_valid_col_names('DAEMONS')}" \
    , "role": "system"}
]

# placeholder figure
fig = ptools.default_fig_factory(1920, 1080)

app.layout = dbc.Container([
    html.Div(children=[
        dbc.Row([
            html.Div(children="EyeVis: LLM-assisted data analysis for eye-tracking Data", 
            className="title"),
            html.Hr(className="line"),
            dbc.Row([
                html.Div(id="dataset-info", children="Currently selected dataset:", className="curr"),
                dcc.Dropdown(options=['DAEMONS'], value='DAEMONS', id='dataset-dropdown', clearable=False, className="dropdown")
                ], class_name="upload-row"),
            dcc.Upload(
                id="file-upload",
                className="upload",
                children=html.Div(
                    ["Drag and Drop or ", html.A("Select Files")]
                ),
                multiple=True),
            dcc.Store(id='dataset-name', data='DAEMONS'),
            dcc.Store(id='csv-data', data=None),
            dcc.Store(id='img-files', data=None),
        ], class_name="top"),
        dbc.Alert(id="alert-missing", color="danger", class_name="alert", is_open=False),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='figure-output',
                    figure=fig,
                    className="graph",
                    style={'width': '100%', 'height': 500}
                ),
                html.Div([
                    dcc.Slider(0, 1, 0.2,
                            value=0.8,
                            id="range-slider"
                    )
                ], id="hide-div", className="slider"),
                dcc.Store(id='fig-type', data=None)
            ], 
            width=7, class_name="graph"),
            dbc.Col([
                html.Div(id='llm-output', className="output"), 
                dcc.Loading(
                    id='loading-response',
                    type='dot',
                    show_initially=False,
                    children=[dcc.Store('messages', data=initial_messages)]
                ),
                dcc.Textarea(
                    id='llm-input',
                    value='Make a scan path plot for subject with id 4 and image 0456.jpg',
                    className="textarea"
                ),
                dbc.Button('Submit', id='llm-submit-button', n_clicks=0, class_name="submit")
            ], class_name="chat")
        ])
    ])
], fluid=True, className="bg")

@callback(
    Output('figure-output', 'figure', allow_duplicate=True),
    Input('range-slider', 'value'),
    State('fig-type', 'data'),
    prevent_initial_call=True
)
def update_opacity(value, fig_type):
    if fig_type == "histogram2d":
        patched_figure = Patch()
        patched_figure["data"][0]["opacity"] = value
        return patched_figure
    return no_update

@callback(
    Output('messages', 'data'),
    Output('llm-output', 'children'),
    Output('figure-output', 'figure'),
    Output('hide-div', 'style'),
    Output('fig-type', 'data'),
    Input('llm-submit-button', 'n_clicks'),
    State('llm-input', 'value'),
    State('dataset-name', 'data'),
    State('csv-data', 'data'),
    State('img-files', 'data'),
    State('messages', 'data'),
    prevent_initial_call=True
)
def update_output(n_clicks, value, dataset_name, csv_data, img_files, messages):
    chat_history = get_chat_history(messages)
    messages.append({"content": value, "role": "human"})
    ai_message = call_llm(messages)
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            fct_name = tool_call["name"]
            fixation_df = None
            img_files_dict = None
            if csv_data != None:
                fixation_df = pd.DataFrame.from_records(csv_data)
            if img_files != None:
                img_files_dict = json.loads(img_files)
            # inject dataset name
            arg_names = list(ptools.tools[fct_name].get_input_schema().model_json_schema()["properties"].keys())
            tool_call["args"]["dataset_name"] = dataset_name
            if "fixation_df" in arg_names:
                tool_call["args"]["fixation_df"] = fixation_df
            if "img_files" in arg_names:
                tool_call["args"]["img_files"] = img_files_dict
            try:
                tool_message = ptools.tools[fct_name].invoke(tool_call)
            except ValueError as exc:
                tool_message = {"content": {str(exc)}, "tool_call_id": n_clicks, "role": "tool"}
                print(exc)
                return messages, chat_history, no_update, no_update, no_update
            messages.append({"content": tool_message.content, "tool_call_id": n_clicks, "role": "tool"})
            print(tool_message.content)

            chat_history = get_chat_history(messages)
            if tool_message.artifact:
                fig_type = tool_message.artifact["data"][0]["type"]
                if fig_type == "histogram2d":
                    return messages, chat_history, tool_message.artifact, {"display": "block"}, fig_type
                return messages, chat_history, tool_message.artifact, {"display": "none"}, fig_type
            
            # no artifact returned by tool, pass content to LLM, without displaying it to the user
            ai_message = call_llm(messages)
            content = ai_message.content
            messages.append({"content": content, "role": "ai"})
            chat_history = get_chat_history(messages)
            return messages, chat_history, no_update, no_update, no_update

    content = ai_message.content
    messages.append({"content": content, "role": "ai"})
    chat_history = get_chat_history(messages)
    return messages, chat_history, no_update, no_update, no_update

def call_llm(messages: list):
    trim_messages(
            messages, 
            token_counter=len,
            strategy="last",
            max_tokens=30,
            include_system=True
        )
    response = llm.invoke(messages)
    print(response.content, response.tool_calls)
    return response

@callback(
    Output('dataset-name', 'data', allow_duplicate=True),
    Output('csv-data', 'data'),
    Output('img-files', 'data'),
    Output('alert-missing', 'is_open'),
    Output('alert-missing', 'children'),
    Output('dataset-dropdown', 'options'),
    Output('dataset-dropdown', 'value'),
    Input('file-upload', 'filename'),
    Input('file-upload', 'contents'),
    prevent_initial_call=True
)
def store_uploaded_dataset(filenames, contents):
    csv_data = None
    img_files = {}
    dataset_name = None
    for filename, content in zip(filenames, contents):
        if filename.endswith(".csv"):
            _, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            missing_columns = ptools.find_missing_columns(df)
            if missing_columns:
                missing_columns_str = ", ".join(missing_columns)
                return no_update, no_update, no_update, True, f"Uploaded csv needs columns: {missing_columns_str}", no_update, no_update
            csv_data = df.to_dict('records')
            dataset_name = filename
        if filename.endswith(".jpg"):
            img_files[filename] = content
    img_files_json = json.dumps(img_files)
    return dataset_name, csv_data, img_files_json, False, no_update, ["DAEMONS", dataset_name], dataset_name

@callback(
    Output('dataset-name', 'data', allow_duplicate=True),
    Input('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def update_selected_dataset(value):
    return value

def get_chat_history(messages: list) -> list[html.Div]:
    message_div = [
        html.Div(message["content"], className=f"msg {message['role']}")
        for message in messages if message["role"] != "system" and message["role"] != "tool"
        ]
    return message_div

if __name__ == '__main__':
    app.run(debug=True)
