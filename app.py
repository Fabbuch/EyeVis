# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import plotting_tools as ptools
from dash import Dash, html, dcc, Input, Output, State, callback, Patch
import dash_bootstrap_components as dbc
import os

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

assets_path = os.getcwd() +'/datasets'

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, assets_folder=assets_path)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434"
).bind_tools(list(ptools.tools.values()))

# initialize message history:
messages = [
    SystemMessage(
        "You are a data analysis assistant for eye-tracking data. " \
        "Use the provided tools to answer the user's questions about the dataset or generate visualisations. " \
        "Only call one tool at a time."
    )
]

# make a scan path plot for subject with id 2 and image 000000005600.jpg
# content, fig = ptools.tool_scan_path_plot([4, 6], "DAEMONS_corpus_potsdam_0456.jpg")
# content, fig = ptools.tool_fixation_heat_map([6, 1], "DAEMONS_corpus_potsdam_1092.jpg")
# content, fig = ptools.tool_scan_path_plot([1,2], "000000275791.jpg")
# content, fig = ptools.tool_fixation_heat_map([9,2,10], "000000252771.jpg)
# content, fig = ptools.tool_fixation_heat_map([9], "000000252771.jpg")
fig = None

app.layout = dbc.Container([
    html.Div(children=[
        dbc.Row([
            html.Div(children='''
                Dash: A web application framework for your data.
            ''', 
            style={
                'textAlign': 'center'
            })
        ]),
        # dbc.Col([
        #         dcc.Graph(
        #             id='test-fig',
        #             figure=fig,
        #             style={'width': '100%', 'height': 500}
        #         )
        #     ], width=7),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='figure-output',
                    figure=fig,
                    style={'width': '100%', 'height': 500}
                ),
                html.Div([
                    dcc.Slider(0, 1, 0.2,
                            value=1,
                            id="range-slider"
                    ),
                ], id="hide-div", style= {'display': 'none'})
            ], 
            width=7),
            dbc.Col([
                html.Div(id='llm-output', style={'whiteSpace': 'pre-line'}),
                dcc.Textarea(
                    id='llm-input',
                    value='Make a scan path plot for subject with id 4 and image DAEMONS_corpus_potsdam_0456.jpg',
                    style={'width': '100%', 'height': 100},
                ),
                html.Button('Submit', id='llm-submit-button', n_clicks=0)
            ])
        ])
    ])
], fluid=True)

@callback(
    Output('figure-output', 'figure', allow_duplicate=True),
    Input('range-slider', 'value'),
    State('figure-output', 'figure'),
    prevent_initial_call=True
)
def update_opacity(value, initial_figure):
    if initial_figure != None and initial_figure["data"][0]["type"] == "histogram2d":
        patched_figure = Patch()
        patched_figure["data"][0]["opacity"] = value
        return patched_figure
    return initial_figure

@callback(
    Output('llm-output', 'children'),
    Output('figure-output', 'figure'),
    Output('hide-div', 'style'),
    Input('llm-submit-button', 'n_clicks'),
    State('llm-input', 'value'),
    State('figure-output', 'figure'),
    prevent_initial_call=True
)
def update_output(n_clicks, value, initial_figure):
    messages.append(HumanMessage(value))
    ai_message = call_llm()
    print(ai_message.content, ai_message.tool_calls)
    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        fct_name = tool_call["name"]
        try:
            tool_message = ptools.tools[fct_name].invoke(tool_call)
        except ValueError as exc:
            tool_message = ToolMessage({str(exc)}, tool_call_id=n_clicks)
        messages.append(ToolMessage(tool_message.content, tool_call_id=n_clicks))

        message_str = get_message_string()
        if tool_message.artifact:
            if tool_message.artifact["data"][0]["type"] == "histogram2d":
                return message_str, tool_message.artifact, {"display": "block"}
            return message_str, tool_message.artifact, {"display": "none"}

    content = ai_message.content
    messages.append(AIMessage(content))
    message_str = get_message_string()
    return message_str, initial_figure, {"display": "none"}

def call_llm():
    trim_messages(
            messages, 
            token_counter=len,
            strategy="last",
            max_tokens=30,
            include_system=True
        )
    response = llm.invoke(messages)
    return response

def get_message_string():
    message_str = "\n".join([
        message.content for message in messages 
        if message.type == "human" or message.type == "ai" or message.type == "tool"
    ])
    return message_str

if __name__ == '__main__':
    app.run(debug=True)
