import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import os
import re
import ast
import numpy as np

def extract_multiline_array(lines, start_index):
    collected = []
    for line in lines[start_index:]:
        line = line.strip()

        # Removing 'True MET' or 'Corrected MET' as headers from the string
        if 'True MET:' in line or 'Corrected MET:' in line:
            line = line.split(':', 1)[1].strip()

        # Breakes when finds other section
        if re.match(r'^[A-Za-z ]+:', line) and not ('Corrected MET' in line or 'True MET' in line):
            break
        collected.append(line)

    # Joins the collected data and cleans
    full_str = ' '.join(collected).replace('[', '').replace(']', '')
    
    # Debug the collected MET string
    print(f"[DEBUG] Full MET string: {full_str[:100]}")  # Print first 100 characters for inspection

    # Return the parsed data
    return np.fromstring(full_str, sep=' ')



"""
    Parsing function
"""
def parse_txt_file(file_path):
    data = {
        'METcorr': {},
        'Best Parameters': {},
        'Evaluation Metrics': {},
        'Features Importance': {},
        'Selected Features': [],
        'RMSE History': [],
        'Corrected MET': [],
        'True MET': []
    }

    rmse_pattern = re.compile(r"\[(\d+)]\s+validation_0-rmse:([0-9.]+)")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        match = rmse_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            rmse = float(match.group(2))
            data['RMSE History'].append((epoch, rmse))

    for line in lines:
        if 'METcorr MIN' in line:
            data['METcorr']['MIN'] = round(float(line.split(':')[-1].strip()), 2)
        elif 'METcorr MAX' in line:
            data['METcorr']['MAX'] = round(float(line.split(':')[-1].strip()), 2)
        elif 'METcorr MEAN' in line:
            data['METcorr']['MEAN'] = round(float(line.split(':')[-1].strip()), 2)
        elif 'METcorr MEDIAN' in line:
            data['METcorr']['MEDIAN'] = round(float(line.split(':')[-1].strip()), 2)

    for i, line in enumerate(lines):
        if 'Best Parameters Found' in line:
            params_line = lines[i + 1]
            data['Best Parameters'] = ast.literal_eval(params_line.strip())
        
        elif 'Evaluation metrics for the model' in line:
            for j in range(1, 4):
                metric_line = lines[i + j]
                key, value = metric_line.strip().split(':')
                data['Evaluation Metrics'][key.strip()] = round(float(value.strip()), 2)
        
        elif 'Features importance dictionary' in line:
            dict_line = lines[i + 1]
            data['Features Importance'] = ast.literal_eval(dict_line.strip())
        
        elif 'Selected features' in line:
            features_line = line.split(':', 1)[-1].strip()
            data['Selected Features'] = ast.literal_eval(features_line)
        
        elif 'Corrected MET:' in line:
            data['Corrected MET'] = extract_multiline_array(lines, i)

        elif 'True MET:' in line:
            data['True MET'] = extract_multiline_array(lines, i)

    print(data['True MET'].size)

    def sample_data(fraction):
         size = int(len(data['True MET']) * fraction)
         indices = np.random.choice(len(data['True MET']), size=size, replace=False)
         return data['True MET'][indices], data['Corrected MET'][indices]

    data['sample_25'] = sample_data(0.25)
    data['sample_50'] = sample_data(0.5)
    return data



"""
    Creating Dash App
"""
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "MET resolution - ML model"

file_list = [f for f in os.listdir('utils') if f.endswith('.txt')]

app.layout = html.Div(style={'backgroundColor': '#1D1D1F', 'color': '#FFFFFF', 'padding': '30px'}, children=[
    html.H1("ML model for MET resolution", style={'textAlign': 'center', 'color': '#00CED1'}),

    dcc.Dropdown(
        id='dataset-selector',
        className = 'custom-dropdown',
        options=[
            {'label': f, 'value': f} for f in os.listdir('utils') if f.endswith('.txt')
        ],
        value = file_list[0] if file_list else None
    ),

    html.Div([
                html.Label("Select data percentage to plot:", style={'color':'#bba8a8', 'marginTop': '20px'}),
                dcc.Dropdown(
                    id='scatter-sample-selector',
                    options=[
                        {'label': '25%', 'value': 0.25},
                        {'label': '50%', 'value': 0.5},
                        {'label': '100%', 'value': 1.0},
                    ],
                    value=0.25,
                    style={'width': '150px'}
                )
            ]),

    html.Div(id='content')
])

@app.callback(
    Output('content', 'children'),
    [Input('dataset-selector', 'value'),
    Input('scatter-sample-selector', 'value')]  
)

def update_dashboard(filename, sample_fraction):
    file_path = os.path.join('utils', filename)
    data = parse_txt_file(file_path)

    if sample_fraction == 0.25:
        x_vals, y_vals = data['sample_25']
    elif sample_fraction == 0.5:
        x_vals, y_vals = data['sample_50']
    else:
        x_vals = data['True MET']
        y_vals = data['Corrected MET']

    return html.Div([
        # First row: MET correction + Evaluation Metrics | Best Parameters
        html.Div([
            html.Div([
                html.H2("📌 MET Correction"),
                html.Ul([
                    html.Li(f"{k}: {v}", style={'marginBottom': '5px'}) 
                    for k, v in data['METcorr'].items()
                ]),
                html.H2("📈 Evaluation Metrics", style={'marginTop': '30px'}),
                html.Ul([
                    html.Li(f"{k}: {v}", style={'marginBottom': '5px'}) 
                    for k, v in data['Evaluation Metrics'].items()
                ])
            ], style={'flex': '1', 'marginRight': '20px'}),

            html.Div([
                html.H2("🔧 Best Hyper Parameters"),
                html.Ul([
                    html.Li(f"{k}: {v}", style={'marginBottom': '5px'}) 
                    for k, v in data['Best Parameters'].items()
                ])
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'marginTop': '30px'}),

        # Features Importance
        html.Div([
            html.H2("📊 Features Importance"),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Bar(
                        x=list(data['Features Importance'].keys()),
                        y=list(data['Features Importance'].values()),
                        marker=dict(color='#6BC9FF')
                    )],
                    layout=go.Layout(
                        template='plotly_dark',
                        title='Features Importance',
                        xaxis=dict(title='Features'),
                        yaxis=dict(title='Importance'),
                        height=400
                    )
                ),
                config={'displayModeBar': False}
            )
        ]),

        # RMSE History
        html.Div([
            html.H2("📉 RMSE Per Era"),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Scatter(
                        x=[epoch for epoch, _ in data['RMSE History']],
                        y=[rmse for _, rmse in data['RMSE History']],
                        mode='markers',
                        marker=dict(color='#6BC9FF'),
                        name='RMSE'
                    )],
                    layout=go.Layout(
                        template='plotly_dark',
                        title='RMSE Trend',
                        xaxis=dict(title='Eras'),
                        yaxis=dict(title='RMSE'),
                        height=400
                    )
                ),
                config={'displayModeBar': False}
            )
        ]),
        
        # Corrected VS True MET Scatter Plot
        html.Div([
            html.H2("🎯 Corrected vs True MET"),

            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='markers',
                            marker=dict(size=3, color='#6BC9FF', opacity=0.7),
                            name='Corrected vs True MET'
                        ),
                        go.Scatter(
                            x=[min(x_vals), max(x_vals)],
                            y=[min(x_vals), max(x_vals)],
                            mode='lines',
                            line=dict(color='lightgray', dash='dash'),
                            name='Ideal Prediction (y = x)'
                        )
                    ],
                    layout=go.Layout(
                        template='plotly_dark',
                        title='Corrected vs True MET',
                        xaxis=dict(title='True MET'),
                        yaxis=dict(title='Corrected MET'),
                        height=400
                    )
                ),
                config={'displayModeBar': False}
            )
        ])

    ])

"""
    Starting the App
"""
if __name__ == '__main__':
    app.run(debug=True)
