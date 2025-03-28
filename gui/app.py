# gui/app.py
import os
import sys

# Add the parent directory (QoS_Gemini) to the Python path
# BEFORE importing from 'sim'
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import copy  # Use deepcopy for config

# Import simulation components
from sim.simulation import Simulation
from sim.config import get_default_config, bps_to_bytes_per_step
from sim.service_class import ServiceClass, PRIORITY_ORDER

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], suppress_callback_exceptions=True)
app.title = "Satellite QoS Simulation"

# --- Helper Functions ---
def create_gauge_figure(value, title, range_max=100, suffix='%'):
    """Creates a Plotly gauge figure."""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18}},
        gauge={'axis': {'range': [0, range_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
               'bar': {'color': "cornflowerblue"},
               'steps': [
                   {'range': [0, range_max * 0.5], 'color': 'lightgreen'},
                   {'range': [range_max * 0.5, range_max * 0.8], 'color': 'yellow'},
                   {'range': [range_max * 0.8, range_max], 'color': 'red'}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': range_max * 0.9} # Example threshold
               },
        number={'suffix': suffix, 'font': {'size': 24}}
    ))

def create_empty_figure(title="Run simulation to see results"):
     fig = go.Figure()
     fig.update_layout(
         xaxis =  { "visible": False },
         yaxis = { "visible": False },
         annotations = [
             {   "text": title,
                 "xref": "paper",
                 "yref": "paper",
                 "showarrow": False,
                 "font": { "size": 20 }
             }
         ]
     )
     return fig

def convert_numpy_types(obj):
    """Recursively converts NumPy types in a dictionary/list to standard Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, np.integer): # Catches int64, int32, etc.
        return int(obj)
    elif isinstance(obj, np.floating): # Catches float64, float32, etc.
        return float(obj)
    elif isinstance(obj, np.ndarray): # Convert numpy arrays to lists
         return obj.tolist()
    else:
        return obj

# --- App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Satellite Return Link QoS Simulation"), width=12), className="mb-4 mt-4"),

    dbc.Row([
        # --- Configuration Column ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Parameters"),
                dbc.CardBody([
                    dbc.Label("Simulation Duration (seconds):"),
                    dbc.Input(id="sim-duration", type="number", value=get_default_config()["simulation_duration_sec"], min=1, step=1),
                    dbc.Label("Time Step (seconds):", className="mt-2"),
                    dbc.Input(id="sim-timestep", type="number", value=get_default_config()["time_step_sec"], min=0.01, step=0.01, max=1.0),

                    html.Hr(),
                    dbc.Label("Packet Generation Rate (packets/sec):"),
                    dbc.Input(id="packet-rate", type="number", value=get_default_config()["ut_config"]["packet_rate_pps"], min=0),
                    dbc.Label("Avg. Packet Size (bytes):", className="mt-2"),
                    dbc.Input(id="packet-size-avg", type="number", value=get_default_config()["ut_config"]["packet_size_bytes_avg"], min=50),
                    dbc.Label("Packet Size Std Dev (bytes):", className="mt-2"),
                    dbc.Input(id="packet-size-std", type="number", value=get_default_config()["ut_config"]["packet_size_bytes_stddev"], min=0),
                    dbc.Label("Max Queue Size (packets per class):", className="mt-2"),
                    dbc.Input(id="max-queue-size", type="number", value=get_default_config()["ut_config"]["max_queue_size_packets"], min=10),


                    html.Hr(),
                    dbc.Label("Bandwidth Type:"),
                    dbc.RadioItems(
                        options=[
                            {"label": "Static", "value": "static"},
                            {"label": "Dynamic (Sine Wave)", "value": "dynamic"},
                        ],
                        value=get_default_config()["bandwidth_type"],
                        id="bw-type",
                        inline=True,
                    ),
                    # Static BW Input
                    html.Div([
                        dbc.Label("Static Bandwidth (Kbps):"),
                        dbc.Input(id="static-bw", type="number", value=get_default_config()["static_bandwidth_kbps"], min=1),
                    ], id='static-bw-div'),
                    # Dynamic BW Inputs
                    html.Div([
                         dbc.Label("Base Bandwidth (Kbps):"),
                         dbc.Input(id="dyn-bw-base", type="number", value=get_default_config()["dynamic_bw_config"]["base_kbps"], min=1),
                         dbc.Label("Amplitude (Kbps):", className="mt-2"),
                         dbc.Input(id="dyn-bw-amp", type="number", value=get_default_config()["dynamic_bw_config"]["amplitude_kbps"], min=0),
                         dbc.Label("Period (seconds):", className="mt-2"),
                         dbc.Input(id="dyn-bw-period", type="number", value=get_default_config()["dynamic_bw_config"]["period_sec"], min=1),
                    ], id='dynamic-bw-div', style={'display': 'none'}), # Initially hidden


                    html.Hr(),
                    dbc.Label("QoS Thresholds (% Guaranteed Bandwidth):"),
                    dbc.Row([
                        dbc.Col(dbc.Label("NC (%):"), width=3),
                        dbc.Col(dbc.Input(id="thresh-nc", type="number", value=get_default_config()["qos_thresholds_percent"][ServiceClass.NETWORK_CONTROL], min=0, max=100, step=1)),
                    ]),
                     dbc.Row([
                        dbc.Col(dbc.Label("EF (%):"), width=3),
                        dbc.Col(dbc.Input(id="thresh-ef", type="number", value=get_default_config()["qos_thresholds_percent"][ServiceClass.EXPEDITED_FORWARDING], min=0, max=100, step=1)),
                    ], className="mt-1"),
                     dbc.Row([
                        dbc.Col(dbc.Label("AF (%):"), width=3),
                        dbc.Col(dbc.Input(id="thresh-af", type="number", value=get_default_config()["qos_thresholds_percent"][ServiceClass.ASSURED_FORWARDING], min=0, max=100, step=1)),
                    ], className="mt-1"),
                    dbc.Row([
                         dbc.Col(dbc.Label("BE (%):"), width=3),
                         dbc.Col(html.Div(id="thresh-be-display", children="Remaining", style={'padding-top': '8px'})), # Display only
                    ]),
                    dbc.FormText(id="thresh-total-text", children="Total: X%", color="secondary"),

                    html.Hr(),
                    dbc.Button("Run Simulation", id="run-button", color="primary", n_clicks=0, className="mt-3"),
                    dbc.Spinner(html.Div(id="spinner-output"), color="primary"), # Show spinner while running
                ])
            ])
        ], width=4), # End Configuration Column

        # --- Results Column ---
        dbc.Col([
             dbc.Row([
                 dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='gauge-bw-util', figure=create_empty_figure("Avg BW Util")))), width=4),
                 dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='gauge-avg-latency-ef', figure=create_empty_figure("Avg EF Latency")))), width=4),
                 dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id='gauge-drop-rate-be', figure=create_empty_figure("BE Drop Rate")))), width=4),
             ], className="mb-3"),

             dbc.Card([
                 dbc.CardHeader("Simulation Results"),
                 dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(label="Queue Lengths", children=[
                            dcc.Graph(id="queue-length-plot", figure=create_empty_figure())
                        ]),
                        dbc.Tab(label="Transmitted Data", children=[
                             dcc.Graph(id="transmitted-plot", figure=create_empty_figure())
                        ]),
                         dbc.Tab(label="Bandwidth", children=[
                             dcc.Graph(id="bandwidth-plot", figure=create_empty_figure())
                        ]),
                        dbc.Tab(label="Summary Stats", children=[
                            html.Pre(id="summary-stats-output", children="Run simulation to see summary stats.", style={'fontSize': '12px', 'padding': '10px', 'background-color': '#f8f9fa'})
                        ]),
                    ])
                 ])
            ])
        ], width=8) # End Results Column
    ])
], fluid=True)

# --- Callbacks ---

# Callback to toggle visibility of bandwidth input fields
@app.callback(
    Output('static-bw-div', 'style'),
    Output('dynamic-bw-div', 'style'),
    Input('bw-type', 'value')
)
def toggle_bandwidth_inputs(bw_type):
    if bw_type == 'static':
        return {'display': 'block'}, {'display': 'none'}
    elif bw_type == 'dynamic':
        return {'display': 'none'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'} # Default hide both if needed

# Callback to update the displayed total threshold percentage
@app.callback(
    Output('thresh-total-text', 'children'),
    Output('thresh-be-display', 'children'),
    Input('thresh-nc', 'value'),
    Input('thresh-ef', 'value'),
    Input('thresh-af', 'value')
)
def update_threshold_total(nc, ef, af):
    total = (nc or 0) + (ef or 0) + (af or 0)
    be_perc = max(0, 100 - total)
    color = "danger" if total > 100 else "secondary"
    return f"Guaranteed Total: {total}%", f"{be_perc}% (Remaining)"


# Main callback to run simulation and update all outputs
@app.callback(
    Output('spinner-output', 'children'), # Just to trigger spinner
    Output('queue-length-plot', 'figure'),
    Output('transmitted-plot', 'figure'),
    Output('bandwidth-plot', 'figure'),
    Output('summary-stats-output', 'children'),
    Output('gauge-bw-util', 'figure'),
    Output('gauge-avg-latency-ef', 'figure'),
    Output('gauge-drop-rate-be', 'figure'),
    Input('run-button', 'n_clicks'),
    State('sim-duration', 'value'),
    State('sim-timestep', 'value'),
    State('packet-rate', 'value'),
    State('packet-size-avg', 'value'),
    State('packet-size-std', 'value'),
    State('max-queue-size', 'value'),
    State('bw-type', 'value'),
    State('static-bw', 'value'),
    State('dyn-bw-base', 'value'),
    State('dyn-bw-amp', 'value'),
    State('dyn-bw-period', 'value'),
    State('thresh-nc', 'value'),
    State('thresh-ef', 'value'),
    State('thresh-af', 'value'),
    prevent_initial_call=True # Don't run on page load
)
def run_and_update(n_clicks, duration, timestep, rate, size_avg, size_std, max_q,
                   bw_type, static_bw, dyn_base, dyn_amp, dyn_period,
                   thresh_nc, thresh_ef, thresh_af):

    if n_clicks == 0:
        # Should be prevented by prevent_initial_call, but good practice
        return dash.no_update

    # --- 1. Create Config from Inputs ---
    config = get_default_config() # Start with defaults
    config["simulation_duration_sec"] = float(duration)
    config["time_step_sec"] = float(timestep)

    config["ut_config"]["packet_rate_pps"] = float(rate)
    config["ut_config"]["packet_size_bytes_avg"] = int(size_avg)
    config["ut_config"]["packet_size_bytes_stddev"] = int(size_std)
    config["ut_config"]["max_queue_size_packets"] = int(max_q)
    # Note: Service class distribution is hardcoded in default config for now,
    # could be added as inputs if needed.

    config["bandwidth_type"] = bw_type
    if bw_type == "static":
        config["static_bandwidth_kbps"] = float(static_bw)
    else: # dynamic
        config["dynamic_bw_config"]["base_kbps"] = float(dyn_base)
        config["dynamic_bw_config"]["amplitude_kbps"] = float(dyn_amp)
        config["dynamic_bw_config"]["period_sec"] = float(dyn_period)

    config["qos_thresholds_percent"] = {
        ServiceClass.NETWORK_CONTROL: float(thresh_nc or 0),
        ServiceClass.EXPEDITED_FORWARDING: float(thresh_ef or 0),
        ServiceClass.ASSURED_FORWARDING: float(thresh_af or 0),
        # BE is handled implicitly by the scheduler
    }

    # --- 2. Run Simulation ---
    sim = Simulation(copy.deepcopy(config)) # Use deepcopy to avoid modifying default
    try:
        sim.run_simulation()
        results_df = sim.get_results_dataframe()
        summary_stats = sim.get_summary_stats()
    except Exception as e:
        print(f"Error during simulation: {e}")
        # Return error messages or empty figures
        error_fig = create_empty_figure(f"Simulation Error: {e}")
        error_summary = f"Simulation failed:\n{e}"
        empty_gauge = create_empty_figure("Error")
        return "", error_fig, error_fig, error_fig, error_summary, empty_gauge, empty_gauge, empty_gauge


    # --- 3. Create Figures and Outputs ---
    if results_df.empty:
         return "", create_empty_figure(), create_empty_figure(), create_empty_figure(), "No results generated.", create_empty_figure(), create_empty_figure(), create_empty_figure()

    cleaned_summary_stats= convert_numpy_types(summary_stats)


    # Queue Length Plot (Packets)
    fig_queues = go.Figure()
    for sc in PRIORITY_ORDER:
        fig_queues.add_trace(go.Scatter(x=results_df['time'], y=results_df[f'queue_pkts_{sc.label}'],
                                       mode='lines', name=f'{sc.label} Queue (Pkts)'))
    fig_queues.update_layout(title="Queue Lengths Over Time (Packets)", xaxis_title="Time (s)", yaxis_title="Packets in Queue")

    # Transmitted Data Plot (Bytes per Time Step - Stacked Bar)
    fig_tx = go.Figure()
    colors = {'NC': 'red', 'EF': 'orange', 'AF': 'blue', 'BE': 'green'}
    for sc in PRIORITY_ORDER:
        fig_tx.add_trace(go.Bar(x=results_df['time'], y=results_df[f'tx_bytes_{sc.label}'],
                                name=f'{sc.label} Tx Bytes', marker_color=colors.get(sc.label)))
    fig_tx.update_layout(barmode='stack', title="Transmitted Bytes per Time Step",
                         xaxis_title="Time (s)", yaxis_title="Bytes Transmitted")

    # Bandwidth Plot
    fig_bw = go.Figure()
    fig_bw.add_trace(go.Scatter(x=results_df['time'], y=results_df['available_bw_bytes'],
                                mode='lines', name='Available BW (Bytes)', line=dict(dash='dot')))
    fig_bw.add_trace(go.Scatter(x=results_df['time'], y=results_df['total_transmitted_bytes'],
                                mode='lines', name='Used BW (Bytes)', fill='tozeroy'))
    fig_bw.update_layout(title="Bandwidth Availability and Usage", xaxis_title="Time (s)", yaxis_title="Bytes per Time Step")


    # Summary Stats Text
    import json
    summary_text = json.dumps(cleaned_summary_stats, indent=2)

    # Gauge Figures
    avg_bw_util = summary_stats.get('avg_bw_utilization_percent', 0)
    gauge_bw = create_gauge_figure(avg_bw_util, "Avg BW Util", range_max=100, suffix='%')

    avg_lat_ef = summary_stats.get('avg_latency_ms', {}).get('EF', 0)
    gauge_lat_ef = create_gauge_figure(avg_lat_ef, "Avg EF Latency", range_max=max(50, avg_lat_ef * 1.5), suffix=' ms') # Dynamic range

    # Calculate BE Drop Rate (%)
    total_gen_be = summary_stats.get('generated_approx', {}).get('BE', 0)
    total_drop_be = summary_stats.get('dropped', {}).get('BE', 0)
    be_drop_rate = (total_drop_be * 100.0 / total_gen_be) if total_gen_be > 0 else 0
    gauge_drop_be = create_gauge_figure(be_drop_rate, "BE Drop Rate", range_max=100, suffix='%')


    # --- 4. Return All Outputs ---
    # The empty string "" corresponds to the 'spinner-output' children - hides spinner
    return "", fig_queues, fig_tx, fig_bw, summary_text, gauge_bw, gauge_lat_ef, gauge_drop_be


# --- Run the App ---
if __name__ == '__main__':
    # Make sure sim directory is discoverable
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    app.run_server(debug=True) # Use debug=False for production