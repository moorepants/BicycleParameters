import io
import base64
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as tbl
import time
from dash.dependencies import Input, Output, State

import bicycleparameters as bp
import matplotlib
matplotlib.use('Agg')

path_to_app_data = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'app-data')

pList = ['rF', 'mF', 'IFxx', 'IFyy', 'rR', 'mR', 'IRxx', 'IRyy',
         'w', 'c', 'lam', 'g',
         'xB', 'zB', 'mB', 'IBxx', 'IByy', 'IBzz', 'IBxz', 'xH', 'zH', 'mH', 'IHxx', 'IHyy', 'IHzz', 'IHxz']

OPTIONS = ['Benchmark',
           'Browser',
           'Browserins',
           'Crescendo',
           'Fisher',
           'Pista',
           'Rigid',
           'Silver',
           'Yellow',
           'Yellowrev']

WHEEL_COLUMNS = [{'name': '', 'id': 'label', 'type': 'text', 'editable': False},
                 {'name': 'Front Wheel [F]', 'id': 'fW', 'type': 'numeric'},
                 {'name': 'Rear Wheel [R]', 'id': 'rW', 'type': 'numeric'}]

WHEEL_LABELS = ['Radius [m]:',
                'Mass [kg]:',
                'Moment Ixx [kg*m²]:',
                'Moment Iyy [kg*m²]:']

FRAME_COLUMNS = [{'name': '', 'id': 'label', 'type': 'text', 'editable': False},
                 {'name': 'Rear Body [B]', 'id': 'rB', 'type': 'numeric'},
                 {'name': 'Front Assembly [H]', 'id': 'fA', 'type': 'numeric'}]

FRAME_LABELS = ['Center of Mass X [m]:',
                'Center of Mass Y [m]:',
                'Total Mass [kg]:',
                'Moment Ixx [kg*m²]:',
                'Moment Iyy [kg*m²]:',
                'Moment Izz [kg*m²]:',
                'Moment Ixz [kg*m²]:']

GENERAL_COLUMNS = [{'name': '', 'id': 'label', 'type': 'text', 'editable': False},
                   {'name': 'Parameters', 'id': 'con', 'type': 'numeric'}]

GENERAL_LABELS = ['Wheel Base [m]:',
                  'Trail [m]:',
                  'Steer Axis Tilt [degrees]:',
                  'Gravity [N/kg]:']

app = dash.Dash(__name__)
server = app.server  # needed for heroku

app.layout = html.Div([
    html.U(html.B(html.H1('Bicycle Dynamics Analysis App',
                          id='main-header'))),
    html.Div(id='dropdown-bin',
             children=[html.H2('Choose a Parameter Set:'),
                       dcc.Dropdown(id='bike-dropdown',
                                    value='Benchmark',
                                    options=[{'label': i, 'value': i} for i in OPTIONS])]),
    html.Div(id='plot-bin',
             children=[dcc.Loading(id='plot-load',
                                   type='dot',
                                   children=[html.Div(id='geometry-bin',
                                                      children=[html.Img(src='',
                                                                         id='geometry-plot')]),
                                             html.Div(id='eigen-bin',
                                                      children=[html.Img(src='',
                                                                         id='eigen-plot')])])]),
    html.Div(id='slider-bin',
             children=[html.H2('Set the EigenValue Speed Range:'),
                       dcc.RangeSlider(id='range-slider',
                                       min=-45,
                                       max=45,
                                       step=5,
                                       value=[0, 10],
                                       marks={-40: {'label': '-40 m/s', 'style': {'color': '#000000'}},
                                              -30: {'label': '-30 m/s', 'style': {'color': '#000000'}},
                                              -20: {'label': '-20 m/s', 'style': {'color': '#000000'}},
                                              -10: {'label': '-10 m/s', 'style': {'color': '#000000'}},
                                              0: {'label': '0 m/s', 'style': {'color': '#000000'}},
                                              10: {'label': '10 m/s', 'style': {'color': '#000000'}},
                                              20: {'label': '20 m/s', 'style': {'color': '#000000'}},
                                              30: {'label': '30 m/s', 'style': {'color': '#000000'}},
                                              40: {'label': '40 m/s', 'style': {'color': '#000000'}}},
                                       allowCross=False)]),
    html.Div(id='table-bin',
             children=[html.H2('Whipple-Carvallo Model Parameters'),
                       html.Button('Reset Table',
                                   id='reset-button',
                                   type='button',
                                   n_clicks=0),
                       tbl.DataTable(id='general-table',
                                     columns=GENERAL_COLUMNS,
                                     data=[],
                                     style_cell={'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                                                 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
                                     style_header={
                                         'textAlign': 'center', 'backgroundColor': 'rgb(30, 30, 30)'},
                                     editable=True),
                       tbl.DataTable(id='frame-table',
                                     columns=FRAME_COLUMNS,
                                     data=[],
                                     style_cell={'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                                                 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
                                     style_header={
                                         'textAlign': 'center', 'backgroundColor': 'rgb(30, 30, 30)'},
                                     editable=True),
                       tbl.DataTable(id='wheel-table',
                                     columns=WHEEL_COLUMNS,
                                     data=[],
                                     style_cell={'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                                                 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
                                     style_header={
                                         'textAlign': 'center', 'backgroundColor': 'rgb(30, 30, 30)'},
                                     editable=True)]),
    html.Div(id='version-bin',
             children=[html.Ul(children=[html.Li('BicycleParameters v{}'.format(bp.__version__)),
                                         html.Li('Dash v{}'.format(
                                             dash.__version__)),
                                         html.Li('NumPy v{}'.format(
                                             np.__version__)),
                                         html.Li('Pandas v{}'.format(
                                             pd.__version__)),
                                         html.Li('Matplolib v{}'.format(
                                             matplotlib.__version__)),
                                         ])]),
])

# creates generic set of Benchmark parameters


def new_par(bike_name):
    bike = bp.Bicycle(bike_name, pathToData=path_to_app_data)
    par = bike.parameters['Benchmark']
    parPure = bp.io.remove_uncertainties(par)
    return parPure

# populates all data tables with bicycleparameters data


@app.callback([Output('wheel-table', 'data'),
               Output('frame-table', 'data'),
               Output('general-table', 'data')],
              [Input('bike-dropdown', 'value'),
               Input('reset-button', 'n_clicks')])
def populate_wheel_data(value, n_clicks):
    # generates data for wheel-table
    wheelPure = new_par(value)
    wheelData = []
    wheelLabels = []
    fW = []
    rW = []
    for i in WHEEL_LABELS:
        wheelLabels.append({'label': i})
    for i in range(8):
        if i < 4:
            fW.append({'fW': wheelPure.get(pList[i])})
        else:
            rW.append({'rW': wheelPure.get(pList[i])})
    for c, d, e in zip(wheelLabels, fW, rW):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        zipped.update(e)
        wheelData.append(zipped)

    # generates data for frame-table
    framePure = new_par(value)
    frameData = []
    frameLabels = []
    rB = []
    fA = []
    for i in FRAME_LABELS:
        frameLabels.append({'label': i})
    for i in range(12, len(pList)):
        if i < 19:
            rB.append({'rB': framePure.get(pList[i])})
        else:
            fA.append({'fA': framePure.get(pList[i])})
    for c, d, e in zip(frameLabels, rB, fA):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        zipped.update(e)
        frameData.append(zipped)

    # generates data for general-table
    genPure = new_par(value)
    genData = []
    genLabels = []
    con = []
    for i in GENERAL_LABELS:
        genLabels.append({'label': i})
    for i in range(8, 12):
        con.append({'con': genPure.get(pList[i])})

    # converts radians to degrees for display in the table
    lamD = con[2]
    radians = lamD.get('con')
    degrees = np.rad2deg(radians)
    con[2] = {'con': int(degrees)}

    for c, d in zip(genLabels, con):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        genData.append(zipped)

    return wheelData, frameData, genData

    # updates geometry-plot & eigen-plot path with Dropdown value or edited DataTable values


@app.callback([Output('geometry-plot', 'src'),
               Output('eigen-plot', 'src')],
              [Input('bike-dropdown', 'value'),
               Input('wheel-table', 'data'),
               Input('frame-table', 'data'),
               Input('general-table', 'data'),
               Input('range-slider', 'value')])
def plot_update(value, wheel, frame, general, slider):
    # accesses Input properties to avoid redundancies
    ctx = dash.callback_context
    wheelData = ctx.inputs.get('wheel-table.data')
    frameData = ctx.inputs.get('frame-table.data')
    genData = ctx.inputs.get('general-table.data')
    rangeSliderData = ctx.inputs.get('range-slider.value')

    # sets the speed range for eigen-plot based on range-slider
    minBound = rangeSliderData[0]
    maxBound = rangeSliderData[1]
    steps = (maxBound-minBound)/0.1
    speeds = np.linspace(minBound, maxBound, num=int(steps))

    # Case 1: Recieves direct bicycle data if bicycle is selected from the dropdown menu
    if ctx.triggered[0].get('prop_id') == 'bike-dropdown.value':

        dropdownBike = bp.Bicycle(value, pathToData=path_to_app_data)

        # create geometry-plot image
        geo_fake = io.BytesIO()
        geo_plot = dropdownBike.plot_bicycle_geometry(show=False)
        geo_plot.savefig(geo_fake)
        geo_image = base64.b64encode(geo_fake.getvalue())
        plt.close(geo_plot)

        # create eigen-plot image
        eigen_fake = io.BytesIO()
        eigen_plot = dropdownBike.plot_eigenvalues_vs_speed(
            speeds, show=False, grid=True, showLegend=False)
        eigen_plot.savefig(eigen_fake)
        eigen_image = base64.b64encode(eigen_fake.getvalue())
        plt.close(eigen_plot)

        return 'data:image/png;base64,{}'.format(geo_image.decode()), 'data:image/png;base64,{}'.format(eigen_image.decode())

    # Case 2: Recieves values from the displayed table in every other case
    else:

        # convert to radians
        if ctx.triggered[0].get('prop_id') == 'wheel-table.data' or ctx.triggered[0].get('prop_id') == 'frame-table.data' or ctx.triggered[0].get('prop_id') == 'general-table.data' or ctx.triggered[0].get('prop_id') == 'range-slider.value':
            degrees = float(genData[2].get('con'))
            radians = np.deg2rad(degrees)
            genData[2]['con'] = radians

        # creates an alternating list of [parameter,value] from table data
        newP = []
        for p in range(8):
            if p < 4:
                newP.extend([pList[p], wheelData[p].get('fW')])
            else:
                newP.extend([pList[p], wheelData[p-4].get('rW')])
        for p in range(12, len(pList)):
            if p < 19:
                newP.extend([pList[p], frameData[p-12].get('rB')])
            else:
                newP.extend([pList[p], frameData[p-19].get('fA')])
        for p in range(8, 12):
            newP.extend([pList[p], genData[p-8].get('con')])

        # edits bicycle parameters based on table data
        currentBike = bp.Bicycle(value, pathToData=path_to_app_data)
        for i in range(0, len(newP), 2):
            currentBike.parameters['Benchmark'][newP[i]] = newP[i+1]

        # create geometry-plot image
        geo_fake = io.BytesIO()
        geo_plot = currentBike.plot_bicycle_geometry(show=False)
        geo_plot.savefig(geo_fake)
        geo_image = base64.b64encode(geo_fake.getvalue())
        plt.close(geo_plot)

        # create eigen-plot image
        eigen_fake = io.BytesIO()
        eigen_plot = currentBike.plot_eigenvalues_vs_speed(
            speeds, show=False, grid=True, showLegend=False)
        eigen_plot.savefig(eigen_fake)
        eigen_image = base64.b64encode(eigen_fake.getvalue())
        plt.close(eigen_plot)

        return 'data:image/png;base64,{}'.format(geo_image.decode()), 'data:image/png;base64,{}'.format(eigen_image.decode())

    # sets loading notification for the plots


@app.callback(Output("plot-load", "children"), [Input("geometry-bin", "children"), Input('eigen-bin','children')])
def input_triggers_spinner(value):
    time.sleep(1)
    return value


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)
