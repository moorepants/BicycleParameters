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

WHEEL_COLUMNS = [{'name': '', 'id': 'label', 'type': 'text'},
                 {'name': 'Front Wheel', 'id': 'fW', 'type': 'numeric'},
                 {'name': 'Rear Wheel', 'id': 'rW', 'type': 'numeric'}]

WHEEL_LABELS = ['Radius',
                'Mass',
                'Moment Ixx',
                'Moment Iyy']

FRAME_COLUMNS = [{'name': '', 'id': 'label', 'type': 'text'},
                 {'name': 'Rear Body', 'id': 'rB', 'type': 'numeric'},
                 {'name': 'Front Assembly', 'id': 'fA', 'type': 'numeric'}]

FRAME_LABELS = ['Center-of-Mass X',
                'Center-of-Mass Y',
                'Total Mass',
                'Moment Ixx',
                'Moment Iyy',
                'Moment Izz',
                'Moment Ixz']

GENERAL_COLUMNS = [{'name': '', 'id': 'label', 'type': 'text'},
                   {'name': 'Contextual Parameters', 'id': 'con', 'type': 'numeric'}]

GENERAL_LABELS = ['Wheel Base',
                  'Trail',
                  'Steer Axis Tilt',
                  'Gravity']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('BicycleParameters Web Application',
            id='main-header'),
    dcc.Dropdown(id='bike-dropdown',
                 value='Benchmark',
                 options=[{'label': i, 'value': i} for i in OPTIONS]),
    html.Div(id='plot-bin',
             children=[html.Img(src='',
                                alt='May take a moment to load...',
                                id='geometry-plot'),
                       html.Img(src='',
                                alt='May take a moment to load...',
                                id='eigen-plot')]),
    html.Div(id='slider-bin',
             children=[dcc.RangeSlider(id='range-slider',
                                       min=-45,
                                       max=45,
                                       step=5,
                                       value=[0, 10],
                                       marks={-40: '-40 m/s',
                                              -30: '-30 m/s',
                                              -20: '-20 m/s',
                                              -10: '-10 m/s',
                                              0: '0 m/s',
                                              10: '10 m/s',
                                              20: '20 m/s',
                                              30: '30 m/s',
                                              40: '40 m/s'},
                                       allowCross=False)]),
    html.Div(id='table-bin',
             children=[tbl.DataTable(id='wheel-table',
                                     columns=WHEEL_COLUMNS,
                                     data=[],
                                     style_cell={},
                                     style_header={},
                                     editable=True),
                       tbl.DataTable(id='frame-table',
                                     columns=FRAME_COLUMNS,
                                     data=[],
                                     style_cell={},
                                     style_header={},
                                     editable=True),
                       tbl.DataTable(id='general-table',
                                     columns=GENERAL_COLUMNS,
                                     data=[],
                                     style_cell={},
                                     style_header={},
                                     editable=True)]),
    html.Button('Reset Table',
                id='reset-button',
                type='button',
                n_clicks=0),
    html.Button('Toggle Dark Mode',
                id='dark-toggle',
                type='button',
                n_clicks=0),

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
    for c, d in zip(genLabels, con):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        genData.append(zipped)

    return wheelData, frameData, genData

    # updates geo-plot & eigen-plot path with Dropdown value or edited DataTable values


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

    # creates an alternating list of [parameter,value] from table data
    newP = []
    for p in range(8):
        if p < 4:
            # index out of range error, but print(newP) yields expected result?
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

    # create geo-plot image
    geo_fake = io.BytesIO()
    geo_plot = currentBike.plot_bicycle_geometry(show=False)
    geo_plot.savefig(geo_fake)
    geo_image = base64.b64encode(geo_fake.getvalue())
    plt.close(geo_plot)

    # create eigen-plot image
    eigen_fake = io.BytesIO()
    eigen_plot = currentBike.plot_eigenvalues_vs_speed(speeds, show=False)
    eigen_plot.savefig(eigen_fake)
    eigen_image = base64.b64encode(eigen_fake.getvalue())
    plt.close(eigen_plot)

    return 'data:image/png;base64,{}'.format(geo_image.decode()), 'data:image/png;base64,{}'.format(eigen_image.decode())

    # toggles dark mode for cells of DataTable


@app.callback([Output('wheel-table', 'style_cell'),
               Output('frame-table', 'style_cell'),
               Output('general-table', 'style_cell')],
              [Input('dark-toggle', 'n_clicks')])
def cell_toggle(n_clicks):
    light = []
    dark = []
    for _ in range(3):
        light.append({'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                      'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'})
        dark.append({'minWidth': '50px', 'width': '50px', 'maxWidth': '50px',
                     'backgroundColor': 'rgb(255, 255, 255)', 'color': 'black'})
    if n_clicks % 2 == 0:
        return light
    else:
        return dark

    # toggles dark mode for header of DataTable


@app.callback([Output('wheel-table', 'style_header'),
               Output('frame-table', 'style_header'),
               Output('general-table', 'style_header')],
              [Input('dark-toggle', 'n_clicks')])
def header_toggle(n_clicks):
    light = []
    dark = []
    for _ in range(3):
        light.append(
            {'textAlign': 'center', 'backgroundColor': 'rgb(30, 30, 30)'})
        dark.append(
            {'textAlign': 'center', 'backgroundColor': 'rgb(235, 235, 235)'})
    if n_clicks % 2 == 0:
        return light
    else:
        return dark


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)
