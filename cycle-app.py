import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as tbl
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from collections import OrderedDict
import os
import base64
import io

import bicycleparameters as bp
import pandas as pd
import numpy as np 

###!!!Must ensure that current working directory is set to directory immediatly containing '\data\'!!!###
###---------------------------------------------------------------------------------------------------###

    # Initializes bike data as Bicycle and retrieves its parameters
def new_par(bike_name):
    bike = bp.Bicycle(bike_name, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    parPure = bp.io.remove_uncertainties(par)
    return parPure

pList=['rF', 'mF', 'IFxx', 'IFyy', 'rR', 'mR', 'IRxx', 'IRyy',
       'w', 'c', 'lam', 'g',
       'xB', 'zB', 'mB', 'IBxx', 'IByy', 'IBzz', 'IBxz', 'xH', 'zH', 'mH', 'IHxx', 'IHyy', 'IHzz', 'IHxz',]

OPTIONS=['Benchmark',
         'Browser',
         'Browserins',
         'Crescendo',
         'Fisher',
         'Pista',
         'Rigid',
         'Silver',
         'Yellow',
         'Yellowrev']

WHEEL_COLUMNS=[{'name': '', 'id': 'label', 'type': 'text'}, 
               {'name': 'Front Wheel', 'id': 'fW', 'type': 'numeric'},
               {'name': 'Rear Wheel', 'id': 'rW', 'type': 'numeric'}]

WHEEL_LABELS=['Radius',
              'Mass',
              'Moment Ixx',
              'Moment Iyy']

FRAME_COLUMNS=[{'name': '', 'id': 'label', 'type': 'text'},
               {'name': 'Rear Body', 'id': 'rB', 'type': 'numeric'},
               {'name': 'Front Assembly', 'id': 'fA', 'type': 'numeric'}]    

FRAME_LABELS=['Center-of-Mass X',
              'Center-of-Mass Y',
              'Total Mass',
              'Moment Ixx',
              'Moment Iyy',
              'Moment Izz',
              'Moment Ixz']

GENERAL_COLUMNS=[{'name': '', 'id': 'label', 'type': 'text'},
                 {'name': 'Contextual Parameters', 'id': 'con', 'type': 'numeric'}]

GENERAL_LABELS=['Wheel Base', 
                'Trail',
                'Steer Axis Tilt',
                'Gravity']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('BicycleParameters Web Application',
            id='main-header'),
    dcc.Dropdown(id='bike-dropdown',
                 value='Benchmark',
                 options=[]),
    tbl.DataTable(id='wheel-table',
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
                  editable=True),                               
    html.Button('Reset Table',
                id='reset-button',
                type='button',
                n_clicks=0),
    html.Button('Calculate & Draw Plot',
                id='calc-button',
                type='button',
                n_clicks=0),
    html.Button('Toggle Dark Mode',
                id='dark-toggle',
                type='button',
                n_clicks=0),
    html.Button('Update Dropdown Menu',
                id='drop-button',
                type='button',
                n_clicks=0),
    dcc.Input(placeholder='Save File As...',
              value='',
              id='save-input',
              type='text'),
    html.Div(id='image-bin',
             children=[html.Img(src='',  
                                alt='A plot revealing the general geometry, centers of mass and moments of inertia of the given bicycle system.',
                                id='geometry-plot',),
                       html.Img(src='',
                                alt='A plot revealing the eigenvalues of the bicycle system as a function of speed.',
                                id='eigen-plot')]),
])  
    
    # Populates wheel-table parameter with data
@app.callback(Output('wheel-table', 'data'), [Input('bike-dropdown', 'value'), Input('reset-button', 'n_clicks')])
def populate_wheel_data(value, n_clicks):
    parPure = new_par(value)
    data = []
    labels = []
    fW = []
    rW = []
    for i in WHEEL_LABELS:
        labels.append({'label': i})
    for i in range(8):
        if i < 4:
            fW.append({'fW':parPure.get(pList[i])}) 
        else:
            rW.append({'rW':parPure.get(pList[i])})
    for c, d, e in zip(labels, fW, rW):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        zipped.update(e)
        data.append(zipped)
    return data

    # Populates frame-table parameter with data
@app.callback(Output('frame-table', 'data'), [Input('bike-dropdown', 'value'), Input('reset-button', 'n_clicks')])
def populate_frame_data(value, n_clicks):
    parPure = new_par(value)
    data = []
    labels = []
    rB = []
    fA = []
    for i in FRAME_LABELS:
        labels.append({'label': i})
    for i in range(12, len(pList)):
        if i < 19:
            rB.append({'rB':parPure.get(pList[i])}) 
        else:
            fA.append({'fA':parPure.get(pList[i])})
    for c, d, e in zip(labels, rB, fA):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        zipped.update(e)
        data.append(zipped)
    return data

    # Populates general-table parameter with data
@app.callback(Output('general-table', 'data'), [Input('bike-dropdown', 'value'), Input('reset-button', 'n_clicks')])
def populate_general_data(value, n_clicks):
    parPure = new_par(value)
    data = []
    labels = []
    con = []
    for i in GENERAL_LABELS:
        labels.append({'label': i})
    for i in range(8, 12):
        con.append({'con':parPure.get(pList[i])}) 
    for c, d in zip(labels, con):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        data.append(zipped)
    return data

    # Updates geo-plot & eigen-plot path with Dropdown value or edited DataTable values
@app.callback([Output('geometry-plot', 'src'),
               Output('eigen-plot', 'src')],
              [Input('bike-dropdown', 'value'),                    
               Input('wheel-table', 'data'),
               Input('frame-table', 'data'),
               Input('general-table', 'data')]) 
def plot_update(value, x, y, z):
    ctx = dash.callback_context
    speeds = np.linspace(0, 10, num=100)
    wheelData = ctx.inputs.get('wheel-table.data')
    frameData = ctx.inputs.get('frame-table.data')
    genData = ctx.inputs.get('general-table.data')
    image = value+'.png' 
    newP = []
    currentBike = bp.Bicycle(value, pathToData=os.getcwd()+'\\data')
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
    for p in range(8,12):
        newP.extend([pList[p], genData[p-8].get('con')])
    for i in range(0,len(newP),2):
        currentBike.parameters['Benchmark'][newP[i]] = newP[i+1]
    print(currentBike.parameters['Benchmark'])
    geo_plot = currentBike.plot_bicycle_geometry() 
    geo_fake = io.BytesIO()
    geo_plot.savefig(geo_fake)
    geo_image = base64.b64encode(geo_fake.getvalue())
    eigen_plot = currentBike.plot_eigenvalues_vs_speed(speeds, show=False)
    eigen_fake = io.BytesIO()
    eigen_plot.savefig(eigen_fake)    
    eigen_image = base64.b64encode(eigen_fake.getvalue())     
    return 'data:image/png;base64,{}'.format(geo_image.decode()), 'data:image/png;base64,{}'.format(eigen_image.decode())

    # Updates dropdown options with save-input value
@app.callback(Output('bike-dropdown', 'options'), [Input('drop-button', 'n_clicks')], [State('save-input', 'value')])
def update_dropdown(n_clicks, value):
    if value == '':
        return [{'label': i, 'value': i} for i in OPTIONS]
    else:
        OPTIONS.append(value)
        return [{'label': i, 'value': i} for i in OPTIONS]

    # Toggles dark mode for cells of DataTable 
@app.callback([Output('wheel-table', 'style_cell'),
               Output('frame-table', 'style_cell'),
               Output('general-table', 'style_cell')],
              [Input('dark-toggle', 'n_clicks')])
def cell_toggle(n_clicks):
    light = []
    dark = []
    for i in range(3):
        light.append({'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'})
        dark.append({'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'backgroundColor': 'rgb(255, 255, 255)', 'color': 'black'})
    if n_clicks%2 == 0:
        return light
    else:
        return dark

    # Toggles dark mode for header of DataTable
@app.callback([Output('wheel-table', 'style_header'),
               Output('frame-table', 'style_header'),
               Output('general-table', 'style_header')],
              [Input('dark-toggle', 'n_clicks')])
def header_toggle(n_clicks):
    light = []
    dark = []
    for i in range(3):
        light.append({'textAlign': 'center', 'backgroundColor': 'rgb(30, 30, 30)'})
        dark.append({'textAlign': 'center', 'backgroundColor': 'rgb(235, 235, 235)'})
    if n_clicks%2 == 0:      
        return light
    else:
        return dark

if __name__ == '__main__':
    app.run_server(debug=True)