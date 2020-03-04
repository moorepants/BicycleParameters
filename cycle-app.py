import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as tbl
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from collections import OrderedDict
import os

import bicycleparameters as bp
import pandas as pd
import numpy as np 
from bicycleparameters import parameter_sets as ps

###!!!Must ensure that current working directory is set to directory immediatly containing '\data\'!!!###
###---------------------------------------------------------------------------------------------------###

    # Initializes bike data as Bicycle and retrieves its parameters
def new_par(bike_name):
    bike = bp.Bicycle(bike_name, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    parPure = bp.io.remove_uncertainties(par)
    return parPure
'''
    # Generates nested dict in {parameter: {column ID: data val}} format 
def genDataDic(columns, bike, index, end):
    data_dic = OrderedDict()
    parPure = new_par(bike)
    for c in columns:     
        for p in range(index,end):   
            data_dic[pList[p]] = {c['id']:parPure.get(pList[p])}
            index += 1
            if index == 4:
                break
    return data_dic
'''

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

WHEEL_COLUMNS=[{'name': 'Front Wheel', 'id': 'fW'},
               {'name': 'Rear Wheel', 'id': 'rW'}]

WHEEL_ROWS=['Radius',
            'Mass',
            'Moment Ixx',
            'Moment Iyy']

FRAME_COLUMNS=[{'name': 'Rear Body', 'id': 'rB'},
               {'name': 'Front Assembly', 'id': 'fA'}]    

FRAME_ROWS=['CoM X',
            'CoM Y',
            'Mass',
            'Moment I11',
            'Moment I22',
            'Moment Izz',
            'Principle Axis Angle']

GENERAL_COLUMNS=[{'name': '', 'id': 'label'},
                 {'name': 'Value', 'id': 'con'}]

GENERAL_LABELS=['Wheel Base', 
                'Trail',
                'Steer Axis Tilt',
                'Gravity']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Bicycle Geometry Plot',
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
def populate_data(value, n_clicks):
    parPure = new_par(value)
    data = []
    fW = []
    rW = []
    for i in range(8):
        if i < 4:
            fW.append({'fW':parPure.get(pList[i])}) 
        else:
            rW.append({'rW':parPure.get(pList[i])})
    for c, d in zip(fW, rW):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        data.append(zipped)
    return data

    # Populates frame-table parameter with data
@app.callback(Output('frame-table', 'data'), [Input('bike-dropdown', 'value'), Input('reset-button', 'n_clicks')])
def populate_data(value, n_clicks):
    parPure = new_par(value)
    data = []
    rB = []
    fA = []
    for i in range(12, len(pList)):
        if i < 19:
            rB.append({'rB':parPure.get(pList[i])}) 
        else:
            fA.append({'fA':parPure.get(pList[i])})
    for c, d in zip(rB, fA):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        data.append(zipped)
    return data

    # Populates general-table parameter with data
@app.callback(Output('general-table', 'data'), [Input('bike-dropdown', 'value'), Input('reset-button', 'n_clicks')])
def populate_data(value, n_clicks):
    dataDic = OrderedDict()
    parPure = new_par(value)
    data = []
    empty = []
    con = []
    for i in GENERAL_LABELS:
        empty.append({'label': i})
    for i in range(8, 12):
        con.append({'con':parPure.get(pList[i])}) 
    for c, d in zip(empty, con):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        data.append(zipped)
    return data
 
    # Updates geometry-plot path with Dropdown value ***(not currently functioning)***
@app.callback(Output('geometry-plot', 'src'),
              [Input('bike-dropdown', 'value'),                    
               Input('calc-button', 'n_clicks')],
              [State('wheel-table', 'data')])
def update_geo_plot(value, data, n_clicks):
    ctx = dash.callback_context
    pDic = {}
    currentBike = bp.Bicycle(value, pathToData=os.getcwd()+'\\data')
    image = value+'.png' 
    for p in range(8):
        if p < 4:
            pDic.update({pList[p]:data})
        else:
            pDic.update({pList[p]:data})
    for key in pDic:
        currentBike.parameters['Benchmark'][key] = pDic.get(key)
    if ctx.triggered[0] == 'n_clicks':
        plot = currentBike.plot_bicycle_geometry()
        plot.savefig(os.getcwd()+'\\assets\\geo-plots\\user-bikes\\'+image)
        return '/assets/geo-plots/user-bikes/'+image
    else: 
        if os.path.exists(os.getcwd()+'\\assets\\geo-plots\\defaults\\'+image):        
            return '/assets/geo-plots/defaults/'+image
    
    # Updates eigen-plot path with Dropdown value
@app.callback(Output('eigen-plot', 'src'), [Input('bike-dropdown', 'value')])
def reveal_eigen_plot(value):
    image = value+'.png'
    if os.path.exists(os.getcwd()+'\\assets\\eigen-plots\\defaults\\'+image):        
        return '/assets/eigen-plots/defaults/'+image
    else: 
        return '/assets/eigen-plots/user-bikes/'+image

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