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

    # Generates nested dict with keys = parameter, and values = {column ID: data val}
def genDataDic(constant, value, index, end):
    data_dic = OrderedDict()
    parPure = new_par(value)
    for i in constant:     
        for p in range(index,end):   
            data_dic[pList[p]] = {i['id']:parPure.get(pList[p])}
            index += 1
            if index == 4:
                break
    return data_dic

pList=['rF',
       'mF',
       'IFxx',
       'IFyy',
       'rR',
       'mR',
       'IRxx',
       'IRyy',
       'w',
       'c',
       'lam',
       'g',
       'xB',
       'zB',
       'mB',
       'IBxx',
       'IByy',
       'IBzz',
       'xH',
       'zH',
       'mH',
       'IHxx',
       'IHyy',
       'IHzz',
       'IHxz',]

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

FRAME_COLUMNS=[{'name': 'General', 'id': 'bg'},
               {'name': 'Frame', 'id': 'rF'},
               {'name': 'Rider', 'id': 'rider'},
               {'name': 'Rear Rack', 'id': 'rack'},
               {'name': 'Front End', 'id': 'fE'},
               {'name': 'Basket', 'id': 'basket'}]    

FRAME_ROWS=['CoM X',
            'CoM Y',
            'Mass',
            'Moment I11',
            'Moment I22',
            'Moment Izz',
            'Principle Axis Angle']

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
    html.Button('Reset Table',
                id='reset-button',
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
                                id='eigen-plot')])
])

    # Updates dropdown options with save-input value
@app.callback(Output('bike-dropdown', 'options'), [Input('drop-button', 'n_clicks')], [State('save-input', 'value')])
def update_dropdown(n_clicks, value):
    if value == '':
        return [{'label': i, 'value': i} for i in OPTIONS]
    else:
        OPTIONS.append(value)
        return [{'label': i, 'value': i} for i in OPTIONS]

    # Populates wheel-table parameter with data
@app.callback(Output('wheel-table', 'data'), [Input('bike-dropdown', 'value'), Input('reset-button', 'n_clicks')])
def populate_data(value, n_clicks):
    dataDic = genDataDic(WHEEL_COLUMNS, value, 0, 8)
    data = []
    fW = []
    rW = []
    index = 0
    for key in dataDic:
        if index < 4:
            fW.append(dataDic.get(key)) 
        if index >= 4:
            rW.append(dataDic.get(key))
        index += 1 
    for c, d in zip(fW, rW):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        data.append(zipped)
    return data
    

'''
    # Populates par-table columns with proper values
@app.callback(Output('par-table', 'columns'), [Input('bike-dropdown', 'value')])
def populate_columns(value):
    bike = bp.Bicycle(value, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    columns = [{'name': i, 'id': i} for i in par]
    return columns
'''    
    # Updates geometry-plot path with Dropdown value
@app.callback(Output('geometry-plot', 'src'), [Input('bike-dropdown', 'value')])
def reveal_geo_plot(value):
    image = value+'.png'
    if os.path.exists(os.getcwd()+'\\assets\\geo-plots\\defaults\\'+image):        
        return '/assets/geo-plots/defaults/'+image
    else: 
        return '/assets/geo-plots/user-bikes/'+image

    # Updates eigen-plot path with Dropdown value
@app.callback(Output('eigen-plot', 'src'), [Input('bike-dropdown', 'value')])
def reveal_geo_plot(value):
    image = value+'.png'
    if os.path.exists(os.getcwd()+'\\assets\\eigen-plots\\defaults\\'+image):        
        return '/assets/eigen-plots/defaults/'+image
    else: 
        return '/assets/eigen-plots/user-bikes/'+image

    # Toggles dark mode for cells of DataTable
@app.callback(Output('wheel-table', 'style_cell'), [Input('dark-toggle', 'n_clicks')])
def cell_toggle(n_clicks):
    if n_clicks%2 == 1:
        return {'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
    else:
        return {'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'backgroundColor': 'rgb(255, 255, 255)', 'color': 'black'}

    # Toggles dark mode for header of DataTable
@app.callback(Output('wheel-table', 'style_header'), [Input('dark-toggle', 'n_clicks')])
def header_toggle(n_clicks):
    if n_clicks%2 == 1:      
        return {'textAlign': 'center', 'backgroundColor': 'rgb(30, 30, 30)'}
    else:
        return {'textAlign': 'center', 'backgroundColor': 'rgb(235, 235, 235)'}

if __name__ == '__main__':
    app.run_server(debug=True)