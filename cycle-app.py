import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as tbl
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

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
    par = bike.parameters[bike_name]
    parPure = bp.io.remove_uncertainties(par)
    return parPure

OPTIONS=['Benchmark',
         'Browser',
         'Browserins',
         'Crescendo',
         'Fisher',
         'Pista',
         'Rigid',
         'Silver',
         'Tms',
         'Yellow',
         'Yellowrev']    

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Bicycle Geometry Plot',
            id='main-header'),
    dcc.Dropdown(id='bike-dropdown',
                 value='Benchmark',
                 options=[{'label': i, 'value': i} for i in OPTIONS]),
    tbl.DataTable(id='par-table',
                  columns=[],
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
    html.Div(id='image-bin',
             children=[html.Img(src='',  
                                alt='A plot revealing the general geometry, centers of mass and moments of inertia of the given bicycle system.',
                                id='geometry-plot',),
                       html.Img(src='',
                                alt='A plot revealing the eigenvalues of the bicycle system as a function of speed.',
                                id='eigen-plot')])
])

    # Populates par-table data with parameters on new dropdown value/button press
@app.callback(Output('par-table', 'data'), [Input('bike-dropdown', 'value'), Input('reset-button', 'n_clicks')])
def populate_data(value, n_clicks):
    bike = bp.Bicycle(value, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    parPure = bp.io.remove_uncertainties(par)
    return [parPure]

    # Populates par-table columns with proper values
@app.callback(Output('par-table', 'columns'), [Input('bike-dropdown', 'value')])
def populate_columns(value):
    bike = bp.Bicycle(value, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    columns = [{'name': i, 'id': i} for i in par]
    return columns
    
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
@app.callback(Output('par-table', 'style_cell'), [Input('dark-toggle', 'n_clicks')])
def cell_toggle(n_clicks):
    if n_clicks%2 == 1:
        return {'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
    else:
        return {'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'backgroundColor': 'rgb(255, 255, 255)', 'color': 'black'}

    # Toggles dark mode for header of DataTable
@app.callback(Output('par-table', 'style_header'), [Input('dark-toggle', 'n_clicks')])
def cell_toggle(n_clicks):
    if n_clicks%2 == 1:      
        return {'textAlign': 'center', 'backgroundColor': 'rgb(30, 30, 30)'}
    else:
        return {'textAlign': 'center', 'backgroundColor': 'rgb(235, 235, 235)'}

if __name__ == '__main__':
    app.run_server(debug=True)