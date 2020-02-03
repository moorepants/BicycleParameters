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

    # Initializes Benchmark and retrieves its parameters
def new_par(bike_name):
    bike = bp.Bicycle(bike_name, pathToData=os.getcwd()+'\\data')
    par = bike.parameters[bike_name]
    parPure = bp.io.remove_uncertainties(par)
    return parPure

BENCHMARK = new_par('Benchmark')  

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
                  editable=True),                 
    html.Button('Create Dummy Bicycle Plot!',
                id='add-button',
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

    # Populates par-table data with parameters
@app.callback(Output('par-table', 'data'), [Input('bike-dropdown', 'value')])
def populate_data(value):
    bike = bp.Bicycle(value, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    parPure = bp.io.remove_uncertainties(par)
    return [parPure]

    # Populates par-table columns with proepr values
@app.callback(Output('par-table', 'columns'), [Input('bike-dropdown', 'value')])
def populate_columns(value):
    bike = bp.Bicycle(value, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    columns = [{'name': i, 'id': i} for i in BENCHMARK]
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

'''
    # Accesses /data/ and draws plot of user input, then adds user input to Dropdown list 
@app.callback(Output('bike-dropdown', 'options'), [Input('add-button', 'n_clicks')], [State('bike-input', 'value')])
def add_option(n_clicks, value):
    if value == '':
        raise PreventUpdate      
    else:
        #new_geo_plot()  # DEBUG # - refreshes page after this command is called, removing the added value from list
        OPTIONS.append(value)  
        OPTIONS.sort() # Find way to alphabetize this list
        return [{'label': i, 'value': i} for i in OPTIONS]
'''

if __name__ == '__main__':
    app.run_server(debug=True)