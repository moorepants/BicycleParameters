import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import os
import bicycleparameters as bp

###!!!Must ensure that current working directory is set to directory immediatly containing '\data\'!!!###
###---------------------------------------------------------------------------------------------------###

    # Initializes Benchmark and draws the geometry plot 
def new_geo_plot():
    bike = bp.Bicycle('Benchmark', pathToData=os.getcwd()+'\\data')
    plot = bike.plot_bicycle_geometry()
    plot.savefig(os.getcwd()+'\\assets\\geo-plots\\user-bikes\\dummy.png') # show=False

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
    html.Div(id='parameter-input-bin',
             children=[html.Div(id='geo-bin', children=[
                                html.H1('Overall Geometry Parameters!'),
                                dcc.Input(id='wheel-base',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='head-angle',
                                          value='', 
                                          type='text'),
                                 dcc.Input(id='trail',
                                          value='', 
                                          type='text')]),
                       html.Div(id='wheel-bin', children=[
                                html.H1('Wheel Parameters!'),
                                dcc.Input(id='front-diameter',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rear-diameter',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='front-mass',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rear-mass',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='front-Ixy',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rear-Ixy',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='front-Iz',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rear-Iz',
                                          value='', 
                                          type='text')]),
                       html.Div(id='frame-bin', children=[
                                html.H1('Frame Parameters!'),
                                dcc.Input(id='frame-CoM-x',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='frame-CoM-y',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='frame-mass',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='frame-I1',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='frame-I2',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='frame-Iz',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='frame-principle-axis',
                                          value='', 
                                          type='text')]),
                       html.Div(id='rider-bin', children=[
                                html.H1('Rider Parameters!'),
                                dcc.Input(id='rider-CoM-x',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rider-CoM-y',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rider-mass',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rider-I1',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rider-I2',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rider-Iz',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rider-principle-axis',
                                          value='', 
                                          type='text')]),
                       html.Div(id='rack-bin', children=[
                                html.H1('Rack Parameters!'),
                                dcc.Input(id='rack-CoM-x',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rack-CoM-y',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rack-mass',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rack-I1',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rack-I2',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rack-Iz',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='rack-principle-axis',
                                          value='', 
                                          type='text')]),
                       html.Div(id='fork-bin', children=[
                                html.H1('Fork Parameters!'),
                                dcc.Input(id='fork-x-axis',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='fork-y-axis',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='fork-mass',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='fork-I1',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='fork-I2',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='fork-Iz',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='fork-principle-axis',
                                          value='', 
                                          type='text')]),
                       html.Div(id='basket-bin', children=[
                                html.H1('Basket Parameters!'),
                                dcc.Input(id='basket-x-axis',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='basket-y-axis',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='basket-mass',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='basket-I1',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='basket-I2',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='basket-Iz',
                                          value='', 
                                          type='text'),
                                dcc.Input(id='basket-principle-axis',
                                          value='', 
                                          type='text')])
                       ]),
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