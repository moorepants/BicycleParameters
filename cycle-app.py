import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as tbl
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
    tbl.DataTable(id='table',
                  columns=[{'name': 'test', 'id': 'testtable'}]),                 
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