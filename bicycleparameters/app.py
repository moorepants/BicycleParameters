import os
import time

try:
    import dash
except ImportError as e:
    msg = ('Dash not installed, make sure to install the optional '
           'dependencies for the web app.')
    raise ImportError(msg) from e

try:
    import dash_bootstrap_components as dbc
except ImportError as e:
    msg = ('dash_bootstrap_components not installed, make sure to install the '
           'optional dependencies for the web app.')
    raise ImportError(msg) from e

try:
    import pandas
except ImportError as e:
    msg = ('pandas not installed, make sure to install the optional '
           'dependencies for the web app.')
    raise ImportError(msg) from e
else:
    del pandas

from dash import dash_table as tbl
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import numpy as np

# TODO : should this be a relative import?
import bicycleparameters as bp

PATH_TO_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
PATH_TO_APP_DATA = os.path.join(PATH_TO_THIS_FILE, 'app-data')
PATH_TO_ASSETS = os.path.join(PATH_TO_THIS_FILE, 'assets')

# list of all the whipple-carvallo parameters

par_list = ['rF', 'mF', 'IFxx', 'IFyy', 'rR', 'mR', 'IRxx', 'IRyy',
            'w', 'c', 'lam', 'g',
            'xB', 'zB', 'mB',
            'IBxx', 'IByy', 'IBzz', 'IBxz',
            'xH', 'zH', 'mH',
            'IHxx', 'IHyy', 'IHzz', 'IHxz']

# list of bicycle models in the dropdown menu

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

OPT_LABELS = [
    'Benchmark',
    'Browser (no rider)',
    'Browserins (no rider)',
    'Crescendo (no rider)',
    'Fisher (no rider)',
    'Pista (no rider)',
    'Rigid (no rider)',
    'Silver (no rider)',
    'Yellow (no rider)',
    'Yellowrev (no rider)',
]

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
                'Center of Mass Z [m]:',
                'Total Mass [kg]:',
                'Moment Ixx [kg*m²]:',
                'Moment Iyy [kg*m²]:',
                'Moment Izz [kg*m²]:',
                'Moment Ixz [kg*m²]:']

GENERAL_COLUMNS = [{'name': '', 'id': 'label', 'type': 'text', 'editable': False},
                   {'name': 'Parameters', 'id': 'data', 'type': 'numeric'}]

GENERAL_LABELS = ['Wheel Base [m]:',
                  'Trail [m]:',
                  'Steer Axis Tilt [degrees]:',
                  'Gravity [N/kg]:']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
app.title = 'Bicycle Dynamics Analysis App'
server = app.server  # needed for heroku or render


# app.layout defines the visual GUI elements on the website

app.layout = html.Div([
    dbc.Container(fluid=False,
                  children=[
                      dbc.Row([dbc.Col(dbc.NavbarSimple(brand='Bicycle Dynamics Analysis App',
                                                        dark=True,
                                                        color='primary'),
                                       width='auto',
                                       className='my-2')],
                                       justify='center'),
                      dbc.Row([dbc.Col(dcc.Loading(id='geometry-load',
                                                   type='dot',
                                                   children=[html.Div(id='geometry-bin',
                                                                      children=[html.Div([dcc.Graph(
                                                                                         id='geometry-plot',
                                                                                         className='img-fluid')])])]),
                                       lg=5,
                                       width=12),
                               dbc.Col(dcc.Loading(id='eigen-load',
                                                   type='dot',
                                                   children=[html.Div(id='eigen-bin',
                                                                      children=[html.Div([dcc.Graph(
                                                                                         id='eigen-plot',
                                                                                         className='img-fluid')])])]),
                                       lg=5,
                                       width=12),
                               dbc.Col(children=[html.H5('Choose a Parameter Set:',
                                                         className='centered'),
                                                 dcc.Dropdown(id='bike-dropdown',
                                                              value='Benchmark',
                                                              options=[{'label': k, 'value': v} for k, v in zip(OPT_LABELS, OPTIONS)],
                                                              style={'color': 'black'}),
                                                 dcc.Checklist(id='geometry-checklist',
                                                               options=[{'label': 'Show Centers of Mass ', 'value': 'centers'},
                                                                        {'label': 'Show Inertia Ellipsoids ', 'value': 'ellipse'},
                                                                        {'label': 'Show Self-stability region', 'value': 'stability'}],
                                                               value=['centers']),
                                                 dbc.Button('Reset Table',
                                                            id='reset-button',
                                                            color='primary',
                                                            size='lg',
                                                            n_clicks=0)],
                                       lg=2)],
                              className="my-2"),
                      dbc.Row([dbc.Col(tbl.DataTable(id='frame-table',
                                                     columns=FRAME_COLUMNS,
                                                     data=[],
                                                     style_cell={
                                                          'backgroundColor': 'white',
                                                          'color': 'black',
                                                          # 'border': '1px solid white',
                                                          'whiteSpace': 'normal',
                                                          'height': 'auto'
                                                      },
                                                      style_header={
                                                           'textAlign': 'center',
                                                           'backgroundColor': 'rgba(39,128,227,0.15)'
                                                      },
                                                      style_data_conditional=[{
                                                             "if": {"state": "selected"},
                                                              'backgroundColor': 'rgba(39,128,227,0.15)',
                                                              'color': 'black',
                                                              "border": "1px solid black",
                                                      },],
                                                      editable=True),
                                       lg=4),
                               dbc.Col(children=[html.H5('Set the EigenValue Speed Range [m/s]:',
                                                         className='centered'),
                                                 dcc.RangeSlider(id='range-slider',
                                                                 min=-45,
                                                                 max=45,
                                                                 step=5,
                                                                 value=[0, 10],
                                                                 marks={-40: {'label': '-40'},
                                                                        -30: {'label': '-30'},
                                                                        -20: {'label': '-20'},
                                                                        -10: {'label': '-10'},
                                                                        0: {'label': '0', },
                                                                        10: {'label': '10'},
                                                                        20: {'label': '20'},
                                                                        30: {'label': '30'},
                                                                        40: {'label': '40'}},
                                                                 allowCross=False),
                                                 dbc.Row([dbc.Col(tbl.DataTable(id='wheel-table',
                                                                                columns=WHEEL_COLUMNS,
                                                                                data=[],
                                                                                style_cell={
                                                                                    'whiteSpace': 'normal',
                                                                                    'height': 'auto'
                                                                                },
                                                                                style_header={
                                                                                    'textAlign': 'center',
                                                                                    'backgroundColor': 'rgba(39,128,227,0.15)'
                                                                                },
                                                                                style_data_conditional=[
                                                                                    {
                                                                                        "if": {"state": "selected"},
                                                                                        "backgroundColor": 'rgba(39,128,227,0.15)',
                                                                                        'color': 'black',
                                                                                        "border": "1px solid black",
                                                                                    },
                                                                                ],
                                                                                editable=True)),
                                                          dbc.Col(tbl.DataTable(id='general-table',
                                                                                columns=GENERAL_COLUMNS,
                                                                                data=[],
                                                                                style_cell={
                                                                                    'whiteSpace': 'normal',
                                                                                    'height': 'auto'
                                                                                },
                                                                                style_header={
                                                                                    'textAlign': 'center',
                                                                                    'backgroundColor': 'rgba(39,128,227,0.15)'
                                                                                },
                                                                                style_data_conditional=[
                                                                                    {
                                                                                        "if": {"state": "selected"},
                                                                                        "backgroundColor": 'rgba(39,128,227,0.15)',
                                                                                        'color': 'black',
                                                                                        "border": "1px solid black",
                                                                                    },
                                                                                ],
                                                                                editable=True))
                                                          ])
                                                 ],
                                       lg=8,
                                       align='end')],
                              className="my-2"),
                      dbc.Row([dbc.Col(dcc.Markdown(open(os.path.join(PATH_TO_ASSETS, 'app-explanation.md')).read()),
                                       width='auto',
                                       className='my-2')
                               ])
                  ])])


# create a set of Benchmark parameters with uncertainties removed

def new_par(bike_name):
    bike = bp.Bicycle(bike_name, pathToData=PATH_TO_APP_DATA)
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
            fW.append({'fW': wheelPure.get(par_list[i])})
        else:
            rW.append({'rW': wheelPure.get(par_list[i])})
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
    for i in range(12, len(par_list)):
        if i < 19:
            rB.append({'rB': framePure.get(par_list[i])})
        else:
            fA.append({'fA': framePure.get(par_list[i])})
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
    data = []
    for i in GENERAL_LABELS:
        genLabels.append({'label': i})
    for i in range(8, 12):
        data.append({'data': genPure.get(par_list[i])})

    # converts radians to degrees for display in the table
    lamD = data[2]
    radians = lamD.get('data')
    degrees = np.rad2deg(radians)
    data[2] = {'data': int(degrees)}

    for c, d in zip(genLabels, data):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        genData.append(zipped)

    return wheelData, frameData, genData


# updates geometry-plot & eigen-plot path with Dropdown value or edited
# DataTable values

@app.callback([Output('geometry-plot', 'figure'),
               Output('eigen-plot', 'figure')],
              [Input('bike-dropdown', 'value'),
               Input('wheel-table', 'data'),
               Input('frame-table', 'data'),
               Input('general-table', 'data'),
               Input('geometry-checklist', 'value'),
               Input('range-slider', 'value')])
def plot_update(value, wheel, frame, general, options, slider):

    # accesses Input properties to avoid redundancies
    ctx = dash.callback_context
    wheelData = ctx.inputs.get('wheel-table.data')
    frameData = ctx.inputs.get('frame-table.data')
    genData = ctx.inputs.get('general-table.data')
    checklistData = ctx.inputs.get('geometry-checklist.value')
    rangeSliderData = ctx.inputs.get('range-slider.value')

    # construct flags for selected values of the geometry plot display options
    mass_boolean = 'centers' in checklistData
    ellipse_boolean = 'ellipse' in checklistData
    stability_option = 'stability' in checklistData

    # sets the speed range for eigen-plot based on range-slider
    minBound = rangeSliderData[0]
    maxBound = rangeSliderData[1]
    steps = (maxBound-minBound)/0.1
    speeds = np.linspace(minBound, maxBound, num=int(steps))

    # create Bike using default data based on dropdown menu value
    Bike = bp.Bicycle(value, pathToData=PATH_TO_APP_DATA)

    # convert steer axis tilt into radians if recieving values from datatable
    # edits
    if ctx.triggered[0].get('prop_id') != 'bike-dropdown.value':

        # convert to radians
        degrees = float(genData[2].get('data'))
        radians = np.deg2rad(degrees)
        genData[2]['data'] = radians

        # creates an alternating list of [parameter,value] from table data
        newP = []
        for p in range(8):
            if p < 4:
                newP.extend([par_list[p], wheelData[p].get('fW')])
            else:
                newP.extend([par_list[p], wheelData[p-4].get('rW')])
        for p in range(12, len(par_list)):
            if p < 19:
                newP.extend([par_list[p], frameData[p-12].get('rB')])
            else:
                newP.extend([par_list[p], frameData[p-19].get('fA')])
        for p in range(8, 12):
            newP.extend([par_list[p], genData[p-8].get('data')])

        # inserts user edited data into the default Bike created earlier
        for i in range(0, len(newP), 2):
            Bike.parameters['Benchmark'][newP[i]] = newP[i+1]

    # create geometry-plot with plotly

    geo_plot = Bike._plot_bicycle_geometry_plotly(
        show=False, centerOfMass=mass_boolean, inertiaEllipse=ellipse_boolean)
    # Create eigenvalues-plot with plotly
    eigen_plot = Bike._plot_eigenvalues_vs_speed_plotly(
        speeds, show=False, stability_region=stability_option)

    return eigen_plot, geo_plot


# sets loading notification for the geometry plot
@app.callback(Output("geometry-load", "children"),
              [Input("geometry-bin", "children")])
def input_triggers_spinner1(value):
    time.sleep(1)
    return value


# sets loading notification for the eigenvalue plot
@app.callback(Output("eigen-load", "children"),
              [Input("eigen-bin", "children")])
def input_triggers_spinner2(value):
    time.sleep(1)
    return value


# omit the `dev_tools_ui` parameter to display debug info in the browser rather
# than in the terminal
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)
