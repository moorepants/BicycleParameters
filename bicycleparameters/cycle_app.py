import io
import base64
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as tbl
import time
from dash.dependencies import Input, Output, State
import urllib
import urllib.parse
import cmath
import bicycleparameters as bp
import matplotlib
matplotlib.use('Agg')

path_to_this_file = os.path.dirname(os.path.abspath(__file__))
path_to_app_data = os.path.join(path_to_this_file, 'app-data')
path_to_assets = os.path.join(path_to_this_file, 'assets')

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
                'Center of Mass Z [m]:',
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = 'Bicycle Dynamics Analysis App'
server = app.server  # needed for heroku

app.layout = html.Div([
    dbc.Container(fluid=False,
                  children=[
                      dbc.Row([dbc.Col(dbc.NavbarSimple(brand='Bicycle Dynamics Analysis App',
                                                        dark=True,
                                                        color='info'),
                                       width='auto',
                                       className='my-2')],
                              justify='center'),
                      dbc.Row([dbc.Col(dcc.Loading(id='geometry-load',
                                                   type='dot',
                                                   children=[html.Div(id='geometry-bin',
                                                                      children=[html.Img(src='',
                                                                                         id='geometry-plot',
                                                                                         className='img-fluid')])]),
                                       lg=5,
                                       width=12),
                               dbc.Col(dcc.Loading(id='eigen-load',
                                                   type='dot',
                                                   children=[html.Div(id='eigen-bin',
                                                                      children=[html.Img(src='',
                                                                                         id='eigen-plot',
                                                                                         className='img-fluid')])]),
                                       lg=5,
                                       width=12),
                               dbc.Col(children=[html.H5('Choose a Parameter Set:',
                                                         className='centered'),
                                                 dcc.Dropdown(id='bike-dropdown',
                                                              value='Benchmark',
                                                              options=[
                                                                  {'label': i, 'value': i} for i in OPTIONS],
                                                              style={'color': 'black'}),
                                                 dcc.Checklist(id='geometry-checklist',
                                                               options=[{'label': 'Show Centers of Mass ', 'value': 'centers'},
                                                                        {'label': 'Show Inertia Ellipsoids ', 'value': 'ellipse'}],
                                                               value=['centers']),
                                                 dbc.Button('Reset Table',
                                                            id='reset-button',
                                                            color='primary',
                                                            size='lg',
                                                            n_clicks=0),
                                                 html.A('Download Profile',
                                                        id='download-button',
                                                        href='',
                                                        className='btn btn-primary',
                                                        target='_blank',
                                                        download='eigenvalues and eigenvectors')],
                                       lg=2)],
                              no_gutters=True,
                              className="my-2"),
                      dbc.Row([dbc.Col(tbl.DataTable(id='frame-table',
                                                     columns=FRAME_COLUMNS,
                                                     data=[],
                                                     style_cell={
                                                         'backgroundColor': 'rgb(50, 50, 50)',
                                                         'color': 'white',
                                                         'border': '1px solid white',
                                                         'whiteSpace': 'normal',
                                                         'height': 'auto'
                                                     },
                                                     style_header={
                                                         'textAlign': 'center',
                                                         'backgroundColor': 'rgb(30, 30, 30)'
                                                     },
                                                     style_data_conditional=[
                                                         {
                                                            "if": {"state": "selected"},             
                                                            "backgroundColor": "rgb(255,255,255)",
                                                            'color': 'black',
                                                            "border": "1px solid black",
                                                         },
                                                     ],
                                                     editable=True),
                                       lg=4),
                               dbc.Col(children=[html.H5('Set the EigenValue Speed Range [m/s]:',
                                                         className='centered'),
                                                 dcc.RangeSlider(id='range-slider',
                                                                 min=-45,
                                                                 max=45,
                                                                 step=5,
                                                                 value=[
                                                                     0, 10],
                                                                 marks={-40: {'label': '-40', 'style': {'color': '#ffffff'}},
                                                                        -30: {'label': '-30', 'style': {'color': '#ffffff'}},
                                                                        -20: {'label': '-20', 'style': {'color': '#ffffff'}},
                                                                        -10: {'label': '-10', 'style': {'color': '#ffffff'}},
                                                                        0: {'label': '0', 'style': {'color': '#ffffff'}},
                                                                        10: {'label': '10', 'style': {'color': '#ffffff'}},
                                                                        20: {'label': '20', 'style': {'color': '#ffffff'}},
                                                                        30: {'label': '30', 'style': {'color': '#ffffff'}},
                                                                        40: {'label': '40', 'style': {'color': '#ffffff'}}},
                                                                 allowCross=False),
                                                 dbc.Row([dbc.Col(tbl.DataTable(id='wheel-table',
                                                                                columns=WHEEL_COLUMNS,
                                                                                data=[],
                                                                                style_cell={
                                                                                    'backgroundColor': 'rgb(50, 50, 50)',
                                                                                    'color': 'white',
                                                                                    'border': '1px solid white',
                                                                                    'whiteSpace': 'normal',
                                                                                    'height': 'auto'
                                                                                },
                                                                                style_header={
                                                                                    'textAlign': 'center',
                                                                                    'backgroundColor': 'rgb(30, 30, 30)'
                                                                                },
                                                                                style_data_conditional=[
                                                                                    {
                                                                                        "if": {"state": "selected"},              
                                                                                        "backgroundColor": "rgb(255,255,255)",
                                                                                        'color': 'black',
                                                                                        "border": "1px solid black",
                                                                                    },
                                                                                ],
                                                                                editable=True)),
                                                          dbc.Col(tbl.DataTable(id='general-table',
                                                                                columns=GENERAL_COLUMNS,
                                                                                data=[],
                                                                                style_cell={
                                                                                    'backgroundColor': 'rgb(50, 50, 50)',
                                                                                    'color': 'white',
                                                                                    'border': '1px solid white',
                                                                                    'whiteSpace': 'normal',
                                                                                    'height': 'auto'
                                                                                },
                                                                                style_header={
                                                                                    'textAlign': 'center',
                                                                                    'backgroundColor': 'rgb(30, 30, 30)'
                                                                                },
                                                                                style_data_conditional=[
                                                                                    {
                                                                                        "if": {"state": "selected"},              
                                                                                        "backgroundColor": "rgb(255,255,255)",
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
                      dbc.Row([dbc.Col(dcc.Markdown(open(os.path.join(path_to_assets, 'app-explanation.md')).read()),
                                       width='auto',
                                       className='my-2')
                               ])
                  ])])

# creates generic set of Benchmark parameters

@app.callback(Output('download-button', 'href'),
              [Input('bike-dropdown', 'value'),
               Input('range-slider', 'value')])

def update_download_link(value,slider):
    bicycle = bp.Bicycle(value, pathToData=path_to_app_data)
    ctx = dash.callback_context
    rangeSliderData = ctx.inputs.get('range-slider.value')
    minBound = rangeSliderData[0]
    maxBound = rangeSliderData[1]
    steps = maxBound-minBound
    speeds = np.linspace(minBound, maxBound, num=int(steps))
    w,v = bicycle.eig(speeds)
    result = np.dstack(
        np.apply_along_axis(
            lambda x: [x.real, x.imag], 0, w)
        ).flatten().reshape(len(w),8)
    header = np.array(['speed','eval1_real','eval2_real','eval3_real','eval4_real','eval1_imag','eval2_imag','eval3_imag','eval4_imag'])
    header.shape = (1,len(header))
    speeds.shape = (len(speeds),1)
    result = np.hstack((speeds,result))
    result = np.vstack((header,result))
    result = pd.DataFrame(result)
    csv_string = result.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," +  urllib.parse.quote(csv_string)
    return csv_string

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

    # sets the speed range for eigen-plot based on range-slider
    minBound = rangeSliderData[0]
    maxBound = rangeSliderData[1]
    steps = (maxBound-minBound)/0.1
    speeds = np.linspace(minBound, maxBound, num=int(steps))

    Bike = bp.Bicycle(value, pathToData=path_to_app_data)

    # must convert steer axis tilt into radians when recieving values from the datatables
    if ctx.triggered[0].get('prop_id') != 'bike-dropdown.value':

        # convert to radians
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
        for i in range(0, len(newP), 2):
            Bike.parameters['Benchmark'][newP[i]] = newP[i+1]

    # create geometry-plot image
    geo_fake = io.BytesIO()
    geo_plot = Bike.plot_bicycle_geometry(
        show=False, centerOfMass=mass_boolean, inertiaEllipse=ellipse_boolean)
    geo_plot.savefig(geo_fake)
    geo_image = base64.b64encode(geo_fake.getvalue())
    plt.close(geo_plot)

    # create eigen-plot image
    eigen_fake = io.BytesIO()
    eigen_plot = Bike.plot_eigenvalues_vs_speed(
        speeds, show=False, grid=True, show_legend=False)
    eigen_plot.savefig(eigen_fake)
    eigen_image = base64.b64encode(eigen_fake.getvalue())
    plt.close(eigen_plot)

    return 'data:image/png;base64,{}'.format(geo_image.decode()), 'data:image/png;base64,{}'.format(eigen_image.decode())

    # sets loading notification for the geometry plot


@app.callback(Output("geometry-load", "children"), [Input("geometry-bin", "children")])
def input_triggers_spinner1(value):
    time.sleep(1)
    return value


@app.callback(Output("eigen-load", "children"), [Input("eigen-bin", "children")])
def input_triggers_spinner2(value):
    time.sleep(1)
    return value


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)
