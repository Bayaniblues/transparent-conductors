import json
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import dash_bio as dashbio
import dash_bio_utils.xyz_reader as xyz_reader


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])

## global state ##

def load_model():
    pkl_filename = "pickle_model.pkl"

    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model

bandgap_poly_svm = load_model()

target = ['formation_energy_ev_natom', 'bandgap_energy_ev']
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
### Scale it down
scal = StandardScaler()
train = pd.DataFrame(scal.fit_transform(train.values), index=train.index, columns=train.columns)
test = pd.DataFrame(scal.fit_transform(test.values), index=test.index, columns=test.columns)
### split data
train, val = train_test_split(train, test_size=0.40, random_state=42)
X_train = train.drop(columns=['formation_energy_ev_natom', 'bandgap_energy_ev', 'id', 'spacegroup'])
y_train = train[target]
X_val = val.drop(columns=['formation_energy_ev_natom', 'bandgap_energy_ev', 'id', 'spacegroup'])
y_val = val[target]
X_test = test.drop(columns=['id', 'spacegroup'])

## higher order logic


## Components



### generates component
def speck_component(index, molecule):
    return dashbio.Speck(
                    id=f'my-dashbio-speck{index}',
                    view={
                        'resolution': 500,
                        'ao': 1,
                        'atomScale': .25,
                        'relativeAtomScale': 0.10,
                        'bonds': True
                    },
                    data=molecule
                )


jumbotron = dbc.Jumbotron(
    [
        dbc.Container(
            [
                html.H4("Search crystals", className="display-3"),
                html.P(
                    "Search Crystals Based Off Of Support Vector Conductor Predictions",
                    className="lead",
                ),
            ],
            fluid=True,
        )
    ],
    fluid=True,
)

df = pd.DataFrame(load_model().support_vectors_)


fig = go.Figure()
fig.add_trace(go.Scatter(x=[-3,3],y=[3.5,3.5],
                    marker=dict(
                    size=3,

                    ),
                    fill='tozeroy',
                    mode='markers',
                    name='y: Better Insulators',
                    line_color='#2B3823'))


fig.add_trace(go.Scatter(x=[-3,3],y=[-6.5,-6.5],
                    marker=dict(
                    size=3,

                    ),
                    fill='tozeroy',
                    mode='markers',
                    name='y: Better Conductors',
                    line_color='#798274'))


fig.add_trace(go.Scatter(x=X_test['lattice_vector_2_ang'],y=df[5],
                    marker=dict(
                    size=3,

                    ),
                    #fill='tozeroy',
                    mode='markers',
                    name='y: support vectors',
                    line_color='#D43931'))

fig.add_trace(go.Scatter(x=X_test['lattice_vector_2_ang'],y=bandgap_poly_svm.predict(X_test),
                    marker=dict(
                    size=3,

                    ),
                    line_color = '#354F27',
                    #fill='tozeroy',
                    mode='markers',
                    name='y: prediction'))

fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll',
        'width': '700px'
    }
}

output_md = html.Div([
            dcc.Markdown("""
                Click on the green points to generate a molecule.
            """),
            html.Pre(id='click-data', style=styles['pre']),
            ], className='three columns')


fig.update_layout(title="X Feature: lattice_vector_2_ang",)

scatter_plot = dcc.Graph(id='basic-interactions',figure=fig)

main_card = dbc.Card(
    children=[
    scatter_plot,
    dbc.CardHeader(dcc.Loading(
        id="loading-2",
        children=[html.Div([html.Div(id="mol-component")])],
        type="circle",
    ), ),
    dbc.CardBody([
        output_md
    ]),

],style={'margin': '5rem',
         'max-width':"800px",
         "margin-top": '-3rem'})


## our layout
app.layout = html.Div(style={
    },children=[jumbotron,main_card])

## callback hell

@app.callback(
    Output('click-data', 'children'),
    [Input('basic-interactions', 'clickData')])


def display_click_data(clickData):
    input_in = json.dumps(clickData, indent=2)
    #print(elements)
    curveNumber = json.loads(input_in)['points'][0]['curveNumber']
    Index = json.loads(input_in)['points'][0]['pointIndex']+1
    print(curveNumber)
    if curveNumber == 3:
        data_in = f'./test_out/{Index}.xyz'
        molecule = xyz_reader.read_xyz(datapath_or_datastring=data_in, is_datafile=True)
        return speck_component(Index, molecule)
        #print(f'notebooks/test_out/{Index}.xyz')
    elif curveNumber == 2:
        pass
    else:
        pass

    return json.dumps(clickData, indent=2)

if __name__ == '__main__':
    app.run_server()