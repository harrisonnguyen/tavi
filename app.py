import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
import os



app = Dash(use_pages=True,external_stylesheets=[dbc.themes.FLATLY,dbc.icons.FONT_AWESOME])

server = app.server

description = """
The PREDICT-TAVI Score
"""



offcanvas = html.Div(
    [
        dbc.Button(id="open-offcanvas", n_clicks=0,children=html.I(className = "fa-solid fa-circle-info fa-xl"),
                style={'color':'white'}),
        dbc.Offcanvas(
            [
                html.P(
                   
                    description
                ),
                #html.P(
                #    [
                #        "This work is based on a paper found in this ", html.A("link", href="",)
                #    ]
                #),
                html.P(
                    [
                        "Code of the methodology and application can be found on ", 
                        html.A("github", href="https://github.com/harisritharan/tavi")
                    ]
                )
                ],
            id="offcanvas",
            title="Information",
            is_open=False,
        ),
    ]
)


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Mortality", href="/")),
        offcanvas
    ],
    brand="PREDICT-TAVI Score",
    brand_href="/",
    color="primary",
    dark=True,
    class_name='mb-3',
    style={'border-radius': '10px'},
    links_left=False,
    fluid=True
)

app.layout = html.Div(
    [

    dbc.Container(
    [
        navbar,
        dash.page_container
        #html.Hr(),
        

    ],
    fluid='sm',
    ),
    dbc.Container([
    dbc.Row(
            [
                dbc.Col(dbc.Table(id='table-content', striped=True, bordered=True, hover=True), lg=12),
            ],
            align="center",
        )
    ],
    fluid=True)
    ]

)

@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open



if __name__ == '__main__':
    app.run(debug=True)
