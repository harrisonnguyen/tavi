import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
import os


LOGO = "tavi_logo.jpeg"
app = Dash(use_pages=True,external_stylesheets=[dbc.themes.FLATLY,dbc.icons.FONT_AWESOME])

server = app.server

description = """
PREDICT-TAVI is a machine learning-based survival analysis model for in-hospital, 30-day and 1-year mortality in patients undergoing transcatheter aortic valve implantation, generating personalized survival curves for any timepoint within a 2-year period. This model has been derived from 16,209 patients in the Australasian Cardiac Outcomes Registry (ACOR) TAVI Registry from 2018 to 2023, achieving a cumulative dynamic AUC of 0.704 using 12 key clinical variables.
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
                        html.A("github", href="https://github.com/harrisonnguyen/tavi",target="_blank")
                    ]
                )
                ],
            id="offcanvas",
            title="Information",
            is_open=False,
        ),
    ]
)


navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand("PREDICT-TAVI Score", className="ms-2",style={"font-size": "24px"})),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            offcanvas   
        ]
    ),
    color="primary",
    dark=True,
    style={'border-radius': '10px'},
)

navbar_old = dbc.NavbarSimple(
    children=[
        
        offcanvas
    ],
    brand="PREDICT-TAVI Score",
    brand_href="/",
    color="primary",
    dark=True,
    class_name='mb-3',
    style={'border-radius': '10px','font-size': '200px !important'},
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
