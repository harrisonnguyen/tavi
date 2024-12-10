import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
import os
import numpy as np
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import utils
dash.register_page(__name__, 
                   path='/',
                   title='TAVI Score',
                   name='TAVI Score')

df_limits = pd.read_csv('variable_ranges.csv',index_col="variable")
df_template = pd.DataFrame(columns=df_limits.index[1:])

default_fig = px.line(x=[],
   labels={'index': 'Probability of Survival', 'value':'Days since Discharge'}
)
default_fig.update_yaxes(range=[0,1])
default_fig.update_xaxes(range=[0,365*2])
#df_template = pd.read_csv('dataframe_template.csv',index_col=0)

continuous_features = [
        'kccq_summ_bl',
        'age',
        'haemoglobin_adj',
        'creatinine',
        'weight',
        'albumin_adj',
        'lvef_value',
        'av_area_tte'] 
categorical_features = ['prior_af_1.0','prior_pad_1.0','chronic_lung_4.0','chronic_lung_3.0','male']
    
    



mortality_year_pipe = joblib.load("model/gbm_final_1.pkl")

preprocesser = joblib.load("model/app_preprocesser.pkl")



PROGRESS_BAR_MIN_VALUE = 4


controls = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Label("Age (years)",class_name="bold", width=5),
                dbc.Col(
                    [dbc.Input(
                        id='age-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['age','min'],df_limits.loc['age','max']),
                ),
                ]),
            ],
            className='mb-2'
        ),
        
        
         dbc.Row(
            [
                dbc.Label("Weight (kg)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='weight-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['weight','min'],df_limits.loc['weight','max']),
                )),
            ],
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label("KCCQ summary score", width=5),
                dbc.Col(
                    dbc.Input(
                        id='kccq-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['kccq_summ_bl','min'],df_limits.loc['kccq_summ_bl','max']),
                )),
            ],
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label("Haemoglobin (g/L)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='haemo-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['haemoglobin_adj','min'],df_limits.loc['haemoglobin_adj','max']),
                )),
            ],
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label("Creatinine (µmol/L)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='creatinine-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['creatinine','min'],df_limits.loc['creatinine','max']),
                )),
            ],
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label("Albumin (g/L)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='albumin-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['albumin_adj','min'],df_limits.loc['albumin_adj','max']),
                )),
            ],
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label(
                    html.Span("LVEF (%)",id="tooltip-target",style={"textDecoration": "underline", "cursor": "help",'text-decoration-style': 'dotted','text-underline-offset':'0.3rem'}), 
                    width=5,color='primary'),
                dbc.Tooltip(html.P([
                    html.Span("Left ventricular ejection fraction on presentation.")
                    ]),
                    target="tooltip-target"),
                dbc.Col(
                    dbc.Input(
                        id='lvef-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['lvef_value','min'],df_limits.loc['lvef_value','max']),
                )),
            ],
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label("Aortic valve area (cm²)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='av-area-input', 
                        type='number',
                        placeholder="{}-{}".format(df_limits.loc['av_area_tte','min'],df_limits.loc['av_area_tte','max']),
                )),
            ],
            className='mb-2'
        ),
        dbc.Row([
            
            dbc.Col([
                dbc.Label('Sex', width=5),
                dbc.RadioItems(
                    id='sex-input',
                    options=['Male', 'Female'], 
                    value='Male',
                    inline=True)
            ], width=5)],
            className='mb-3'
        ),
       
        html.Div(
            [
                dbc.Label('Chronic lung disease'),
                dbc.RadioItems(
                    id='chronic-lung-input',
                    options=['None', 'Mild', 'Moderate','Severe'], 
                    value='None',
                    inline=True
                ),
            ],
            className='mb-3'
        ),
       
        html.Div(
            [
                dbc.Label('Prior atrial fibrillation/flutter'),
                dbc.Switch(
                    id='af-input',
                    value=False,
                ),
            ],
            className='mb-2'
        ),
        html.Div(
            [
                dbc.Label('Peripheral arterial disease'),
                dbc.Switch(
                    id="pad-input",
                    value=False,
                )
            ],
            className='mb-2'
        ),
        html.Hr(),
        html.Div(
            [
               dbc.Button(
                "Calculate", id="example-button", className="d-grid gap-2 col-6 mx-auto", color='primary',n_clicks=0
        ),
            ]
        ),
    ],
    body=True,
)


mortality_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("Inhospital", className="card-subtitle mb-2"),
            dbc.Progress(
                value=PROGRESS_BAR_MIN_VALUE, id="inhospital-mortality-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            ),
            html.H6("30 Day", className="card-subtitle mb-2 mt-2"),
            dbc.Progress(
                value=PROGRESS_BAR_MIN_VALUE, id="month-mortality-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            ),
            html.H6("1 Year", className="card-subtitle mb-2 mt-2"),
            dbc.Progress(
                value=PROGRESS_BAR_MIN_VALUE, id="mortality-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            ),
            dcc.Loading(
                id="loading-1",
                children=[dcc.Graph(id="graph",figure=default_fig)],
                type="circle")
        ]
    ),
    className="mb-3"
)





risk_score = dbc.Card(
    [
        dbc.CardBody([
            html.H4("Survival Rate", className="card-title"),
            html.Hr(),
            mortality_result,
            
        ])
    ]
)



layout = html.Div(
    [

    dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(lg=1),
                dbc.Col(controls, lg=5),
                dbc.Col([
                    risk_score
                ], lg=5),
                #dbc.Col(dcc.Graph(id='graph-content'), md=4)

            ],
            align="center",
        ),
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


@callback(
    #Output('table-content', 'children'),
    Output('mortality-prob', 'value'),
    Output('mortality-prob', 'label'),
    Output('month-mortality-prob', 'value'),
    Output('month-mortality-prob', 'label'),
    Output('inhospital-mortality-prob', 'value'),
    Output('inhospital-mortality-prob', 'label'),
    Output("graph", "figure"), 
    Input('example-button', 'n_clicks'),
    State('age-input', 'value'),
    State('sex-input', 'value'),
    State('weight-input', 'value'),
    State('kccq-input', 'value'),
    State('haemo-input', 'value'),
    State('creatinine-input', 'value'),
    State('albumin-input', 'value'),
    State('lvef-input', 'value'),
    State('av-area-input', 'value'),
    State('chronic-lung-input', 'value'),
    State('af-input', 'value'),
    State('pad-input', 'value'),
    prevent_initial_call=True
)
def predict_risk(n_clicks,age,sex,weight,kccq,haemo,creatinine,albumin,lvef,av_area,chronic_lung,af,pad):
    first_row = 0
 

    if (check_age_validity(age,n_clicks) or  
        check_weight_validity(weight) or  
        check_kccq_validity(kccq) or
        check_haemoglobin_validity(haemo) or
        check_creatinine_validity(creatinine) or
        check_albumin_validity(albumin) or
        check_lvef_validity(lvef) or
        check_av_area_validity(av_area) 
        ):
        return (
        PROGRESS_BAR_MIN_VALUE,
        "",
        PROGRESS_BAR_MIN_VALUE,
        "",
        PROGRESS_BAR_MIN_VALUE,
        "",
        default_fig
    )


    df_template.loc[first_row,'age'] = age
    df_template.loc[first_row,'weight'] = weight
    df_template.loc[first_row,'kccq_summ_bl'] = kccq
    df_template.loc[first_row,'haemoglobin_adj'] = haemo
    df_template.loc[first_row,'creatinine'] = creatinine
    df_template.loc[first_row,'albumin_adj'] = albumin
    df_template.loc[first_row,'lvef_value'] = lvef
    df_template.loc[first_row,'av_area_tte'] = av_area
    


    # dummy variables
    if chronic_lung == 'Severe':
        df_template.loc[first_row,'chronic_lung_4.0'] = 1
        df_template.loc[first_row, 'chronic_lung_3.0'] = 0
    elif chronic_lung == 'Moderate':
        df_template.loc[first_row, 'chronic_lung_4.0'] = 0
        df_template.loc[first_row, 'chronic_lung_3.0'] = 1
    else:
        df_template.loc[first_row, 'chronic_lung_4.0'] = 0
        df_template.loc[first_row, 'chronic_lung_3.0'] = 0


    
    if af:
        df_template.loc[first_row, 'prior_af_1.0'] = 1
    else:
        df_template.loc[first_row, 'prior_af_1.0'] = 0

    if pad:
        df_template.loc[first_row, 'prior_pad_1.0'] = 1
    else:
        df_template.loc[first_row, 'prior_pad_1.0'] = 0

    if pad:
        df_template.loc[first_row, 'male'] = 1
    else:
        df_template.loc[first_row, 'male'] = 0
    
    
    data = preprocess_data(df_template.astype(float))
    print(data)
    survival_curve = mortality_year_pipe.predict_survival_function(data)

    year_index = np.where(survival_curve[0].x == 365)[0][0]
    year_mortality_value = round(survival_curve[0].y[year_index],4)
    year_mortality_label = "{:.1%}".format(year_mortality_value)

    month_index = np.where(survival_curve[0].x == 30)[0][0]
    month_mortality_value = round(survival_curve[0].y[month_index],4)
    month_mortality_label = "{:.1%}".format(month_mortality_value)

    inhospital_mortality_value = round(survival_curve[0].y[0],4)
    inhospital_mortality_label = "{:.1%}".format(inhospital_mortality_value)

    
    print(year_mortality_value)


    if year_mortality_value < PROGRESS_BAR_MIN_VALUE/100:
        year_mortality_value = PROGRESS_BAR_MIN_VALUE/100

    if month_mortality_value < PROGRESS_BAR_MIN_VALUE/100:
        month_mortality_value = PROGRESS_BAR_MIN_VALUE/100

    
    two_year_index = np.where(survival_curve[0].x <= 400)[0]
    
    fig = px.line(
        x=survival_curve[0].x[two_year_index], y=survival_curve[0].y[two_year_index],labels={'x': 'Days since Discharge', 'y':'Probability of Survival'})


    return (
        #dbc.Table.from_dataframe(df_template),
        year_mortality_value*100,
        year_mortality_label,
        month_mortality_value*100,
        month_mortality_label,
        inhospital_mortality_value*100,
        inhospital_mortality_label,
        fig
    )



def preprocess_data(df):

    sub_df = df[continuous_features]
    sub_df = sub_df.rename(columns={'haemoglobin_adj':'haemoglobin','albumin_adj':'albumin'})
    preprocessed_df = pd.DataFrame(preprocesser.transform(sub_df),columns=preprocesser.get_feature_names_out())
    combined_df = pd.concat([preprocessed_df,df[categorical_features]],axis=1)

    combined_df = combined_df.rename(columns={'haemoglobin':'haemoglobin_adj','albumin':'albumin_adj'})
    combined_df['haemoglobin_adj'] = combined_df.apply(utils.exponential_transformation_gender, axis=1, k=2e-2,
                                      column='haemoglobin_adj',gender_col='male',
                                     bounds={'male':(130,170),'female':(120,150)})
    
    combined_df = combined_df.drop('male',axis=1)[df_limits.index] # reorder columns according to model 
    
    print(combined_df)
    return combined_df

@callback(
    Output("age-input", "invalid"),
    Input("age-input", "value"),
    Input('example-button', 'n_clicks'),
)
def check_age_validity(value,n_clicks):
    if n_clicks == 0:
        raise PreventUpdate 
    if value:
        is_invalid = value < df_limits.loc['age','min'] or value > df_limits.loc['age','max']
        return is_invalid
    elif value is None:
        return True
    return False

@callback(
    Output("weight-input", "invalid"),
    Input("weight-input", "value")
)
def check_weight_validity(value):
    if value:
        is_invalid = value < df_limits.loc['weight','min'] or value > df_limits.loc['weight','max']
        return is_invalid
    return False


@callback(
    Output("kccq-input", "invalid"),
    Input("kccq-input", "value")
)
def check_kccq_validity(value):
    if value:
        is_invalid = value < df_limits.loc['kccq_summ_bl','min'] or value > df_limits.loc['kccq_summ_bl','max']
        return is_invalid
    return False

@callback(
    Output("haemo-input", "invalid"),
    Input("haemo-input", "value")
)
def check_haemoglobin_validity(value):
    if value:
        is_invalid = value < df_limits.loc['haemoglobin_adj','min'] or value > df_limits.loc['haemoglobin_adj','max']
        return is_invalid
    return False
@callback(
    Output("creatinine-input", "invalid"),
    Input("creatinine-input", "value")
)
def check_creatinine_validity(value):
    if value:
        is_invalid = value < df_limits.loc['creatinine','min'] or value > df_limits.loc['creatinine','max']
        return is_invalid
    return False
@callback(
    Output("albumin-input", "invalid"),
    Input("albumin-input", "value")
)
def check_albumin_validity(value):
    if value:
        is_invalid = value < df_limits.loc['albumin_adj','min'] or value > df_limits.loc['albumin_adj','max']
        return is_invalid
    return False
@callback(
    Output("lvef-input", "invalid"),
    Input("lvef-input", "value")
)
def check_lvef_validity(value):
    if value:
        is_invalid = value < df_limits.loc['lvef_value','min'] or value > df_limits.loc['lvef_value','max']
        return is_invalid
    return False
@callback(
    Output("av-area-input", "invalid"),
    Input("av-area-input", "value")
)
def check_av_area_validity(value):
    if value:
        is_invalid = value < df_limits.loc['av_area_tte','min'] or value > df_limits.loc['av_area_tte','max']
        return is_invalid
    return False

#if __name__ == '__main__':
#    app.run(debug=True)
