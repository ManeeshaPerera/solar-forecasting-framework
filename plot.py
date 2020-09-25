import plotly.graph_objs as go
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator


def plot_fc(fc_data_frame, actual):
    figure = go.Figure()
    color = px.colors.qualitative.Vivid

    raw_symbols = SymbolValidator().values
    symbols = []

    for i in range(0, len(raw_symbols), 2):
        name = raw_symbols[i + 1]
        symbols.append(raw_symbols[i])

    methods = fc_data_frame.columns
    name_conversion = {
        "sn": 'SN',
        "mlr": 'MLR',
        "svr": 'SVR',
        "(s)arima": '(s)ARIMA',
        "(s)arimax": '(s)ARIMAX',
        "pso": "PSO",
        "pso [0,1]": "PSO [0,1]",
        "pso- convex": "PSO -convex",
        "average": "Average",
        "re": "RE",
        "lgb": "LightGBM"
    }

    symbol_conversion = {
        "sn": 0,
        "mlr": 4,
        "svr": 8,
        "(s)arima": 12,
        "(s)arimax": 16,
        "pso": 20,
        "pso [0,1]": 24,
        "pso- convex": 28,
        "average": 32,
        "re": 36,
        "lgb": 40
    }

    color_index = 0
    figure.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual['solarpower'],
            marker_color='black',
            name='actual'
        )
    )

    for method in methods:
        figure.add_trace(
            go.Scatter(
                x=fc_data_frame.index,
                y=fc_data_frame[method],
                name=name_conversion[method],
                line=dict(color=color[color_index], width=1, dash='dash'),
                marker_symbol=symbols[symbol_conversion[method]]
            )
        )

        color_index = color_index + 1

    figure.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400,
                         width=800,
                         margin=go.layout.Margin(
                             l=50,
                             r=20,
                             b=5,
                             t=20,
                             pad=0
                         ),
                         font=dict(size=11), showlegend=True)
    figure.update_xaxes(title='Date/ Time', gridcolor="rgb(240, 240, 240)",
                        zerolinecolor="rgb(240, 240, 240)")
    figure.update_yaxes(title='Generation (kW)', gridcolor="rgb(240, 240, 240)",
                        zerolinecolor="rgb(240, 240, 240)")

    return figure
