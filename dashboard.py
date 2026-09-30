
from __future__ import annotations
 
import sys
from pathlib import Path
 
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
 
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import prediction as pred  # noqa: E402
 
FACT = pred.load_fact(pred.TREND_VARIABLES)  # chỉ đọc các cột số cần cho hồi quy
VAR_OPTIONS = [{"label": pred.COURSE_LABELS[v], "value": v} for v in pred.TREND_VARIABLES]
NOTE = {"color": "#5b6570", "fontSize": "13px"}
WARN = {"color": "#b9770e", "fontSize": "13px"}
FIELD = {"width": "100%", "height": "36px"}
 
 
def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(template="plotly_white", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
 
 
def table_payload(df: pd.DataFrame):
    df = df.round(1)
    return df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]
 
 
def warning_list(items: list[str]) -> html.Ul:
    return html.Ul([html.Li(t) for t in items], style=WARN)
 
 
app = Dash(__name__, title="Dự báo xu hướng")
server = app.server
 
app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px", "fontFamily": "Segoe UI, Arial, sans-serif"},
    children=[
        html.H1("Dự báo xu hướng bằng hồi quy tuyến tính"),
 
        # ---------------- A. Đường xu hướng giữa hai biến của khóa học
        html.H2("Đường xu hướng giữa hai biến của khóa học"),
        html.Div("Hồi quy cắt ngang trên snapshot (không có trục thời gian), không hàm ý nhân quả.", style=NOTE),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "8px"}, children=[
            html.Div([html.Label("Biến x"), dcc.Dropdown(id="x-var", options=VAR_OPTIONS, value="hours_to_complete", clearable=False)]),
            html.Div([html.Label("Biến y"), dcc.Dropdown(id="y-var", options=VAR_OPTIONS, value="enrolled_num", clearable=False)]),
            html.Div([html.Label("Thang đo"), dcc.Checklist(
                id="log-opts", value=["x", "y"], inline=True,
                options=[{"label": " log10(x)", "value": "x"}, {"label": " log10(y)", "value": "y"}])]),
            html.Div([html.Label("Dự đoán y tại x = (cách nhau bởi dấu phẩy)"),
                      dcc.Input(id="x-new", type="text", debounce=True, value="10, 40, 100", style=FIELD)]),
        ]),
        dcc.Graph(id="fig-trend"),
        html.Div(id="trend-summary"),
        html.Div(id="trend-warnings"),
        dash_table.DataTable(id="trend-table", style_table={"maxWidth": "600px"}),
    ],
)
 
 
# Gọi pred.build_course_trend
@app.callback(
    Output("fig-trend", "figure"), Output("trend-summary", "children"), Output("trend-warnings", "children"),
    Output("trend-table", "data"), Output("trend-table", "columns"),
    Input("x-var", "value"), Input("y-var", "value"), Input("log-opts", "value"), Input("x-new", "value"),
)
def update_trend(x_var, y_var, log_opts, x_new):
    log_opts = log_opts or []
    try:
        chart = pred.build_course_trend(FACT, x_col=x_var, y_col=y_var, log_x="x" in log_opts, log_y="y" in log_opts,
                                        x_new=pred.parse_number_list(x_new) or None)
    except (ValueError, KeyError) as exc:
        return empty_figure(str(exc)), "", "", [], []
    data, cols = table_payload(chart.table)
    return chart.fig, chart.summary, warning_list(chart.warnings), data, cols
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
