import sys
from pathlib import Path

from dash import Dash, dcc, html

# A direct file run has no package context, so make the project package visible.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Hoa.forecast_app.forecast_figure import build_figure

app = Dash(__name__)
app.title = "Dự báo enrollment Coursera"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px", "fontFamily": "sans-serif"},
    children=[
        html.H2("Dự đoán số người học các khóa Coursera"),
        html.P("Mô hình dự đoán trên 20% dữ liệu kiểm tra. "
               "Kết quả chỉ để tham khảo, không phải dự báo theo thời gian."),
        dcc.Graph(id="forecast-chart", figure=build_figure()),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
