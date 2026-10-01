from dash import Dash, dcc, html
from forecast_figure import build_figure

app = Dash(__name__)
app.title = "Dự báo enrollment Coursera"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px", "fontFamily": "sans-serif"},
    children=[
        html.H2("Dự báo lượng đăng ký khóa học Coursera"),
        html.P("Mô hình Gradient Boosting dự báo log10(enrollment) trên tập test 20%. "
               "Đây là dự báo hồi quy cắt ngang, không phải chuỗi thời gian."),
        dcc.Graph(id="forecast-chart", figure=build_figure()),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
