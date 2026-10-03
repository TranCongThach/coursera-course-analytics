"""Interactive Coursera dashboard. Run: python -m Thach.dashboard."""
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

# Khi chạy trực tiếp `python Thach/dashboard.py`, Python chỉ đưa thư mục Thach
# vào sys.path. Thêm project root để các package Thach và Hoa import được.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Thach import dashboard_utils as du
from Hoa import prediction as pred

FACT, BRIDGES = du.load_data()
MODEL = pred.build_enrollment_model(FACT)
MODEL_ACTUAL_FIG, MODEL_RESIDUAL_FIG = [du.style_figure(f) for f in pred.enrollment_diagnostic_figures(MODEL)]
MODEL_ACTUAL_FIG.update_layout(title_text=du.chart_title(
    "Độ chính xác trên tập kiểm tra",
    f"So sánh thực tế và ước lượng trên {MODEL.test_rows:,} khóa",
))
MODEL_RESIDUAL_FIG.update_layout(title_text=du.chart_title(
    "Phân bố sai số dự đoán",
    f"Sai số log10 của {MODEL.test_rows:,} khóa trong tập kiểm tra",
))
HOURS_MAX = int(np.ceil(FACT.hours_to_complete.max()))
ENROLL_MAX = int(np.ceil(np.log10(FACT.enrolled_num.max())))
GRAPH_CONFIG = {"displaylogo": False, "responsive": True,
                "toImageButtonOptions": {"format": "png", "scale": 2}}
GRAPH_HEIGHT = du.CHART_HEIGHT
GRAPH_STYLE = {"width": "100%", "height": f"{GRAPH_HEIGHT}px"}
CHART_KEYS = ["level", "coverage", "histogram", "rating", "tree", "map", "scatter", "heatmap", "ecdf", "organizations"]
TABLE_LABELS = ["Mã", "Tên khóa học", "Tổ chức", "Trình độ", "Điểm / 5", "Người học", "Lượt đánh giá", "Giờ học"]
DRILL_LABELS = {"level_clean": "Trình độ", "subject": "Chủ đề", "skill": "Kỹ năng",
                "organization_hq_country": "Trụ sở", "Organization": "Tổ chức"}
PAGE_COPY = {
    "overview": ("Tổng quan khóa học Coursera", "Theo dõi quy mô, mức độ đầy đủ và cơ cấu của dữ liệu đang chọn."),
    "insight": ("Phân tích mối liên hệ dữ liệu", "So sánh các chỉ số trên cùng phạm vi khóa học, có ghi rõ cỡ mẫu sử dụng."),
    "prediction": ("Ước lượng số người học", "Nhập đặc điểm khóa học để tham khảo mức người học và khoảng dự đoán của mô hình."),
    "data": ("Danh mục dữ liệu khóa học", "Kiểm tra từng bản ghi và mở nguồn Coursera khi cần đối chiếu."),
}


def field(label, component):
    return html.Div([html.Label(label), component], className="field")


def kpi(value, label, note=""):
    return html.Div([html.Div(label, className="kpi-label"), html.Strong(value),
                     html.Small(note)], className="kpi")


def card(key, note, extra=None):
    return html.Section([
        *([extra] if extra is not None else []),
        dcc.Graph(id=f"chart-{key}", config=GRAPH_CONFIG, responsive=True,
                  style=GRAPH_STYLE, figure=du.empty_figure("Đang tải dữ liệu…")),
        html.P(note, className="chart-note"),
    ], className="chart-card")


def coefficient_figure():
    # Chiều dài thể hiện độ lớn; màu và nhãn giữ lại chiều tác động.
    # Nhờ đó, cả hệ số âm và dương đều hướng sang phải, không đè lên nhãn.
    labels = ["Lượt đánh giá tăng 10 lần", "Điểm đánh giá +1", "Giờ học tăng 10 lần", "Thêm 1 kỹ năng",
              "Intermediate / Beginner", "Advanced / Beginner", "Not specified / Beginner"]
    signed_values = np.asarray(MODEL.coefficients[1:], dtype=float)
    magnitudes = np.abs(signed_values)
    fig = go.Figure()
    for name, color, selected in (
        ("Dương (+)", "#12a594", signed_values >= 0),
        ("Âm (−)", "#e5484d", signed_values < 0),
    ):
        fig.add_trace(go.Bar(
            x=[value if keep else None for value, keep in zip(magnitudes, selected, strict=True)],
            y=labels,
            orientation="h",
            name=name,
            customdata=[value if keep else None for value, keep in zip(signed_values, selected, strict=True)],
            marker_color=color,
            marker_line=dict(color="white", width=1),
            text=[f"{value:+.3f}" if keep else None for value, keep in zip(signed_values, selected, strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Hệ số: %{customdata:+.4f}<br>Độ lớn: %{x:.4f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        title=du.chart_title(
            "Hệ số liên hệ của mô hình",
            "Mọi thanh cùng hướng sang phải · Thanh dài hơn = liên hệ mạnh hơn · Beginner là nhóm tham chiếu",
        ),
        xaxis_title="Độ lớn tuyệt đối của hệ số trên thang log10",
        barmode="overlay",
        bargap=0.28,
        legend=dict(
            title_text="Chiều liên hệ",
            orientation="h",
            x=0,
            y=-0.25,
            yanchor="top",
        ),
    )
    fig.update_xaxes(range=[0, magnitudes.max() * 1.18], rangemode="tozero")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return du.style_figure(fig)


app = Dash(__name__, title="Phân tích xu hướng học trực tuyến", assets_folder=str(Path(__file__).parent / "assets"))
GRAPH_CONFIG["topojsonURL"] = app.get_asset_url("plotly-topojson/")
server = app.server

app.layout = html.Div([
    dcc.Store(id="filter-state", data={}), dcc.Store(id="drill-state", data={}),
    html.Div(id="resize-hack", style={"display": "none"}),
    html.Aside([
        html.Div([
            html.Div(html.Img(
                id="hcmute-logo",
                src=app.get_asset_url("Logo HCM-UTE_ 1 (1).png"),
                alt="Logo Trường Đại học Sư phạm Kỹ thuật TP.HCM",
            ), className="logo-box"),
            html.Span([html.Strong("Coursera"), html.Small("Course Analytics")], className="brand-text"),
        ], className="brand"),
        html.Div(className="sidebar-divider"),
        html.Div([
            html.Span("ĐỀ TÀI", className="project-label"),
            html.Strong("Phân tích xu hướng học trực tuyến"),
            html.Small("Phân tích dữ liệu khóa học Coursera"),
        ], id="project-identity", className="project-identity"),
        html.Div("ĐIỀU HƯỚNG", className="nav-label"),
        dcc.Tabs(id="page-tab", value="overview", vertical=True, className="tabs-vertical", children=[
            dcc.Tab(label="Tổng quan", value="overview", className="nav-tab", selected_className="nav-tab-selected"),
            dcc.Tab(label="Phân tích", value="insight", className="nav-tab", selected_className="nav-tab-selected"),
            dcc.Tab(label="Dự đoán người học", value="prediction", className="nav-tab", selected_className="nav-tab-selected"),
            dcc.Tab(label="Dữ liệu", value="data", className="nav-tab", selected_className="nav-tab-selected"),
        ]),
    ], id="app-sidebar", className="sidebar"),
    html.Main([
        html.Div([
            html.H1(PAGE_COPY["overview"][0], id="page-title"),
            html.P(PAGE_COPY["overview"][1], id="page-subtitle", className="muted"),
        ], id="page-intro", className="page-intro"),
        html.Section([
            html.Div([
                html.Div([
                    html.Span("PHẠM VI PHÂN TÍCH", className="filter-kicker"),
                    html.Strong("Bộ lọc khóa học"),
                    html.Div(id="breadcrumb", className="breadcrumb"),
                ], className="filter-heading"),
                html.Div([
                    html.Button("Xóa lựa chọn", id="clear-drill", n_clicks=0, className="button quiet-button small-button"),
                    html.Button("Đặt lại bộ lọc", id="reset-filters", className="button secondary small-button", n_clicks=0),
                ], className="filter-toolbar-actions"),
            ], className="filter-toolbar"),
            html.Details([
                html.Summary([
                    html.Span("Điều chỉnh bộ lọc"),
                    html.Small("Tổ chức, trình độ và phạm vi chỉ số"),
                ]),
                html.Div([
                    html.Section([
                        html.Div([
                            html.H4("Danh mục khóa học"),
                            html.P("Có thể chọn nhiều giá trị trong cùng một nhóm."),
                        ], className="filter-section-heading"),
                        html.Div([
                            field("Tổ chức", dcc.Dropdown(id="filter-organizations", options=sorted(FACT.Organization.dropna().unique()), multi=True, placeholder="Tất cả tổ chức")),
                            field("Trình độ", dcc.Dropdown(id="filter-levels", options=du.LEVELS, multi=True, placeholder="Tất cả trình độ")),
                            field("Chủ đề", dcc.Dropdown(id="filter-subjects", options=du.bridge_options(BRIDGES["subject"], "subject"), multi=True, placeholder="Tất cả chủ đề")),
                            field("Kỹ năng", dcc.Dropdown(id="filter-skills", options=du.bridge_options(BRIDGES["skill"], "skill"), multi=True, placeholder="Tất cả kỹ năng")),
                        ], className="filter-category-grid"),
                    ], className="filter-section"),
                    html.Section([
                        html.Div([
                            html.H4("Phạm vi chỉ số"),
                            html.P("Kéo hai đầu thanh để giới hạn khoảng dữ liệu."),
                        ], className="filter-section-heading"),
                        html.Div([
                            field("Điểm đánh giá", dcc.RangeSlider(id="filter-rating", min=0, max=5, step=0.1, value=[0, 5], marks={0: "0", 3: "3", 5: "5"}, tooltip={"placement": "bottom"})),
                            field("Giờ học", dcc.RangeSlider(id="filter-hours", min=0, max=HOURS_MAX, step=0.1, value=[0, HOURS_MAX], marks={0: "0", HOURS_MAX // 2: str(HOURS_MAX // 2), HOURS_MAX: str(HOURS_MAX)}, tooltip={"placement": "bottom"})),
                            field("Người học (log10)", dcc.RangeSlider(id="filter-enrollment", min=0, max=ENROLL_MAX, step=0.05, value=[0, ENROLL_MAX], marks={0: "1", 3: "1 nghìn", 6: "1 triệu"}, tooltip={"placement": "bottom"})),
                        ], className="filter-range-grid"),
                    ], className="filter-section"),
                    html.Div([
                        html.Div([
                            html.Strong("Tùy chọn dữ liệu"),
                            html.Small("Kiểm soát cách xử lý các giá trị chưa đầy đủ."),
                        ], className="filter-options-copy"),
                        dcc.Checklist(id="filter-options", options=[
                            {"label": " Giữ khóa thiếu dữ liệu", "value": "missing"},
                            {"label": " Chỉ dùng giờ chính thức", "value": "reported"}], value=["missing"], className="check-options"),
                    ], className="filter-options-row"),
                ], className="filter-grid"),
            ], open=False, className="filter-details"),
        ], id="filter-panel", className="panel filter-panel"),
        html.Div([
            html.Div(id="kpi-row", className="kpi-grid"),
            html.Div([html.H2("Bức tranh dữ liệu chính"), html.P("Bấm vào cột, ô hoặc quốc gia để lọc danh sách trong tab Dữ liệu.")], className="section-heading"),
            html.Div([
                card("level", "“Chưa rõ” là khóa thiếu thông tin. Bấm vào cột để xem chi tiết."),
                card("coverage", "Mức độ đầy đủ của dữ liệu lịch học."),
                card("histogram", "Chỉ tính các khóa có số người học."),
                card("rating", "Hộp thể hiện trung vị và độ phân tán của từng nhóm."),
                card("tree", "Một khóa có thể thuộc nhiều nhóm. Bấm vào ô để lọc.",
                     dcc.RadioItems(id="tree-kind", options=[{"label": " Chủ đề", "value": "subject"}, {"label": " Kỹ năng", "value": "skill"}], value="subject", inline=True)),
                card("map", "Vị trí trụ sở tổ chức, không phải nơi ở của người học."),
                card("organizations", "Bấm vào tổ chức để xem danh sách khóa học."),
            ], className="chart-grid"),
        ], id="overview-panel"),
        html.Div([
            html.Div([html.H2("Các chỉ số đi cùng nhau như thế nào?"), html.P("Mỗi biểu đồ ghi rõ số khóa thực sự được dùng sau khi loại dữ liệu thiếu.")], className="section-heading"),
            html.Div([
                card("scatter", "Mối liên hệ không đồng nghĩa với quan hệ nhân quả."),
                card("heatmap", "Ô trống là không đủ dữ liệu để tính."),
                card("ecdf", "Tỷ lệ khóa có số người học thấp hơn từng ngưỡng."),
            ], className="chart-grid"),
            html.Details([
                html.Summary("Xem thêm: so sánh hai chỉ số"),
                html.P("Kết quả bên dưới chỉ để tham khảo, không dùng để kết luận chắc chắn.", className="muted"),
                html.Div([
                    field("Chỉ số ngang", dcc.Dropdown(id="x-var", options=[{"label": v, "value": k} for k, v in pred.COURSE_LABELS.items()], value="hours_to_complete", clearable=False)),
                    field("Chỉ số dọc", dcc.Dropdown(id="y-var", options=[{"label": v, "value": k} for k, v in pred.COURSE_LABELS.items()], value="enrolled_num", clearable=False)),
                    field("Thang log", dcc.Checklist(id="log-opts", options=[{"label": " Log trục ngang", "value": "x"}, {"label": " Log trục dọc", "value": "y"}], value=["x", "y"], inline=True)),
                    field("Giá trị muốn dự đoán, cách nhau dấu phẩy", dcc.Input(id="x-new", type="text", value="", debounce=True)),
                ], className="form-grid"),
                dcc.Graph(id="fig-trend", config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), html.P(id="trend-summary"), html.Div(id="trend-warnings"),
                dash_table.DataTable(id="trend-table", style_table={"overflowX": "auto"}),
            ], className="panel exploration"),
        ], id="insight-panel", style={"display": "none"}),
        html.Div([
            html.Div([
                html.Div([
                    html.Span("MÔ HÌNH SNAPSHOT", className="model-context-label"),
                    html.Strong("Hồi quy trên dữ liệu Coursera hiện có"),
                    html.P("Mô hình mô tả quan hệ trong cùng một thời điểm, không phải dự báo tăng trưởng theo thời gian."),
                ]),
                html.Div([
                    html.Span(f"Huấn luyện {MODEL.train_rows:,}"),
                    html.Span(f"Hiệu chỉnh {MODEL.calibration_rows:,}"),
                    html.Span(f"Kiểm tra {MODEL.test_rows:,}"),
                ], className="model-split"),
            ], className="model-context"),
            html.Div([
                kpi(f"{MODEL.eligible_rows / MODEL.total_rows:.1%}", "Độ phủ dữ liệu", f"{MODEL.eligible_rows:,} / {MODEL.total_rows:,} khóa"),
                kpi(f"{MODEL.metrics['r2']:.3f}", "R² trên tập kiểm tra", f"Baseline chỉ review: {MODEL.baseline_metrics['r2']:.3f}"),
                kpi(f"{MODEL.metrics['rmse_log10']:.3f}", "RMSE (log10)", f"MAE: {MODEL.metrics['mae_log10']:.3f}"),
                kpi(f"{MODEL.metrics['interval_coverage']:.1%}", "Bao phủ khoảng 95%", f"Đánh giá trên {MODEL.test_rows:,} khóa"),
            ], className="kpi-grid model-kpis"),
            html.Section([
                html.Div([
                    html.H3("Thông tin khóa học"),
                    html.P("Các trường có dấu * cần nằm trong phạm vi hợp lệ của dữ liệu."),
                    html.Div([
                        field("Số lượt đánh giá *", dcc.Input(id="model-reviews", type="number", value=1000, min=1, step=1, debounce=True)),
                        field("Điểm đánh giá / 5 *", dcc.Input(id="model-rating", type="number", value=4.6, min=0, max=5, step=0.1, debounce=True)),
                        field("Số giờ học *", dcc.Input(id="model-hours", type="number", value=20, min=0.1, debounce=True)),
                        field("Số kỹ năng", dcc.Input(id="model-skills", type="number", value=6, min=0, step=1, debounce=True)),
                        field("Trình độ", dcc.Dropdown(id="model-level", options=list(pred.LEVELS), value="Beginner", clearable=False)),
                    ], className="model-input-grid"),
                ], className="model-inputs"),
                html.Div([
                    html.Div(id="model-estimate", className="estimate"),
                    html.Div(id="model-warning", className="warning"),
                ], className="model-result"),
            ], className="panel model-workbench"),
            html.Div([html.H2("Kiểm tra chất lượng mô hình"), html.P("Ba biểu đồ dưới đây dùng tập kiểm tra, không dùng dữ liệu đang nhập ở trên.")], className="section-heading model-diagnostics-heading"),
            html.Div([
                html.Section(dcc.Graph(figure=MODEL_ACTUAL_FIG, config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), className="chart-card"),
                html.Section(dcc.Graph(figure=MODEL_RESIDUAL_FIG, config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), className="chart-card"),
                html.Section(dcc.Graph(figure=coefficient_figure(), config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), className="chart-card model-coefficient-card"),
            ], className="chart-grid"),
        ], id="prediction-panel", style={"display": "none"}),
        html.Div([
            html.Div([
                html.H2("Danh sách khóa học đã lọc"),
                html.P("Danh sách được cập nhật theo bộ lọc và lựa chọn trực tiếp trên biểu đồ."),
            ], className="section-heading"),
            html.Section([
                html.H3("Khóa học trong lựa chọn"),
                html.P(id="table-caption", className="muted"),
                dash_table.DataTable(id="course-table", columns=[{"name": label, "id": key, "type": "numeric" if key in du.NUMERIC else "text"} for key, label in zip(du.TABLE_COLUMNS, TABLE_LABELS)],
                    hidden_columns=["course_id"], data=[], page_size=10, sort_action="native", cell_selectable=True,
                    style_table={"overflowX": "auto"}, style_cell={"fontFamily": "sans-serif", "fontSize": 12, "padding": "12px", "textAlign": "left", "minWidth": "90px", "maxWidth": "300px", "whiteSpace": "normal", "border": "none", "borderBottom": "1px solid #eef1f5"},
                    style_header={"fontWeight": "600", "backgroundColor": "#f8f9fb", "color": "#344054"},
                    style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": "#eef4ff", "border": "1px solid #1769e0"}],
                    css=[{"selector": ".show-hide", "rule": "display: none"}]),
                html.Div(id="course-detail", className="course-detail", children="Bấm vào một dòng trong bảng hoặc một điểm trên biểu đồ để xem chi tiết."),
            ], className="panel course-explorer"),
        ], id="data-panel", style={"display": "none"}),
    ], id="dashboard-main", className="main"),
], className="app-shell")


@app.callback(Output("overview-panel", "style"), Output("insight-panel", "style"),
              Output("prediction-panel", "style"), Output("data-panel", "style"),
              Output("filter-panel", "style"), Output("page-title", "children"),
              Output("page-subtitle", "children"),
              Input("page-tab", "value"))
def switch_tab(tab):
    panels = ("overview", "insight", "prediction", "data")
    panel_styles = tuple({"display": "block" if tab == key else "none"} for key in panels)
    filter_style = {"display": "none"} if tab == "prediction" else {"display": "block"}
    title, subtitle = PAGE_COPY.get(tab, PAGE_COPY["overview"])
    return (*panel_styles, filter_style, title, subtitle)


# Plotly có thể giữ kích thước mặc định khi graph được tạo trong tab ẩn.
# Khi mở tab, đọc chiều cao thật của wrapper rồi tính lại khung vẽ.
app.clientside_callback(
    """
    function(tab) {
        function resizeVisibleGraphs() {
            document.querySelectorAll('.js-plotly-plot').forEach(function(graph) {
                if (graph.offsetParent === null || !window.Plotly) return;

                var wrapper = graph.closest('.dash-graph');
                var height = wrapper ? wrapper.getBoundingClientRect().height : 400;
                window.Plotly.relayout(graph, {height: height, autosize: true});
                window.Plotly.Plots.resize(graph);
            });
        }

        [0, 80, 250, 600].forEach(function(ms) {
            setTimeout(resizeVisibleGraphs, ms);
        });
        return tab;
    }
    """,
    Output("resize-hack", "children"),
    Input("page-tab", "value"),
)


@app.callback(Output("filter-state", "data"),
    Input("filter-organizations", "value"), Input("filter-levels", "value"), Input("filter-subjects", "value"), Input("filter-skills", "value"),
    Input("filter-rating", "value"), Input("filter-hours", "value"), Input("filter-enrollment", "value"), Input("filter-options", "value"))
def store_filters(organizations, levels, subjects, skills, rating, hours, enrollment, options):
    return dict(organizations=organizations, levels=levels, subjects=subjects, skills=skills,
                rating=rating, hours=hours, enrollment=enrollment,
                include_missing="missing" in (options or []), reported_only="reported" in (options or []))


@app.callback(*[Output(f"filter-{key}", "value") for key in ("organizations", "levels", "subjects", "skills", "rating", "hours", "enrollment", "options")], Input("reset-filters", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [], [0, 5], [0, HOURS_MAX], [0, ENROLL_MAX], ["missing"]


@app.callback(Output("drill-state", "data"),
    Input("chart-level", "clickData"), Input("chart-tree", "clickData"), Input("chart-map", "clickData"), Input("chart-organizations", "clickData"),
    Input("clear-drill", "n_clicks"), Input("reset-filters", "n_clicks"), Input("tree-kind", "value"), State("drill-state", "data"), prevent_initial_call=True)
def update_drill(level, tree, country, organization, clear, reset, kind, current):
    trigger = ctx.triggered_id
    if trigger in ("clear-drill", "reset-filters"):
        return {}
    result = dict(current or {})
    if trigger == "tree-kind":
        result.pop("subject", None)
        result.pop("skill", None)
        return result
    event = {"chart-level": level, "chart-tree": tree, "chart-map": country, "chart-organizations": organization}.get(trigger)
    if not event or not event.get("points"):
        return no_update
    point = event["points"][0]
    if trigger == "chart-level":
        result["level_clean"] = point["x"]
    elif trigger == "chart-tree":
        # Plotly IDs are stable normalized labels. Avoid customdata on the virtual
        # treemap root: Dash 4.4's hover serializer cannot handle its null pointNumber.
        value = point.get("id")
        if not isinstance(value, str) or value not in set(BRIDGES[kind][kind + "_normalized"]):
            return no_update
        result[kind] = value
    elif trigger == "chart-map":
        inverse = {v: k for k, v in du.COUNTRY_ISO.items()}
        if point.get("location") not in inverse:
            return no_update
        result["organization_hq_country"] = inverse[point["location"]]
    else:
        result["Organization"] = point["y"]
    return result


@app.callback(*[Output(f"chart-{key}", "figure") for key in CHART_KEYS],
    Output("kpi-row", "children"), Output("course-table", "data"), Output("course-table", "page_current"),
    Output("breadcrumb", "children"), Output("table-caption", "children"),
    Input("filter-state", "data"), Input("drill-state", "data"), Input("tree-kind", "value"))
def update_dashboard(filters, drill, kind):
    df = du.filter_courses(FACT, filters or {}, BRIDGES, drill)
    figs = du.build_figures(df, BRIDGES, kind)
    n = len(df)
    enrolled, ratings = df.enrolled_num.dropna(), df.rating_num.dropna()
    kpis = [kpi(f"{n:,}", "Khóa học đang xem", f"Trên {len(FACT):,} khóa trong dữ liệu"),
            kpi(f"{enrolled.median():,.0f}" if len(enrolled) else "—", "Người học (mức giữa)", f"Có dữ liệu: {len(enrolled):,}/{n:,}"),
            kpi(f"{ratings.mean():.2f} / 5" if len(ratings) else "—", "Điểm đánh giá trung bình", f"Có dữ liệu: {len(ratings):,}/{n:,}"),
             kpi(f"{df.Organization.nunique():,}", "Tổ chức", f"Chưa rõ tổ chức: {df.Organization.isna().sum():,} khóa")]
    breadcrumb = "Đang xem" + "".join(f"  ›  {DRILL_LABELS[k]}: {v}" for k, v in (drill or {}).items())
    return (*[figs[key] for key in CHART_KEYS], kpis, du.table_records(df), 0, breadcrumb,
            f"Có {n:,} khóa phù hợp. Bấm vào một dòng để xem chi tiết.")


def render_detail(course_id, df):
    selected = df[df.course_id.eq(course_id)]
    if selected.empty:
        return "Khóa đã chọn không còn trong điều kiện lọc. Hãy chọn lại trong bảng hoặc biểu đồ."
    row = selected.iloc[0]
    def display(column):
        value = row[column]
        return "Chưa có dữ liệu" if pd.isna(value) else f"{value:,.2f}" if isinstance(value, (float, int, np.number)) else str(value)
    facts = [html.Div([html.Small(label), html.Strong(display(column))]) for column, label in (
        ("Organization", "Tổ chức"), ("level_clean", "Trình độ"), ("rating_num", "Điểm / 5"),
        ("enrolled_num", "Người học"), ("num_reviews", "Lượt đánh giá"), ("hours_to_complete", "Giờ học"))]
    items = []
    for kind, kind_label in (("subject", "Chủ đề"), ("skill", "Kỹ năng")):
        names = BRIDGES[kind].loc[BRIDGES[kind].course_id.eq(course_id), kind].dropna().unique()
        items.append(html.P(f"{kind_label}: " + (", ".join(names) if len(names) else "Chưa có dữ liệu")))
    url = str(row.URL) if pd.notna(row.URL) else ""
    link = html.A("Mở trang gốc của khóa học", href=url, target="_blank", rel="noopener noreferrer", className="course-link") if urlparse(url).scheme in ("https", "http") else None
    estimated = row.hours_to_complete_is_estimated
    hours_note = "Số giờ học này là ước lượng, không phải số chính thức." if pd.notna(estimated) and bool(estimated) else ""
    return [html.Div("CHI TIẾT KHÓA HỌC", className="eyebrow"), html.H3(row.title),
            html.Div(facts, className="detail-grid"), html.P(hours_note, className="warning"), *items, link]


@app.callback(Output("course-detail", "children"), Input("chart-scatter", "clickData"), Input("course-table", "active_cell"), Input("filter-state", "data"), Input("drill-state", "data"))
def update_detail(scatter, cell, filters, drill):
    if ctx.triggered_id in ("filter-state", "drill-state", None):
        return "Bấm vào một dòng trong bảng hoặc một điểm trên biểu đồ để xem chi tiết."
    course_id = None
    if ctx.triggered_id == "chart-scatter" and scatter and scatter.get("points"):
        course_id = scatter["points"][0]["customdata"][0]
    elif ctx.triggered_id == "course-table" and cell:
        course_id = cell.get("row_id")
    return render_detail(course_id, du.filter_courses(FACT, filters or {}, BRIDGES, drill))


@app.callback(Output("fig-trend", "figure"), Output("trend-summary", "children"), Output("trend-warnings", "children"), Output("trend-table", "data"), Output("trend-table", "columns"),
    Input("x-var", "value"), Input("y-var", "value"), Input("log-opts", "value"), Input("x-new", "value"), Input("filter-state", "data"), Input("drill-state", "data"))
def update_trend(x_var, y_var, log_opts, x_new, filters, drill):
    try:
        chart = pred.build_course_trend(du.filter_courses(FACT, filters or {}, BRIDGES, drill), x_col=x_var, y_col=y_var,
            log_x="x" in (log_opts or []), log_y="y" in (log_opts or []), x_new=pred.parse_number_list(x_new) or None)
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        return du.empty_figure(str(exc)), "", "", [], []
    return du.style_figure(chart.fig), chart.summary, html.Ul([html.Li(w) for w in chart.warnings], className="warning"), chart.table.round(2).to_dict("records"), [{"name": c, "id": c} for c in chart.table.columns]


@app.callback(Output("model-estimate", "children"), Output("model-warning", "children"), Input("model-reviews", "value"), Input("model-rating", "value"), Input("model-hours", "value"), Input("model-skills", "value"), Input("model-level", "value"))
def update_model(reviews, rating, hours, skills, level):
    try:
        if any(value is None for value in (reviews, rating, hours, skills, level)):
            raise ValueError("Vui lòng nhập đủ các ô phía trên.")
        result, warnings = MODEL.estimate(reviews, rating, hours, skills, level)
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Số liệu nằm ngoài phạm vi của mô hình, không dự đoán được.")
    except (ValueError, TypeError, OverflowError) as exc:
        return "Chưa thể ước lượng", str(exc)
    return [html.Small("SỐ NGƯỜI HỌC DỰ ĐOÁN"), html.Strong(f"{result['estimate']:,.0f}"), html.Span(f"Khoảng dự đoán 95%: {result['lower']:,.0f} – {result['upper']:,.0f}")], html.Ul([html.Li(w) for w in warnings]) if warnings else ""


if __name__ == "__main__":
    app.run(debug=False)
