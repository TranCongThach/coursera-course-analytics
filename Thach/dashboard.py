import sys
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Thach import dashboard_utils as du
from Hoa import prediction as pred

FACT, BRIDGES = du.load_data()
MODEL = pred.build_enrollment_model(FACT)
MODEL_ACTUAL_FIG, MODEL_RESIDUAL_FIG = [du.style_figure(f) for f in pred.enrollment_diagnostic_figures(MODEL)]
MODEL_ACTUAL_FIG.update_layout(title_text=du.chart_title(
    "So sánh giá trị dự đoán và thực tế",
    f"Mỗi chấm là một trong {MODEL.test_rows:,} khóa dùng để kiểm tra",
))
MODEL_RESIDUAL_FIG.update_layout(title_text=du.chart_title(
    "Phân bố sai số dự đoán",
    f"Điểm càng gần 0 thì dự đoán càng sát số thật · {MODEL.test_rows:,} khóa kiểm tra",
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

def field(label, component):
    return html.Div([html.Label(label), component], className="field")

def kpi(value, label, note="", icon=None):
    icon_node = html.Span(
        html.Img(src=f"/assets/{icon}", alt="", className="kpi-icon-image") if icon else None,
        className="kpi-icon",
    )
    return html.Div([icon_node, html.Div(label, className="kpi-label"), html.Strong(value),
                     html.Small(note)], className="kpi")

def card(key, note, extra=None, wide=False):
    return html.Section([
        *([extra] if extra is not None else []),
        dcc.Graph(id=f"chart-{key}", config=GRAPH_CONFIG, responsive=True,
                  style=GRAPH_STYLE, figure=du.empty_figure("Đang tải…")),
        html.P(note, className="chart-note"),
    ], className="chart-card chart-card-wide" if wide else "chart-card")

def coefficient_figure():
    labels = [
        "Lượt đánh giá<br>tăng 10 lần",
        "Điểm đánh giá<br>tăng 1 điểm",
        "Giờ học<br>tăng 10 lần",
        "Thêm 1<br>kỹ năng",
        "Intermediate<br>so với Beginner",
        "Advanced<br>so với Beginner",
        "Chưa rõ trình độ<br>so với Beginner",
    ]
    signed_values = np.asarray(MODEL.coefficients[1:], dtype=float)
    fig = go.Figure()
    for name, color, selected in (
        ("Hệ số dương (+)", "#12a594", signed_values >= 0),
        ("Hệ số âm (−)", "#e5484d", signed_values < 0),
    ):
        fig.add_trace(go.Bar(
            x=labels,
            y=[value if keep else None for value, keep in zip(signed_values, selected, strict=True)],
            name=name,
            marker_color=color,
            marker_line=dict(color="white", width=1),
            text=[f"{value:+.3f}" if keep else None for value, keep in zip(signed_values, selected, strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Hệ số mô hình: %{y:+.4f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        title=du.chart_title(
            "Mức ảnh hưởng của các yếu tố đến số người học",
            "Dương: dự đoán tăng · Âm: dự đoán giảm · Beginner là nhóm tham chiếu",
        ),
        barmode="overlay",
        bargap=0.22,
        legend=dict(
            title_text="Chiều hệ số",
            orientation="h",
            x=0,
            y=-0.34,
            yanchor="top",
        ),
    )
    lower, upper = min(0, signed_values.min()), max(0, signed_values.max())
    padding = max(0.04, (upper - lower) * 0.12)
    fig.update_xaxes(showgrid=False, tickangle=-12, tickfont=dict(size=10.5))
    fig.update_yaxes(
        title_text="Hệ số mô hình (log10)",
        range=[lower - padding, upper + padding],
        tickformat=".2f",
        zeroline=True,
        zerolinecolor="#344054",
        zerolinewidth=1.4,
    )
    fig = du.style_figure(fig)
    fig.update_layout(margin=dict(l=72, r=34, t=82, b=112))
    return fig

app = Dash(__name__, title="Phân tích khóa học Coursera", assets_folder=str(Path(__file__).parent / "assets"))
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
            html.Span([html.Strong("Coursera"), html.Small("Phân tích khóa học")], className="brand-text"),
            html.Div([
                html.Span("DỮ LIỆU ĐANG PHÂN TÍCH", className="header-stats-label"),
                html.Div([
                    html.Span([
                        html.Strong(f"{len(FACT):,}"),
                        html.Small("khóa học"),
                    ], className="header-stat header-courses"),
                    html.Span([
                        html.Strong(f"{FACT.Organization.nunique():,}"),
                        html.Small("tổ chức"),
                    ], className="header-stat header-organizations"),
                ], className="header-stat-list"),
            ], id="header-stats", className="header-stats"),
        ], className="brand"),
        html.Div(className="sidebar-divider"),
        html.Div([
            html.Strong("DASHBOARD"),
            html.Strong("COURSERA"),
        ], id="project-identity", className="project-identity"),
        html.Div("CÁC TRANG", className="nav-label"),
        dcc.Tabs(id="page-tab", value="overview", vertical=True, className="tabs-vertical", children=[
            dcc.Tab(label="Tổng quan", value="overview", className="nav-tab nav-overview", selected_className="nav-tab-selected"),
            dcc.Tab(label="Phân tích", value="insight", className="nav-tab nav-insight", selected_className="nav-tab-selected"),
            dcc.Tab(label="Dự đoán người học", value="prediction", className="nav-tab nav-prediction", selected_className="nav-tab-selected"),
            dcc.Tab(label="Dữ liệu", value="data", className="nav-tab nav-data", selected_className="nav-tab-selected"),
        ]),
    ], id="app-sidebar", className="sidebar"),
    html.Main([
        html.Section([
            html.Div([
                html.Div([
                    html.Span(className="filter-icon"),
                    html.Span("BỘ LỌC ĐANG DÙNG", className="filter-kicker"),
                    html.Strong("Lọc khóa học"),
                    html.Div(id="breadcrumb", className="breadcrumb"),
                ], className="filter-heading"),
                html.Div([
                    html.Button("Bỏ lựa chọn trên biểu đồ", id="clear-drill", n_clicks=0, className="button quiet-button small-button"),
                    html.Button("Xóa tất cả bộ lọc", id="reset-filters", className="button secondary small-button", n_clicks=0),
                ], className="filter-toolbar-actions"),
            ], className="filter-toolbar"),
            html.Details([
                html.Summary([
                    html.Span("Chọn khóa học muốn xem"),
                    html.Small("Lọc theo thông tin khóa học hoặc khoảng số liệu"),
                ]),
                html.Div([
                    html.Section([
                        html.Div([
                            html.H4("Thông tin cơ bản"),
                            html.P("Có thể chọn nhiều mục trong mỗi ô."),
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
                            html.H4("Khoảng số liệu"),
                            html.P("Kéo hai đầu thanh để chọn khoảng; log10 giúp thu gọn các mức người học chênh lệch lớn."),
                        ], className="filter-section-heading"),
                        html.Div([
                            field("Điểm đánh giá", dcc.RangeSlider(id="filter-rating", min=0, max=5, step=0.1, value=[0, 5], marks={0: "0", 3: "3", 5: "5"}, tooltip={"placement": "bottom"})),
                            field("Giờ học", dcc.RangeSlider(id="filter-hours", min=0, max=HOURS_MAX, step=0.1, value=[0, HOURS_MAX], marks={0: "0", HOURS_MAX // 2: str(HOURS_MAX // 2), HOURS_MAX: str(HOURS_MAX)}, tooltip={"placement": "bottom"})),
                            field("Người học (log10)", dcc.RangeSlider(id="filter-enrollment", min=0, max=ENROLL_MAX, step=0.05, value=[0, ENROLL_MAX], marks={0: "1", 3: "1 nghìn", 6: "1 triệu"}, tooltip={"placement": "bottom"})),
                        ], className="filter-range-grid"),
                    ], className="filter-section"),
                    html.Div([
                        html.Div([
                            html.Strong("Khi dữ liệu còn thiếu"),
                            html.Small("Chọn có giữ các khóa chưa đủ thông tin hay không."),
                        ], className="filter-options-copy"),
                        dcc.Checklist(id="filter-options", options=[
                            {"label": " Giữ khóa còn thiếu số liệu", "value": "missing"},
                            {"label": " Chỉ dùng giờ học do Coursera công bố", "value": "reported"}], value=["missing"], className="check-options"),
                    ], className="filter-options-row"),
                ], className="filter-grid"),
            ], open=False, className="filter-details"),
        ], id="filter-panel", className="panel filter-panel"),
        html.Div([
            html.Div(id="kpi-row", className="kpi-grid"),
            html.Div([
                card("level", "Not specified là khóa chưa có thông tin trình độ. Bấm vào cột để lọc."),
                card("coverage", "Mỗi phần cho biết thời lượng học được ghi theo cách nào hoặc còn thiếu."),
                card("histogram", "Trục log10 giúp thu gọn khoảng cách: 3 tương ứng 1.000 người học."),
                card("rating", "Đường giữa là trung vị; mỗi hộp chứa 50% số điểm ở giữa."),
                card("tree", "Một khóa có thể có nhiều chủ đề hoặc kỹ năng. Bấm vào ô để lọc.",
                     dcc.RadioItems(id="tree-kind", options=[{"label": " Chủ đề", "value": "subject"}, {"label": " Kỹ năng", "value": "skill"}], value="subject", inline=True)),
                card("map", "Bản đồ ghi trụ sở của tổ chức, không phải nơi ở của người học."),
                card("organizations", "Bấm vào một tổ chức để lọc các khóa do tổ chức đó cung cấp.", wide=True),
            ], className="chart-grid section-gap"),
        ], id="overview-panel"),
        html.Div([
            html.Div([
                card("scatter", "Hai chỉ số cùng tăng không có nghĩa cái này gây ra cái kia."),
                card("heatmap", "Ô trống: chưa đủ khóa có cả hai chỉ số để tính. Spearman gần 1 là cùng tăng, gần -1 là ngược chiều."),
                card("ecdf", "Tại mỗi mốc, đường biểu diễn tỷ lệ khóa có số người học thấp hơn hoặc bằng mốc đó."),
            ], className="chart-grid section-gap"),
            html.Details([
                html.Summary("Tự chọn hai chỉ số để so sánh"),
                html.P("Chọn hai trục; log10 giúp thu gọn các giá trị chênh lệch lớn.", className="muted"),
                html.Div([
                    field("Trục ngang", dcc.Dropdown(id="x-var", options=[{"label": v, "value": k} for k, v in pred.COURSE_LABELS.items()], value="hours_to_complete", clearable=False)),
                    field("Trục dọc", dcc.Dropdown(id="y-var", options=[{"label": v, "value": k} for k, v in pred.COURSE_LABELS.items()], value="enrolled_num", clearable=False)),
                    field("Cách hiển thị", dcc.Checklist(id="log-opts", options=[{"label": " Thu gọn trục ngang (log10)", "value": "x"}, {"label": " Thu gọn trục dọc (log10)", "value": "y"}], value=["x", "y"], inline=True)),
                    field("Giá trị muốn thử trên trục ngang (cách nhau bằng dấu phẩy)", dcc.Input(id="x-new", type="text", value="", debounce=True)),
                ], className="form-grid"),
                dcc.Graph(id="fig-trend", config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), html.P(id="trend-summary"), html.Div(id="trend-warnings"),
                dash_table.DataTable(id="trend-table", style_table={"overflowX": "auto"}),
            ], className="panel exploration"),
        ], id="insight-panel", style={"display": "none"}),
        html.Div([
            html.Div([
                html.Div([
                    html.Span("CÁCH TÍNH", className="model-context-label"),
                    html.Strong("Ước lượng từ các khóa học hiện có"),
                    html.P("Con số này là ước lượng dựa trên các khóa hiện có, không phải dự đoán số người học sẽ tăng bao nhiêu sau này."),
                ]),
                html.Div([
                    html.Span(f"Học từ {MODEL.train_rows:,} khóa"),
                    html.Span(f"Tính khoảng dự đoán từ {MODEL.calibration_rows:,} khóa"),
                    html.Span(f"Kiểm tra bằng {MODEL.test_rows:,} khóa"),
                ], className="model-split"),
            ], className="model-context"),
            html.Div([
                kpi(f"{MODEL.eligible_rows / MODEL.total_rows:.1%}", "Khóa đủ dữ liệu", f"{MODEL.eligible_rows:,}/{MODEL.total_rows:,} khóa có đủ các mục cần thiết"),
                kpi(f"{MODEL.metrics['r2']:.3f}", "R² khi kiểm tra", f"Giải thích khoảng {MODEL.metrics['r2']:.0%} chênh lệch; chỉ dùng lượt đánh giá: {MODEL.baseline_metrics['r2']:.3f}"),
                kpi(f"{MODEL.metrics['rmse_log10']:.3f}", "RMSE (log10)", f"log10 thu gọn các số lớn; càng gần 0 càng tốt, MAE = {MODEL.metrics['mae_log10']:.3f}"),
                kpi(f"{MODEL.metrics['interval_coverage']:.1%}", "Số thật nằm trong khoảng 95%", f"Tính trên {MODEL.test_rows:,} khóa; càng gần 95% càng đúng mức đã đặt"),
            ], className="kpi-grid model-kpis"),
            html.Section([
                html.Div([
                    html.H3("Nhập thông tin khóa học"),
                    html.P("Nhập đủ năm mục để nhận kết quả."),
                    html.Div([
                        field("Lượt đánh giá", dcc.Input(id="model-reviews", type="number", value=1000, min=1, step=1, debounce=True)),
                        field("Điểm đánh giá / 5", dcc.Input(id="model-rating", type="number", value=4.6, min=0, max=5, step=0.1, debounce=True)),
                        field("Giờ học", dcc.Input(id="model-hours", type="number", value=20, min=0.1, debounce=True)),
                        field("Số kỹ năng", dcc.Input(id="model-skills", type="number", value=6, min=0, step=1, debounce=True)),
                        field("Trình độ", dcc.Dropdown(id="model-level", options=list(pred.LEVELS), value="Beginner", clearable=False)),
                    ], className="model-input-grid"),
                ], className="model-inputs"),
                html.Div([
                    html.Div(id="model-estimate", className="estimate"),
                    html.Div(id="model-warning", className="warning"),
                ], className="model-result"),
            ], className="panel model-workbench"),
            html.Div([
                html.Section(dcc.Graph(figure=MODEL_ACTUAL_FIG, config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), className="chart-card"),
                html.Section(dcc.Graph(figure=MODEL_RESIDUAL_FIG, config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), className="chart-card"),
                html.Section(dcc.Graph(figure=coefficient_figure(), config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE), className="chart-card model-coefficient-card"),
            ], className="chart-grid section-gap"),
        ], id="prediction-panel", style={"display": "none"}),
        html.Div([
            html.Section([
                html.H3("Danh sách kết quả"),
                html.P(id="table-caption", className="muted"),
                dash_table.DataTable(id="course-table", columns=[{"name": label, "id": key, "type": "numeric" if key in du.NUMERIC else "text"} for key, label in zip(du.TABLE_COLUMNS, TABLE_LABELS)],
                    hidden_columns=["course_id"], data=[], page_size=10, sort_action="native", cell_selectable=True,
                    style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Be Vietnam Pro, Segoe UI, Tahoma, Arial, sans-serif", "fontSize": 12, "padding": "12px", "textAlign": "left", "minWidth": "90px", "maxWidth": "300px", "whiteSpace": "normal", "border": "none", "borderBottom": "1px solid #eef1f5"},
                    style_header={"fontWeight": "600", "backgroundColor": "#f8f9fb", "color": "#344054"},
                    style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": "#eef4ff", "border": "1px solid #1769e0"}],
                    css=[{"selector": ".show-hide", "rule": "display: none"}]),
                html.Div(id="course-detail", className="course-detail", children="Bấm vào một dòng để xem thông tin khóa học."),
            ], className="panel course-explorer"),
        ], id="data-panel", style={"display": "none"}),
    ], id="dashboard-main", className="main"),
], className="app-shell")


@app.callback(Output("overview-panel", "style"), Output("insight-panel", "style"),
              Output("prediction-panel", "style"), Output("data-panel", "style"),
              Output("filter-panel", "style"),
              Input("page-tab", "value"))
def switch_tab(tab):
    panels = ("overview", "insight", "prediction", "data")
    panel_styles = tuple({"display": "block" if tab == key else "none"} for key in panels)
    filter_style = {"display": "none"} if tab == "prediction" else {"display": "block"}
    return (*panel_styles, filter_style)

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
    kpis = [kpi(f"{n:,}", "Khóa học", f"Trong tổng số {len(FACT):,} khóa", "course.png"),
            kpi(f"{enrolled.median():,.0f}" if len(enrolled) else "Chưa có", "Người học ở mức giữa", f"{len(enrolled):,}/{n:,} khóa có số người học", "people.png"),
            kpi(f"{ratings.mean():.2f} / 5" if len(ratings) else "Chưa có", "Điểm đánh giá trung bình", f"{len(ratings):,}/{n:,} khóa có điểm đánh giá", "star.png"),
             kpi(f"{df.Organization.nunique():,}", "Tổ chức", f"{df.Organization.isna().sum():,} khóa chưa có tên tổ chức", "organize.png")]
    breadcrumb = "Tất cả khóa học" + "".join(f"  ›  {DRILL_LABELS[k]}: {v}" for k, v in (drill or {}).items())
    return (*[figs[key] for key in CHART_KEYS], kpis, du.table_records(df), 0, breadcrumb,
            f"{n:,} khóa phù hợp. Bấm vào một dòng để xem chi tiết.")

def render_detail(course_id, df):
    selected = df[df.course_id.eq(course_id)]
    if selected.empty:
        return "Khóa này không còn khớp với bộ lọc. Hãy chọn lại trong bảng."
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
    link = html.A("Mở khóa học trên Coursera", href=url, target="_blank", rel="noopener noreferrer", className="course-link") if urlparse(url).scheme in ("https", "http") else None
    estimated = row.hours_to_complete_is_estimated
    hours_note = "Giờ học được ước tính từ thời lượng theo tháng, không phải số Coursera công bố." if pd.notna(estimated) and bool(estimated) else ""
    return [html.Div("THÔNG TIN KHÓA HỌC", className="eyebrow"), html.H3(row.title),
            html.Div(facts, className="detail-grid"), html.P(hours_note, className="warning"), *items, link]

@app.callback(Output("course-detail", "children"), Input("chart-scatter", "clickData"), Input("course-table", "active_cell"), Input("filter-state", "data"), Input("drill-state", "data"))
def update_detail(scatter, cell, filters, drill):
    if ctx.triggered_id in ("filter-state", "drill-state", None):
        return "Bấm vào một dòng để xem thông tin khóa học."
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
            raise ValueError("Thông tin nhập nằm ngoài khoảng có thể ước tính.")
    except (ValueError, TypeError, OverflowError) as exc:
        return "Chưa thể ước lượng", str(exc)
    return [html.Small("SỐ NGƯỜI HỌC ƯỚC LƯỢNG"), html.Strong(f"{result['estimate']:,.0f}"), html.Span(f"Khoảng dự đoán 95%: {result['lower']:,.0f} đến {result['upper']:,.0f}")], html.Ul([html.Li(w) for w in warnings]) if warnings else ""

if __name__ == "__main__":
    app.run(debug=False)
