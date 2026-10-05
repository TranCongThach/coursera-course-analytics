from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from Hoa import insights
ROOT = Path(__file__).resolve().parents[1]
LEVELS = insights.LEVEL_ORDER
# Categorical: mỗi nhóm 1 màu riêng (phân biệt nhanh, thân thiện mù màu).
# Sequential: dùng cho treemap / choropleth (giá trị lớn = màu đậm).
COLORS = ["#2f6fed", "#12a594", "#f59e0b", "#94a3b8", "#8e4ec6", "#0ea5e9", "#e5484d"]
LEVEL_COLORS = {
    "Beginner": "#2f6fed",       
    "Intermediate": "#12a594",  
    "Advanced": "#f59e0b",       
    "Not specified": "#94a3b8",  
}
COVERAGE_COLORS = {
    "reported_hours_and_weeks": "#2f6fed",
    "reported_hours_only": "#12a594",
    "estimated_from_months": "#f59e0b",
    "missing": "#94a3b8",
}
BLUE_SCALE = [[0, "#eef4ff"], [0.35, "#bcd0ff"], [0.65, "#5b8def"], [1, "#0b3d9c"]]
DIVERGING_SCALE = [[0, "#d92d20"], [0.5, "#f8fafc"], [1, "#175cd3"]]
CHART_HEIGHT = 400
NUMERIC = ["enrolled_num", "num_reviews", "rating_num", "hours_to_complete", "skill_count"]
LABELS = ["Người học", "Lượt đánh giá", "Điểm đánh giá", "Giờ học", "Số kỹ năng"]
COUNTRY_ISO = dict(zip(
    ["United States", "South Africa", "Denmark", "Spain", "Mexico", "South Korea",
     "India", "Netherlands", "United Kingdom", "Canada", "Australia", "Israel",
     "Japan", "Belgium", "France", "Hong Kong", "Sweden", "China", "Switzerland",
     "Taiwan", "Italy", "Norway", "Germany", "Ireland", "Morocco", "Czech Republic",
     "Argentina", "Colombia", "Saudi Arabia", "Peru", "United Arab Emirates",
     "Singapore", "Brazil", "Chile"],
    ["USA", "ZAF", "DNK", "ESP", "MEX", "KOR", "IND", "NLD", "GBR", "CAN",
     "AUS", "ISR", "JPN", "BEL", "FRA", "HKG", "SWE", "CHN", "CHE", "TWN",
     "ITA", "NOR", "DEU", "IRL", "MAR", "CZE", "ARG", "COL", "SAU", "PER",
     "ARE", "SGP", "BRA", "CHL"], strict=True))

def load_data():
    directory = ROOT / "Data" / "processed"
    fact = pd.read_csv(directory / "fact_courses_eda_ready.csv")
    if fact.course_id.isna().any() or fact.course_id.duplicated().any():
        raise ValueError("Fact phải có course_id đầy đủ, duy nhất.")
    bridges = {}
    for kind in ("subject", "skill"):
        bridge = pd.read_csv(directory / f"bridge_course_{kind}.csv")
        if not bridge.course_id.isin(fact.course_id).all():
            raise ValueError(f"Bridge {kind} chứa course_id không thuộc fact.")
        bridges[kind] = bridge.drop_duplicates(["course_id", f"{kind}_normalized"])
    return fact, bridges

def bridge_options(bridge, kind):
    labels = bridge.groupby(f"{kind}_normalized")[kind].first().sort_values()
    return [{"label": label, "value": key} for key, label in labels.items()]

def filter_courses(fact, filters, bridges, drill=None):
    out = fact
    for key, column in (("organizations", "Organization"), ("levels", "level_clean")):
        if filters.get(key):
            out = out[out[column].isin(filters[key])]
    for kind in ("subject", "skill"):
        selected = filters.get(kind + "s")
        if selected:
            bridge = bridges[kind]
            ids = bridge.loc[bridge[f"{kind}_normalized"].isin(selected), "course_id"]
            out = out[out.course_id.isin(ids)]
    for key, column in (("rating", "rating_num"), ("hours", "hours_to_complete"),
                        ("enrollment", "enrolled_num")):
        bounds = filters.get(key)
        if bounds is not None:
            values = out[column]
            if key == "enrollment":
                values = np.log10(values.where(values.gt(0)))
            mask = values.between(*bounds)
            if filters.get("include_missing", True):
                mask |= out[column].isna()
            out = out[mask]
    if filters.get("reported_only"):
        out = out[out.hours_to_complete_is_estimated.eq(False)]
    for key, value in (drill or {}).items():
        if key in bridges:
            bridge = bridges[key]
            ids = bridge.loc[bridge[f"{key}_normalized"].eq(value), "course_id"]
            out = out[out.course_id.isin(ids)]
        elif key in ("level_clean", "organization_hq_country", "Organization"):
            out = out[out[key].eq(value)]
    return out.copy()

def chart_title(title, subtitle):
    """Tiêu đề Plotly hai tầng: nội dung chính và phạm vi dữ liệu."""
    return (
        f"<b>{title}</b><br>"
        f"<span style='font-size:11px;color:#667085'>{subtitle}</span>"
    )


def style_figure(fig):
    # Giữ khoảng trái/phải riêng của heatmap, map và treemap; chuẩn hóa
    # khoảng trên để tiêu đề hai tầng không chạm vùng vẽ.
    m = fig.layout.margin
    is_default_margin = (m.l == 80 and m.r == 80 and m.t == 100 and m.b == 80)
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                      # Tahoma/Arial render các dấu kép tiếng Việt (ấ, ố, ể...)
                      font=dict(family="Tahoma, Arial, Segoe UI, sans-serif", color="#344054", size=12.5),
                      colorway=COLORS,
                      height=CHART_HEIGHT, autosize=True,
                      title=dict(x=0.01, xanchor="left", y=0.95, yanchor="top",
                                 font=dict(size=18, color="#172033")),
                      legend=dict(orientation="h", y=-0.28, x=0, yanchor="top",
                                  font=dict(size=11)),
                      hoverlabel=dict(bgcolor="white", align="left"))
    if is_default_margin:
        fig.update_layout(margin=dict(l=58, r=28, t=76, b=88))
    else:
        fig.update_layout(margin=dict(
            l=m.l, r=m.r, t=max(m.t or 0, 76), b=m.b,
        ))
    fig.update_xaxes(gridcolor="#f0f2f5", automargin=True)
    fig.update_yaxes(gridcolor="#f0f2f5", automargin=True)
    return fig

def empty_figure(message="Không có khóa học phù hợp với bộ lọc."):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig)

def pairwise_spearman(df):
    rho = np.full((len(NUMERIC), len(NUMERIC)), np.nan)
    counts = np.zeros_like(rho, dtype=int)
    for i, left in enumerate(NUMERIC):
        for j, right in enumerate(NUMERIC):
            pair = df[[left, right]].dropna()
            counts[i, j] = len(pair)
            a, b = pair.iloc[:, 0], pair.iloc[:, 1]
            if len(pair) >= 3 and a.nunique() > 1 and b.nunique() > 1:
                rho[i, j] = a.rank().corr(b.rank())
    return rho, counts

def build_figures(df, bridges, tree_kind="subject"):
    keys = ["level", "coverage", "histogram", "rating", "tree", "map", "scatter", "heatmap", "ecdf", "organizations"]
    if df.empty:
        return {key: empty_figure() for key in keys}
    figures = {}
    # 01 / level: cột đứng, mỗi trình độ 1 màu riêng + nhãn số lượng + % khi hover.
    level_counts = df["level_clean"].fillna("Not specified").value_counts()
    level_order = [lv for lv in LEVELS if lv in level_counts.index] or level_counts.index.tolist()
    level_values = [int(level_counts[lv]) for lv in level_order]
    level_colors = [LEVEL_COLORS.get(lv, COLORS[i % len(COLORS)]) for i, lv in enumerate(level_order)]
    total_lv = max(sum(level_values), 1)
    figures["level"] = go.Figure(go.Bar(
        x=level_order, y=level_values, marker_color=level_colors,
        marker_line=dict(color="white", width=1),
        text=[f"{v:,}" for v in level_values], textposition="outside", cliponaxis=False,
        customdata=[[v / total_lv] for v in level_values],
        hovertemplate="%{x}<br>%{y:,} khóa học · %{customdata[0]:.1%}<extra></extra>"))
    figures["level"].update_layout(
        title=chart_title("Phân bố khóa học theo trình độ", f"Đang tính trên {len(df):,} khóa"),
        yaxis_title="Số khóa học", showlegend=False,
    )
    status_order = ["reported_hours_and_weeks", "reported_hours_only", "estimated_from_months", "missing"]
    status = df.schedule_parse_status.fillna("missing").value_counts().reindex(status_order, fill_value=0)
    status = status[status.gt(0)]
    status_labels = {"reported_hours_and_weeks": "Có giờ và số tuần", "reported_hours_only": "Chỉ có số giờ",
                     "estimated_from_months": "Ước tính giờ học từ số tháng", "missing": "Chưa có lịch học"}
    # 02 / coverage: pie 4 lát màu phân loại (không dùng 4 sắc xanh gần nhau).
    figures["coverage"] = go.Figure(go.Pie(labels=[status_labels.get(s, s) for s in status.index],
        values=status.values, hole=0.55,
        marker_colors=[COVERAGE_COLORS.get(key, "#94a3b8") for key in status.index], sort=False,
        marker_line=dict(color="white", width=2),
        textinfo="percent", textfont_size=12,
        hovertemplate="%{label}<br>%{value:,} khóa học · %{percent}<extra></extra>"))
    figures["coverage"].update_layout(
        title=chart_title("Tình trạng thông tin lịch học", f"Đang kiểm tra {len(df):,} khóa"),
    )
    enrolled = df.loc[df.enrolled_num.gt(0), "enrolled_num"]
    figures["histogram"] = empty_figure("Chưa có khóa học nào ghi số người học.")
    figures["ecdf"] = empty_figure("Chưa có khóa học nào ghi số người học.")
    if len(enrolled):
        # 03 / histogram: 1 màu duy nhất là đúng (phân phối 1 biến) + viền trắng từng cột.
        figures["histogram"] = go.Figure(go.Histogram(x=np.log10(enrolled), nbinsx=30,
            marker_color="#2f6fed", marker_line=dict(color="white", width=0.8), opacity=0.88,
            hovertemplate="log10 người học: %{x:.2f}<br>Số khóa học: %{y}<extra></extra>"))
        figures["histogram"].update_layout(
            title=chart_title("Phân bố số người học", f"{len(enrolled):,} khóa có số người học · trục ngang dùng log10"),
            xaxis_title="Người học (log10)", yaxis_title="Số khóa học",
        )
        values, counts = np.unique(enrolled, return_counts=True)
        # 10 / ecdf: 1 đường duy nhất + vùng tô nhạt để đọc tích lũy.
        figures["ecdf"] = go.Figure(go.Scatter(x=values, y=np.cumsum(counts) / len(enrolled),
            mode="lines", line=dict(shape="hv", width=3, color="#0b3d9c"), fill="tozeroy",
            fillcolor="rgba(47,111,237,0.12)",
            hovertemplate="Người học ≤ %{x:,.0f}<br>%{y:.1%} số khóa học<extra></extra>"))
        figures["ecdf"].update_layout(title=chart_title(
                "Phân phối tích lũy số người học",
                f"{len(enrolled):,} khóa có số người học · trục ngang dùng log10",
            ),
            xaxis=dict(type="log", title="Người học (log10)"),
            yaxis=dict(title="Tỷ lệ khóa học", tickformat=".0%", range=[0, 1.02]))
    # 04 / rating: mỗi box 1 màu theo level (trước đây cả 4 box cùng 1 màu xanh).
    figures["rating"] = insights.fig_rating_by_level(df)
    for trace in figures["rating"].data:
        lv = str(trace.name).split(" (n=")[0]
        trace.marker.color = LEVEL_COLORS.get(lv, "#2f6fed")
        trace.line.color = LEVEL_COLORS.get(lv, "#2f6fed")
    figures["rating"].update_layout(title=chart_title(
        "Điểm đánh giá ở từng trình độ",
        f"{df.rating_num.count():,} khóa có điểm đánh giá",
    ), yaxis_title="Điểm đánh giá (0–5)")
    bridge = bridges[tree_kind]
    selected = bridge[bridge.course_id.isin(df.course_id)]
    norm = tree_kind + "_normalized"
    tree = selected.groupby(norm).agg(n=("course_id", "nunique"), label=(tree_kind, "first"))
    tree = tree.sort_values(["n", "label"], ascending=[False, True]).head(20)
    tree_label = "chủ đề" if tree_kind == "subject" else "kỹ năng"
    figures["tree"] = empty_figure(f"Chưa có khóa học nào ghi {tree_label}.")
    if len(tree):
        # 05 / treemap: sequential (giá trị lớn = đậm) + chữ trắng trên ô đậm.
        figures["tree"] = go.Figure(go.Treemap(ids=tree.index.tolist(), labels=tree.label,
            parents=[""] * len(tree), values=tree.n, root_color="#f1f5f9", tiling=dict(pad=3, packing="squarify"),
            marker=dict(colors=tree.n, colorscale=BLUE_SCALE, showscale=True,
                        colorbar=dict(title="Số khóa học", thickness=12, len=0.6, x=1.0, xpad=6),
                        line=dict(color="white", width=2)),
            textinfo="label+value", textfont=dict(size=12),
            hovertemplate="%{label}<br>%{value:,} khóa học<extra></extra>"))
        figures["tree"].update_layout(
            title=chart_title(
                f"Top {tree_label} phổ biến nhất",
                f"Hiển thị {len(tree):,} nhóm có nhiều khóa nhất · {selected.course_id.nunique():,}/{len(df):,} khóa có thông tin",
            ),
            margin=dict(l=20, r=72, t=76, b=20),
        )
    countries = df.groupby("organization_hq_country").agg(n=("course_id", "nunique"), organizations=("Organization", "nunique"))
    countries["iso"] = countries.index.map(COUNTRY_ISO)
    countries = countries.dropna(subset=["iso"])
    figures["map"] = empty_figure("Chưa đủ thông tin trụ sở để vẽ bản đồ.")
    if len(countries):
        # 06 / map: sequential xanh + viền trắng, nước nhiều khóa đậm hơn rõ rệt.
        figures["map"] = go.Figure(go.Choropleth(locations=countries.iso, z=countries.n,
            locationmode="ISO-3", colorscale=BLUE_SCALE, marker_line_color="white", marker_line_width=0.8,
            customdata=np.column_stack([countries.index, countries.organizations]),
            colorbar=dict(title="Số khóa học", thickness=12, len=0.65, x=1.0, xpad=6),
            hovertemplate="%{customdata[0]}<br>%{z:,} khóa học<br>%{customdata[1]} tổ chức<extra></extra>"))
        figures["map"].update_layout(title=chart_title(
                "Bản đồ trụ sở tổ chức",
                f"{countries.n.sum():,}/{len(df):,} khóa xác định được quốc gia của tổ chức",
            ),
            margin=dict(l=20, r=72, t=76, b=20),
            geo=dict(projection_type="natural earth", showframe=False, showcoastlines=False,
                     showland=True, landcolor="#eef2f6", bgcolor="white"))
    scatter = df[df.num_reviews.gt(0) & df.enrolled_num.gt(0)]
    figures["scatter"] = empty_figure("Chưa có khóa học nào ghi cả lượt đánh giá và số người học.")
    if len(scatter):
        # 08 / scatter: mỗi level 1 màu phân loại tương phản (xanh / ngọc / cam / xám).
        fig = go.Figure()
        for level in LEVELS:
            rows = scatter[scatter.level_clean.eq(level)]
            if len(rows) == 0:
                continue
            fig.add_trace(go.Scattergl(x=rows.num_reviews, y=rows.enrolled_num, mode="markers", name=level,
                customdata=np.column_stack([rows.course_id, rows.title]),
                marker=dict(color=LEVEL_COLORS.get(level, "#2f6fed"), size=6, opacity=0.6,
                            line=dict(color="white", width=0.5)),
                hovertemplate="%{customdata[1]}<br>Lượt đánh giá: %{x:,.0f}<br>Người học: %{y:,.0f}<extra>%{fullData.name}</extra>"))
        fig.update_layout(title=chart_title(
                               "Quan hệ giữa lượt đánh giá và số người học",
                               f"{len(scatter):,} khóa có đủ cả hai chỉ số · hai trục dùng log10",
                           ),
                           xaxis=dict(type="log", title="Lượt đánh giá (log10)",
                                      exponentformat="power", showexponent="all"),
                           yaxis=dict(type="log", title="Người học (log10)",
                                      exponentformat="power", showexponent="all"))
        figures["scatter"] = fig
    rho, counts = pairwise_spearman(df)
    # 09 / heatmap: diverging đỏ–trắng–xanh (âm = đỏ, dương = xanh), số in trực tiếp.
    figures["heatmap"] = go.Figure(go.Heatmap(z=rho, x=LABELS, y=LABELS, zmin=-1, zmax=1,
        colorscale=DIVERGING_SCALE,
        customdata=counts, texttemplate="%{z:.2f}", textfont=dict(size=11),
        hoverongaps=False,
        xgap=1, ygap=1,
        colorbar=dict(title="Mức liên hệ", thickness=12, len=0.75, x=1.0, xpad=6),
        hovertemplate="%{x} × %{y}<br>Spearman: %{z:.3f}<br>Số khóa có đủ hai chỉ số: %{customdata:,}<extra></extra>"))
    figures["heatmap"].update_layout(title=chart_title(
            "Tương quan giữa các chỉ số khóa học",
            "Số gần 1 là cùng tăng, gần -1 là ngược chiều · mỗi ô chỉ tính khóa có đủ hai chỉ số",
        ),
        margin=dict(l=105, r=72, t=76, b=78),
        xaxis=dict(tickangle=-20, automargin=True),
        yaxis=dict(automargin=True))
    # 07 / organizations: cột ngang gradient theo số khóa (nhiều = đậm) + nhãn số ở đầu cột.
    figures["organizations"] = insights.fig_top_organizations(df)
    org_trace = figures["organizations"].data[0]
    org_vals = list(org_trace.x)
    figures["organizations"].update_traces(
        marker=dict(color=org_vals, colorscale=BLUE_SCALE, showscale=False,
                    line=dict(color="white", width=1)),
        text=[f"{v:,}" for v in org_vals], textposition="outside", cliponaxis=False)
    figures["organizations"].update_layout(title=chart_title(
        "Top 10 tổ chức có nhiều khóa học nhất",
        f"10 tổ chức có nhiều khóa nhất trong tổng số {df.Organization.nunique():,} tổ chức",
    ))
    return {key: style_figure(fig) for key, fig in figures.items()}

def insight_text(df, bridges):
    if df.empty:
        return ["Không có khóa phù hợp. Thử nới bộ lọc hoặc xóa lựa chọn."]
    messages = []
    counts = df.level_clean.value_counts()
    leaders = counts[counts.eq(counts.max())]
    messages.append(f"Trình độ có nhiều khóa nhất: {', '.join(leaders.index)} — "
                    f"{int(leaders.iloc[0]):,} khóa ({leaders.iloc[0] / len(df):.1%}).")
    pair = df[["num_reviews", "enrolled_num"]].dropna()
    if len(pair) >= 3 and pair.nunique().min() > 1:
        rho = pair.num_reviews.rank().corr(pair.enrolled_num.rank())
        messages.append(f"Số lượt đánh giá và số người học liên hệ ở mức {rho:.2f} trên {len(pair):,} khóa. "
                        "Hai số này được ghi cùng thời điểm nên không kết luận cái này gây ra cái kia.")
    enrollment = df.enrolled_num.dropna()
    if len(enrollment):
        messages.append(f"Số người học ở giữa là {enrollment.median():,.0f}; trung bình là {enrollment.mean():,.0f}. "
                        f"Có dữ liệu ở {len(enrollment):,}/{len(df):,} khóa.")
    skill = insights.top_skills(df, bridges["skill"], n=1)
    if not skill.empty:
        row = skill.iloc[0]
        messages.append(f"Kỹ năng phổ biến nhất: {row['skill']} ({int(row.n_courses):,} khóa). "
                        f"Có thông tin kỹ năng ở {df.course_id.isin(bridges['skill'].course_id).sum():,}/{len(df):,} khóa.")
    return messages

TABLE_COLUMNS = ["course_id", "title", "Organization", "level_clean", "rating_num", "enrolled_num", "num_reviews", "hours_to_complete"]

def table_records(df):
    result = df[TABLE_COLUMNS].copy().round(2)
    result["id"] = result.course_id
    return result.astype(object).where(result.notna(), None).to_dict("records")
