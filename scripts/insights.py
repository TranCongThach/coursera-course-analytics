from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
FACT_PATH = PROCESSED_DIR / "fact_courses_eda_ready.csv"
SKILL_BRIDGE_PATH = PROCESSED_DIR / "bridge_course_skill.csv"
SUBJECT_BRIDGE_PATH = PROCESSED_DIR / "bridge_course_subject.csv"

FACT_COLUMNS = [
    "course_id", "title", "Organization", "URL", "level_clean",
    "enrolled_num", "rating_num", "num_reviews", "satisfaction_rate_num",
    "hours_to_complete", "duration_weeks", "skill_count",
    "organization_hq_country", "popularity_score",
]
LEVEL_ORDER = ["Beginner", "Intermediate", "Advanced", "Not specified"]
PALETTE = {"main": "#1f4e79", "accent": "#c0392b", "muted": "#9aa5b1"}


# --------------------------------------------------------------------------- #
# 1. Nạp dữ liệu (dashboard nên bọc bằng st.cache_data)
# --------------------------------------------------------------------------- #
def load_fact(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or FACT_PATH, usecols=FACT_COLUMNS)


def load_skill_bridge(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or SKILL_BRIDGE_PATH, usecols=["course_id", "skill", "skill_normalized"])


def load_subject_bridge(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or SUBJECT_BRIDGE_PATH, usecols=["course_id", "subject", "subject_normalized"])


# --------------------------------------------------------------------------- #
# 2. Bộ lọc
# --------------------------------------------------------------------------- #
def filter_options(df: pd.DataFrame) -> dict[str, list]:
    """Giá trị để đổ vào widget bộ lọc."""
    levels = [x for x in LEVEL_ORDER if x in set(df["level_clean"].dropna())]
    hq = df["organization_hq_country"].value_counts().index.tolist()
    return {"levels": levels, "hq_countries": hq}


def apply_filters(
    df: pd.DataFrame,
    levels: list[str] | None = None,
    hq_countries: list[str] | None = None,
    organization_query: str | None = None,
) -> pd.DataFrame:
    """Lọc fact. Danh sách rỗng/None = không lọc theo tiêu chí đó."""
    out = df
    if levels:
        out = out[out["level_clean"].isin(levels)]
    if hq_countries:
        out = out[out["organization_hq_country"].isin(hq_countries)]
    if organization_query:
        out = out[out["Organization"].fillna("").str.contains(organization_query, case=False, regex=False)]
    return out


# --------------------------------------------------------------------------- #
# 3. KPI và bảng thống kê
# --------------------------------------------------------------------------- #
def compute_kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    """KPI tổng quan. Median (không dùng mean) vì enrollment/review lệch phải mạnh."""
    n = len(df)

    def med(col):
        s = df[col].dropna()
        return float(s.median()) if len(s) else None

    def cov(col):
        return float(df[col].notna().mean()) if n else 0.0

    return {
        "n_courses": n,
        "n_organizations": int(df["Organization"].nunique()),
        "median_enrolled": med("enrolled_num"),
        "median_rating": med("rating_num"),
        "median_hours": med("hours_to_complete"),
        "coverage_enrolled": cov("enrolled_num"),
        "coverage_rating": cov("rating_num"),
        "coverage_hours": cov("hours_to_complete"),
    }


def level_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Số khóa theo level (Not specified = thiếu dữ liệu, không phải level thật)."""
    counts = df["level_clean"].fillna("Not specified").value_counts()
    out = counts.reindex([x for x in LEVEL_ORDER if x in counts.index]).rename_axis("level").reset_index(name="n_courses")
    out["share_pct"] = out["n_courses"] / max(out["n_courses"].sum(), 1) * 100
    return out


def top_organizations(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top tổ chức theo số URL khóa, kèm median enrollment/rating (bỏ NaN, có cỡ mẫu)."""
    g = df.groupby("Organization", dropna=True)
    out = pd.DataFrame({
        "n_courses": g.size(),
        "median_enrolled": g["enrolled_num"].median(),
        "median_rating": g["rating_num"].median(),
        "n_with_enrolled": g["enrolled_num"].count(),
    })
    return out.sort_values("n_courses", ascending=False).head(n).reset_index()


def rating_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """Rating theo level kèm cỡ mẫu thực tế (số khóa có rating)."""
    d = df.dropna(subset=["rating_num"])
    d = d[d["level_clean"].isin(LEVEL_ORDER)]
    g = d.groupby("level_clean")["rating_num"]
    out = pd.DataFrame({"n_rated": g.size(), "median": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75)})
    return out.reindex([x for x in LEVEL_ORDER if x in out.index]).rename_axis("level").reset_index()


def _top_from_bridge(df: pd.DataFrame, bridge: pd.DataFrame, label: str, norm: str, n: int) -> pd.DataFrame:
    """Đếm số khóa (đã lọc) theo skill/subject; bridge đã unique theo (course, item)."""
    b = bridge[bridge["course_id"].isin(df["course_id"])]
    out = (
        b.groupby(norm)
        .agg(n_courses=("course_id", "nunique"), **{label: (label, lambda s: s.mode().iloc[0])})
        .sort_values("n_courses", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return out[[label, "n_courses"]]


def top_skills(df: pd.DataFrame, skill_bridge: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return _top_from_bridge(df, skill_bridge, "skill", "skill_normalized", n)


def top_subjects(df: pd.DataFrame, subject_bridge: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return _top_from_bridge(df, subject_bridge, "subject", "subject_normalized", n)


def coverage_of(df: pd.DataFrame, bridge: pd.DataFrame) -> tuple[int, int]:
    """(số khóa trong bộ lọc có ít nhất 1 skill/subject, tổng số khóa) — để in coverage."""
    return int(df["course_id"].isin(bridge["course_id"]).sum()), len(df)


# ---------------------------------------------------------------------------
# 4. Biểu đồ (Plotly)
# ---------------------------------------------------------------------------
def empty_figure(message: str = "Không có dữ liệu sau khi lọc") -> go.Figure:
    """Figure trống kèm thông báo, dùng khi bộ lọc không còn dữ liệu hoặc hồi quy lỗi."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(template="plotly_white", xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


def bar_figure(data: pd.DataFrame, label_col: str, value_col: str, title: str,
               xlabel: str = "Số khóa học", horizontal: bool = True) -> go.Figure:
    """Biểu đồ cột chung (dùng cho level, tổ chức, skill, subject)."""
    if data.empty:
        return empty_figure()
    labels = data[label_col].astype(str).tolist()
    values = data[value_col].tolist()
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker_color=PALETTE["main"],
                             text=[f"{v:,.0f}" for v in values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", automargin=True)  # mục lớn nhất nằm trên cùng
        fig.update_xaxes(title=xlabel)
    else:
        fig.add_trace(go.Bar(x=labels, y=values, marker_color=PALETTE["main"],
                             text=[f"{v:,.0f}" for v in values], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title=xlabel)
    fig.update_layout(template="plotly_white", title=title, margin=dict(l=20, r=40, t=60, b=40), showlegend=False)
    return fig


def fig_level_distribution(df: pd.DataFrame) -> go.Figure:
    return bar_figure(level_distribution(df), "level", "n_courses", "Số khóa theo level", horizontal=False)


def fig_top_organizations(df: pd.DataFrame, n: int = 10) -> go.Figure:
    return bar_figure(top_organizations(df, n), "Organization", "n_courses", f"Top {n} tổ chức theo số khóa")


def fig_top_skills(df: pd.DataFrame, skill_bridge: pd.DataFrame, n: int = 15) -> go.Figure:
    return bar_figure(top_skills(df, skill_bridge, n), "skill", "n_courses", f"Top {n} skill (mỗi skill đếm 1 lần/khóa)")


def fig_top_subjects(df: pd.DataFrame, subject_bridge: pd.DataFrame, n: int = 15) -> go.Figure:
    return bar_figure(top_subjects(df, subject_bridge, n), "subject", "n_courses", f"Top {n} subject (multi-label)")


def fig_rating_by_level(df: pd.DataFrame) -> go.Figure:
    """Boxplot rating theo level, ghi n trên tên nhóm."""
    d = df.dropna(subset=["rating_num"])
    levels = [x for x in LEVEL_ORDER if (d["level_clean"] == x).any()]
    if not levels:
        return empty_figure("Không có dữ liệu rating sau khi lọc")
    fig = go.Figure()
    for lv in levels:
        r = d.loc[d["level_clean"] == lv, "rating_num"]
        fig.add_trace(go.Box(y=r.to_numpy(), name=f"{lv} (n={len(r):,})", boxpoints=False, marker_color=PALETTE["main"]))
    fig.update_layout(template="plotly_white", title="Rating theo level", yaxis_title="Rating (0–5)", showlegend=False,
                      margin=dict(l=50, r=20, t=60, b=40))
    return fig


# --------------------------------------------------------------------------- #
# 5. Câu diễn giải tự sinh (luôn kèm coverage, không suy ra nhân quả)
# --------------------------------------------------------------------------- #
def generate_insights(df: pd.DataFrame, skill_bridge: pd.DataFrame | None = None) -> list[str]:
    """Danh sách câu insight mô tả cho bộ lọc hiện tại."""
    if df.empty:
        return ["Không có khóa học nào khớp bộ lọc."]
    k = compute_kpis(df)
    out = [f"Bộ lọc hiện có {k['n_courses']:,} khóa từ {k['n_organizations']:,} tổ chức."]

    orgs = top_organizations(df, 1)
    if len(orgs):
        o = orgs.iloc[0]
        out.append(f"Tổ chức có nhiều khóa nhất: {o['Organization']} ({int(o['n_courses'])} khóa, "
                   f"{o['n_courses'] / k['n_courses'] * 100:.1f}% bộ lọc).")

    if k["median_enrolled"] is not None:
        out.append(f"Enrollment trung vị {k['median_enrolled']:,.0f} (phủ {k['coverage_enrolled'] * 100:.0f}% số khóa); "
                   "phân phối lệch phải mạnh nên không dùng trung bình.")
    if k["median_rating"] is not None:
        out.append(f"Rating trung vị {k['median_rating']:.2f}/5 (phủ {k['coverage_rating'] * 100:.0f}% số khóa).")

    rl = rating_by_level(df)
    rl = rl[rl["n_rated"] >= 30]
    if len(rl) >= 2:
        hi, lo = rl.loc[rl["median"].idxmax()], rl.loc[rl["median"].idxmin()]
        if hi["median"] > lo["median"]:
            out.append(f"Trong các level có ≥30 khóa có rating, median cao nhất ở {hi['level']} ({hi['median']:.2f}), "
                       f"thấp nhất ở {lo['level']} ({lo['median']:.2f}). Chênh lệch mô tả, chưa kiểm định thống kê.")

    if skill_bridge is not None:
        sk = top_skills(df, skill_bridge, 1)
        if len(sk):
            have, total = coverage_of(df, skill_bridge)
            out.append(f"Skill xuất hiện nhiều nhất: {sk.iloc[0]['skill']} ({int(sk.iloc[0]['n_courses'])} khóa); "
                       f"skill phủ {have:,}/{total:,} khóa trong bộ lọc.")
    return out


if __name__ == "__main__":
    fact = load_fact()
    print(compute_kpis(fact))
    print(level_distribution(fact))
    print(top_organizations(fact, 5))
    for line in generate_insights(fact, load_skill_bridge()):
        print("-", line)
