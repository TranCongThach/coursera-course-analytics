from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "Data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
FACT_PATH = PROCESSED_DIR / "fact_courses_eda_ready.csv"
INTERNET_RAW_PATH = RAW_DIR / "internet_usage.csv"

# Giống scripts/join_internet.py: tên HQ trong fact -> tên trong file World Bank.
COUNTRY_NAME_FIX = {
    "South Korea": "Korea, Rep.",
    "Hong Kong": "Hong Kong SAR, China",
    "Czech Republic": "Czechia",
}


# --------------------------------------------------------------------------- #
# 1. Lõi hồi quy tuyến tính
# --------------------------------------------------------------------------- #
def _t_critical(df: int, confidence: float) -> float:
    """Giá trị tới hạn t hai phía. Dùng scipy nếu có, không thì xấp xỉ Cornish-Fisher."""
    q = 0.5 + confidence / 2.0
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(q, df))
    except Exception:
        z = NormalDist().inv_cdf(q)
        return (
            z
            + (z**3 + z) / (4 * df)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        )


@dataclass
class LinearTrend:
    """Kết quả hồi quy y = intercept + slope * x."""

    slope: float
    intercept: float
    r2: float
    n: int
    se_slope: float
    resid_std: float  # sai số chuẩn của phần dư (s)
    x_mean: float
    sxx: float
    confidence: float = 0.95

    @property
    def df(self) -> int:
        return self.n - 2

    def predict(self, x) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=float)

    def interval(self, x, kind: str = "prediction") -> tuple[np.ndarray, np.ndarray]:
        """Khoảng tin cậy quanh đường hồi quy.

        kind="prediction": khoảng dự báo cho một quan sát mới (rộng hơn, dùng để dự báo).
        kind="mean":       khoảng tin cậy cho giá trị trung bình của đường hồi quy.
        """
        x = np.asarray(x, dtype=float)
        yhat = self.predict(x)
        lever = 1.0 / self.n + (x - self.x_mean) ** 2 / self.sxx
        extra = 1.0 if kind == "prediction" else 0.0
        half = _t_critical(self.df, self.confidence) * self.resid_std * np.sqrt(extra + lever)
        return yhat - half, yhat + half

    def slope_ci(self) -> tuple[float, float]:
        half = _t_critical(self.df, self.confidence) * self.se_slope
        return self.slope - half, self.slope + half


def fit_linear_trend(x, y, confidence: float = 0.95) -> LinearTrend:
    """Khớp OLS. Tự bỏ các cặp NaN/inf. Cần >= 5 điểm và x không hằng số."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        raise ValueError(f"Cần ít nhất 5 điểm dữ liệu hợp lệ để hồi quy và tính khoảng tin cậy (hiện có {n}).")
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx == 0:
        raise ValueError("Biến x không đổi (mọi giá trị bằng nhau), không thể hồi quy.")
    slope = float(((x - x_mean) * (y - y_mean)).sum() / sxx)
    intercept = float(y_mean - slope * x_mean)
    resid = y - (intercept + slope * x)
    sse = float((resid**2).sum())
    sst = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    resid_std = float(np.sqrt(sse / (n - 2)))
    return LinearTrend(
        slope=slope,
        intercept=intercept,
        r2=r2,
        n=n,
        se_slope=resid_std / float(np.sqrt(sxx)),
        resid_std=resid_std,
        x_mean=float(x_mean),
        sxx=sxx,
        confidence=confidence,
    )


@dataclass
class TrendChart:
    """Gói trả về cho Dashboard."""

    fig: go.Figure
    trend: LinearTrend
    table: pd.DataFrame  # bảng dự báo / điểm dự đoán
    summary: str  # đoạn diễn giải ngắn, hiển thị dưới biểu đồ
    warnings: list[str] = field(default_factory=list)


BLUE, RED = "#1f4e79", "#c0392b"
RED_FILL = "rgba(192, 57, 43, 0.15)"


def parse_number_list(text: str | None) -> list[float]:
    """"10, 40; 100" -> [10.0, 40.0, 100.0]. Bỏ token không phải số."""
    if not text:
        return []
    out = []
    for tok in text.replace(";", ",").replace(" ", ",").split(","):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _stat_box(fig: go.Figure, text: str) -> None:
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
                       showarrow=False, align="left", bgcolor="white", bordercolor="#cccccc", borderwidth=1)


def _base_layout(fig: go.Figure, title: str, xtitle: str, ytitle: str) -> None:
    fig.update_layout(template="plotly_white", title=title, xaxis_title=xtitle, yaxis_title=ytitle,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(l=60, r=20, t=60, b=40),
                      hovermode="closest")


# --------------------------------------------------------------------------- #
# 2. Chế độ 1: dự báo Internet Usage theo năm
# --------------------------------------------------------------------------- #
def load_internet_history(source=None) -> pd.DataFrame:
    """Đọc internet_usage.csv (dạng rộng, cột 2000..2023) -> dạng dài.

    source: đường dẫn, hoặc file-like (vd kết quả st.file_uploader); None = Data/raw/internet_usage.csv.
    Trả về cột: country, country_code, year (int), internet_usage_pct (float).
    Nên gọi MỘT LẦN rồi cache (ví dụ st.cache_data) và truyền vào các hàm bên dưới.
    """
    if source is None:
        source = INTERNET_RAW_PATH
    if not hasattr(source, "read"):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(
                f"Không thấy {source}. File raw không nằm trong repo (Data/raw/ bị .gitignore); "
                "hãy đặt internet_usage.csv vào Data/raw/ hoặc truyền source=... tới file đó."
            )
    net = pd.read_csv(source, na_values=["..", "N/A", "-", ""], keep_default_na=True)
    year_cols = sorted((c for c in net.columns if str(c).isdigit()), key=int)
    if not {"Country Name", "Country Code"}.issubset(net.columns) or not year_cols:
        raise ValueError("internet_usage.csv cần cột 'Country Name', 'Country Code' và các cột năm.")
    long = net.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="internet_usage_pct",
    )
    long["year"] = long["year"].astype(int)
    long["internet_usage_pct"] = pd.to_numeric(long["internet_usage_pct"], errors="coerce")
    long = long.rename(columns={"Country Name": "country", "Country Code": "country_code"})
    return long.dropna(subset=["internet_usage_pct"]).reset_index(drop=True)


def list_hq_countries(min_courses: int = 1, fact: pd.DataFrame | None = None) -> pd.DataFrame:
    """Các quốc gia trụ sở có trong fact (kèm số khóa) để đổ vào dropdown.

    Cột ``wb_name`` là tên đã đổi sang chuẩn World Bank, dùng làm tham số ``country``.
    """
    if fact is None:
        fact = pd.read_csv(FACT_PATH, usecols=["organization_hq_country"])
    counts = (
        fact["organization_hq_country"].dropna().value_counts().rename_axis("hq_country").reset_index(name="n_courses")
    )
    counts = counts[counts["n_courses"] >= min_courses].copy()
    counts["wb_name"] = counts["hq_country"].replace(COUNTRY_NAME_FIX)
    return counts.reset_index(drop=True)


def build_internet_forecast(
    country: str,
    history: pd.DataFrame | None = None,
    horizon: int = 7,
    start_year: int | None = None,
    confidence: float = 0.95,
    title: str | None = None,
) -> TrendChart:
    """Hồi quy tuyến tính Internet Usage ~ năm và dự báo ``horizon`` năm tiếp theo.

    country     : tên theo World Bank (vd "Viet Nam", "United States", "Korea, Rep.").
    history     : kết quả của load_internet_history(); None thì tự đọc file raw.
    start_year  : chỉ khớp từ năm này trở đi (vd 2010 nếu muốn bỏ giai đoạn bùng nổ đầu).
    """
    if history is None:
        history = load_internet_history()
    sub = history[history["country"] == country].sort_values("year")
    if start_year is not None:
        sub = sub[sub["year"] >= start_year]
    if sub.empty:
        raise ValueError(f"Không có dữ liệu Internet Usage cho '{country}'.")

    trend = fit_linear_trend(sub["year"], sub["internet_usage_pct"], confidence)
    last_year = int(sub["year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)
    lo, hi = trend.interval(future_years, "prediction")
    yhat = trend.predict(future_years)

    warnings: list[str] = []
    if (yhat > 100).any():
        warnings.append("Đường thẳng vượt 100% ở một số năm dự báo; giá trị hiển thị đã bị chặn ở 100%. "
                        "Mô hình tuyến tính không mô tả được hiện tượng bão hòa.")
    if trend.n < 10:
        warnings.append(f"Chỉ có {trend.n} năm dữ liệu, dự báo kém tin cậy.")

    table = pd.DataFrame(
        {
            "year": future_years,
            "forecast_pct": np.clip(yhat, 0, 100),
            "lower": np.clip(lo, 0, 100),
            "upper": np.clip(hi, 0, 100),
        }
    )

    pct = int(confidence * 100)
    fit_years = np.array([sub["year"].min(), last_year])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["year"].to_numpy(), y=sub["internet_usage_pct"].to_numpy(), mode="markers", name="Quan sát (World Bank/ITU)",
                             marker=dict(color=BLUE, size=8),
                             hovertemplate="%{x}: %{y:.1f}%<extra>Quan sát</extra>"))
    fig.add_trace(go.Scatter(x=fit_years, y=np.clip(trend.predict(fit_years), 0, 100), mode="lines",
                             name="Đường hồi quy", line=dict(color=RED, width=2), hoverinfo="skip"))
    # Khoảng dự báo: trace 'upper' phải đứng ngay trước trace 'lower' để fill="tonexty" tô đúng vùng.
    fig.add_trace(go.Scatter(x=future_years, y=table["upper"].to_numpy(), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_years, y=table["lower"].to_numpy(), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=RED_FILL, name=f"Khoảng dự báo {pct}%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.r_[last_year, future_years],
                             y=np.clip(np.r_[trend.predict(last_year), yhat], 0, 100), mode="lines+markers",
                             name=f"Dự báo {horizon} năm", line=dict(color=RED, width=2, dash="dash"),
                             marker=dict(size=5), hovertemplate="%{x}: %{y:.1f}%<extra>Dự báo</extra>"))
    fig.add_vline(x=last_year, line_dash="dot", line_color="grey")
    _base_layout(fig, title or f"Xu hướng và dự báo Internet Usage — {country}", "Năm", "Người dùng Internet (% dân số)")
    fig.update_yaxes(range=[0, 105])
    _stat_box(fig, f"Độ dốc {trend.slope:+.2f} điểm %/năm<br>R² = {trend.r2:.3f} · n = {trend.n} năm")

    last = table.iloc[-1]
    summary = (
        f"{country}: Internet Usage thay đổi trung bình {trend.slope:+.2f} điểm %/năm "
        f"(R² = {trend.r2:.2f}, {trend.n} năm). Dự báo {int(last.year)}: {last.forecast_pct:.1f}% "
        f"(khoảng {int(confidence * 100)}%: {last.lower:.1f}–{last.upper:.1f}%). "
        "Đây là ngoại suy tuyến tính, không phải dự báo hành vi người học Coursera."
    )
    return TrendChart(fig, trend, table, summary, warnings)


# --------------------------------------------------------------------------- #
# 3. Chế độ 2: đường xu hướng giữa hai biến của khóa học
# --------------------------------------------------------------------------- #
COURSE_LABELS = {
    "hours_to_complete": "Số giờ hoàn thành",
    "duration_weeks": "Số tuần",
    "enrolled_num": "Số học viên (enrolled)",
    "num_reviews": "Số review",
    "rating_num": "Rating (0–5)",
    "skill_count": "Số skill",
    "satisfaction_rate_num": "Satisfaction (%)",
}


# Biến số dùng được làm x/y. Cố ý KHÔNG có enrolled_percentile, rating_normalized,
# popularity_score vì chúng được tính trực tiếp từ enrollment/rating (leakage, xem README).
TREND_VARIABLES = list(COURSE_LABELS)
_TAUTOLOGICAL_PAIRS = [{"enrolled_num", "num_reviews"}]


def load_fact(columns: list[str] | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """Đọc fact canonical (chỉ các cột cần để nhẹ). Nên cache ở phía dashboard."""
    return pd.read_csv(path or FACT_PATH, usecols=columns)


def build_course_trend(
    df: pd.DataFrame,
    x_col: str = "hours_to_complete",
    y_col: str = "enrolled_num",
    log_x: bool = True,
    log_y: bool = True,
    x_new: list[float] | None = None,
    confidence: float = 0.95,
    max_points: int = 4000,
    title: str | None = None,
) -> TrendChart:
    """Đường hồi quy tuyến tính y ~ x trên dữ liệu khóa học (đã lọc theo bộ lọc dashboard).

    log_x / log_y : khớp trên log10 (nên bật cho enrolled_num, num_reviews, hours vì lệch phải mạnh).
    x_new         : các giá trị x (thang gốc) muốn dự đoán y, vd [10, 40, 100] giờ.
    Lưu ý: đây là quan hệ cắt ngang, không phải dự báo theo thời gian, và không hàm ý nhân quả.
    """
    for c in (x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"Thiếu cột '{c}' trong DataFrame.")
    d = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if log_x:
        d = d[d[x_col] > 0]
    if log_y:
        d = d[d[y_col] > 0]
    n_used = len(d)
    if x_col == y_col:
        raise ValueError("x và y phải là hai biến khác nhau.")
    if n_used < 5:
        raise ValueError(f"Chỉ còn {n_used} khóa có đủ '{x_col}' và '{y_col}' sau khi lọc.")

    fx = np.log10(d[x_col]) if log_x else d[x_col]
    fy = np.log10(d[y_col]) if log_y else d[y_col]
    trend = fit_linear_trend(fx, fy, confidence)

    grid_f = np.linspace(fx.min(), fx.max(), 200)
    grid_x = 10**grid_f if log_x else grid_f
    lo, hi = trend.interval(grid_f, "prediction")
    line = trend.predict(grid_f)
    back = (lambda v: 10**v) if log_y else (lambda v: v)

    warnings: list[str] = []
    if {x_col, y_col} in _TAUTOLOGICAL_PAIRS:
        warnings.append("Số review gần như tỉ lệ thuận với enrollment (Spearman ≈ 0.93); "
                        "đường xu hướng này gần như hiển nhiên, không phải insight mới.")
    if trend.r2 < 0.1:
        warnings.append(f"R² = {trend.r2:.3f}: x giải thích rất ít biến thiên của y; đường xu hướng chỉ mang tính tham khảo.")
    if n_used < len(df):
        warnings.append(f"Dùng {n_used:,}/{len(df):,} khóa (bỏ các khóa thiếu hoặc không hợp lệ cho log).")

    table = pd.DataFrame(columns=[x_col, f"{y_col}_pred", "lower", "upper"])
    if x_new:
        xn = np.asarray(x_new, dtype=float)
        fxn = np.log10(xn) if log_x else xn
        l2, h2 = trend.interval(fxn, "prediction")
        table = pd.DataFrame({x_col: xn, f"{y_col}_pred": back(trend.predict(fxn)),
                              "lower": back(l2), "upper": back(h2)})

    plot = d.sample(max_points, random_state=0) if n_used > max_points else d
    xl = COURSE_LABELS.get(x_col, x_col)
    yl = COURSE_LABELS.get(y_col, y_col)
    pct = int(confidence * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot[x_col].to_numpy(), y=plot[y_col].to_numpy(), mode="markers", name="Khóa học",
                             marker=dict(color=BLUE, size=5, opacity=0.25),
                             hovertemplate=f"{xl}: %{{x:,.4g}}<br>{yl}: %{{y:,.4g}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid_x, y=back(hi), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grid_x, y=back(lo), mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=RED_FILL, name=f"Khoảng dự báo {pct}%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grid_x, y=back(line), mode="lines", name="Đường hồi quy",
                             line=dict(color=RED, width=2), hoverinfo="skip"))
    if len(table):
        fig.add_trace(go.Scatter(x=table[x_col].to_numpy(), y=table[f"{y_col}_pred"].to_numpy(), mode="markers", name="Điểm dự đoán",
                                 marker=dict(symbol="diamond", size=11, color="#f39c12", line=dict(color="black", width=1)),
                                 hovertemplate=f"{xl}: %{{x:,.4g}}<br>Dự đoán: %{{y:,.4g}}<extra></extra>"))
    _base_layout(fig, title or f"Xu hướng tuyến tính: {yl} theo {xl}", xl + (" (log)" if log_x else ""),
                 yl + (" (log)" if log_y else ""))
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    _stat_box(fig, f"R² = {trend.r2:.3f} · n = {trend.n:,}")

    if log_x and log_y:
        meaning = f"khi {xl} tăng 10%, {yl} thay đổi khoảng {((1.1 ** trend.slope) - 1) * 100:+.1f}% (độ co giãn {trend.slope:.2f})"
    else:
        meaning = f"mỗi +1 đơn vị {xl} ứng với {trend.slope:+.4g} đơn vị {yl}" + (" (thang log10)" if log_y or log_x else "")
    summary = (f"{meaning}. R² = {trend.r2:.2f}, n = {trend.n:,}. "
               "Quan hệ cắt ngang trên snapshot, không phải dự báo theo thời gian và không hàm ý nhân quả.")
    return TrendChart(fig, trend, table, summary, warnings)


# --------------------------------------------------------------------------- #
# 4. Chạy thử: python scripts/prediction.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    out = PROJECT_DIR / "outputs" / "trend"
    out.mkdir(parents=True, exist_ok=True)

    fact = load_fact(["hours_to_complete", "enrolled_num"])
    c = build_course_trend(fact, "hours_to_complete", "enrolled_num", x_new=[10, 40, 100])
    c.fig.write_html(out / "trend_course_hours_enrollment.html")
    print(c.summary)
    print(c.table.round(1).to_string(index=False))
    for w in c.warnings:
        print("Cảnh báo:", w)

    try:
        hist = load_internet_history()
    except FileNotFoundError as exc:
        print(exc)
    else:
        f = build_internet_forecast("Viet Nam", hist, horizon=7)
        f.fig.write_html(out / "forecast_internet_viet_nam.html")
        print(f.summary)
        print(f.table.round(1).to_string(index=False))
