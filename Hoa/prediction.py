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
COUNTRY_NAME_FIX = {
    "South Korea": "Korea, Rep.",
    "Hong Kong": "Hong Kong SAR, China",
    "Czech Republic": "Czechia",
}

# 1. Lõi hồi quy tuyến tính
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
        raise ValueError(f"Cần ít nhất 5 khóa có đủ dữ liệu (hiện có {n}).")
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx == 0:
        raise ValueError("Các giá trị trục ngang giống hệt nhau nên không vẽ được đường xu hướng.")
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


BLUE, TREND = "#2f6fed", "#e67e22"
TREND_FILL = "rgba(230, 126, 34, 0.16)"


def parse_number_list(text: str | None) -> list[float]:
    """"10, 40; 100" -> [10.0, 40.0, 100.0]; báo lỗi khi nhập sai."""
    if not text:
        return []
    out = []
    for tok in text.replace(";", ",").replace(" ", ",").split(","):
        if not tok:
            continue
        try:
            value = float(tok)
        except ValueError as exc:
            raise ValueError(f"Giá trị '{tok}' không phải là số.") from exc
        if not np.isfinite(value):
            raise ValueError("Giá trị phải là số hợp lệ.")
        out.append(value)
    return out


def _stat_box(fig: go.Figure, text: str) -> None:
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
                       showarrow=False, align="left", bgcolor="white", bordercolor="#cccccc", borderwidth=1)


def _base_layout(fig: go.Figure, title: str, xtitle: str, ytitle: str) -> None:
    fig.update_layout(template="plotly_white", title=title, xaxis_title=xtitle, yaxis_title=ytitle,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(l=60, r=20, t=60, b=40),
                      hovermode="closest")


# 2. Chế độ 1: dự báo Internet Usage theo năm
def load_internet_history(source=None) -> pd.DataFrame:
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
    if history is None:
        history = load_internet_history()
    sub = history[history["country"] == country].sort_values("year")
    if start_year is not None:
        sub = sub[sub["year"] >= start_year]
    if sub.empty:
        raise ValueError(f"Không có dữ liệu Internet Usage cho '{country}'.")
    if not isinstance(horizon, int) or horizon < 1 or horizon > 10:
        raise ValueError("Số năm dự báo phải là số nguyên từ 1 đến 10.")

    trend = fit_linear_trend(sub["year"], sub["internet_usage_pct"], confidence)
    last_year = int(sub["year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)
    lo, hi = trend.interval(future_years, "prediction")
    yhat = trend.predict(future_years)

    warnings: list[str] = []
    if ((yhat > 100) | (yhat < 0) | (lo < 0) | (hi > 100)).any():
        warnings.append("Dự báo hoặc khoảng 95% vượt miền [0, 100]%; số gốc vẫn hiển thị để thấy "
                        "giới hạn của hồi quy tuyến tính. Không diễn giải các năm này như tỷ lệ khả thi.")
    if trend.n < 10:
        warnings.append(f"Chỉ có {trend.n} năm dữ liệu, dự báo kém tin cậy.")
    backtest_note = ""
    if trend.n >= 12:
        historical_train, historical_test = sub.iloc[:-5], sub.iloc[-5:]
        historical_model = fit_linear_trend(
            historical_train["year"], historical_train["internet_usage_pct"], confidence
        )
        backtest_mae = float(np.mean(np.abs(
            historical_test["internet_usage_pct"].to_numpy()
            - historical_model.predict(historical_test["year"])
        )))
        backtest_note = f" Backtest 5 năm cuối: MAE = {backtest_mae:.1f} điểm %."

    table = pd.DataFrame(
        {
            "year": future_years,
            "forecast_pct": yhat,
            "lower": lo,
            "upper": hi,
        }
    )

    pct = int(confidence * 100)
    fit_years = np.array([sub["year"].min(), last_year])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["year"].to_numpy(), y=sub["internet_usage_pct"].to_numpy(), mode="markers", name="Quan sát (World Bank/ITU)",
                             marker=dict(color=BLUE, size=8),
                             hovertemplate="%{x}: %{y:.1f}%<extra>Quan sát</extra>"))
    fig.add_trace(go.Scatter(x=fit_years, y=trend.predict(fit_years), mode="lines",
                             name="Đường hồi quy", line=dict(color=TREND, width=2), hoverinfo="skip"))
    # Khoảng dự báo: trace 'upper' phải đứng ngay trước trace 'lower' để fill="tonexty" tô đúng vùng.
    fig.add_trace(go.Scatter(x=future_years, y=table["upper"].to_numpy(), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_years, y=table["lower"].to_numpy(), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=TREND_FILL, name=f"Khoảng dự báo {pct}%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.r_[last_year, future_years],
                             y=np.r_[trend.predict(last_year), yhat], mode="lines+markers",
                             name=f"Dự báo {horizon} năm", line=dict(color=TREND, width=2, dash="dash"),
                             marker=dict(size=5), hovertemplate="%{x}: %{y:.1f}%<extra>Dự báo</extra>"))
    fig.add_vline(x=last_year, line_dash="dot", line_color="grey")
    fig.add_hline(y=100, line_dash="dot", line_color="#b9770e")
    _base_layout(fig, title or f"Xu hướng và dự báo Internet Usage — {country}", "Năm", "Người dùng Internet (% dân số)")
    fig.update_yaxes(range=[min(0, float(lo.min()) - 3), max(105, float(hi.max()) + 3)])
    _stat_box(fig, f"Độ dốc {trend.slope:+.2f} điểm %/năm<br>R² = {trend.r2:.3f} · n = {trend.n} năm")
    if warnings and "[0, 100]" in warnings[0]:
        fig.add_annotation(x=0.01, y=0.82, xref="paper", yref="paper", xanchor="left",
                           showarrow=False, align="left", font=dict(color="#b9770e"),
                           text="Cảnh báo: ngoại suy vượt 100%; không phải tỷ lệ khả thi")

    last = table.iloc[-1]
    summary = (
        f"{country}: Internet Usage thay đổi trung bình {trend.slope:+.2f} điểm %/năm "
        f"(R² = {trend.r2:.2f}, {trend.n} năm). Dự báo {int(last.year)}: {last.forecast_pct:.1f}% "
        f"(khoảng {int(confidence * 100)}%: {last.lower:.1f}–{last.upper:.1f}%). "
        f"Dữ liệu cục bộ kết thúc ở {last_year}; đây là ngoại suy tuyến tính từ file đó."
        f"{backtest_note} "
        "Không phải dự báo hành vi người học Coursera."
    )
    return TrendChart(fig, trend, table, summary, warnings)

# 3. Chế độ 2: đường xu hướng giữa hai biến của khóa học
COURSE_LABELS = {
    "hours_to_complete": "Giờ học",
    "duration_weeks": "Tuần học",
    "enrolled_num": "Người học",
    "num_reviews": "Lượt đánh giá",
    "rating_num": "Điểm đánh giá (0 đến 5)",
    "skill_count": "Số kỹ năng",
    "satisfaction_rate_num": "Mức hài lòng (%)",
}


# Biến số dùng được làm x/y. Cố ý KHÔNG có enrolled_percentile, rating_normalized,
# popularity_score vì chúng được tính trực tiếp từ enrollment/rating (leakage, xem README).
TREND_VARIABLES = list(COURSE_LABELS)
_SAME_SNAPSHOT_PAIRS = [{"enrolled_num", "num_reviews"}]

# Mô hình chính: ước lượng enrollment tại cùng một snapshot. Các biến được tính
# từ enrollment (enrolled_percentile, popularity_score, ratio) không nằm ở đây.
ENROLLMENT_COLUMNS = [
    "course_id", "enrolled_num", "num_reviews", "rating_num",
    "hours_to_complete", "skill_count", "level_clean",
]
LEVELS = ("Beginner", "Intermediate", "Advanced", "Not specified")
FEATURE_NAMES = (
    "log10(reviews)", "rating", "log10(hours)", "skill_count",
    "level=Intermediate", "level=Advanced", "level=Not specified",
)


def _enrollment_design(data: pd.DataFrame) -> np.ndarray:
    """Một thiết kế duy nhất cho train, test và biểu mẫu trên Dashboard."""
    return np.column_stack([
        np.ones(len(data)),
        np.log10(data["num_reviews"].to_numpy(dtype=float)),
        data["rating_num"].to_numpy(dtype=float),
        np.log10(data["hours_to_complete"].to_numpy(dtype=float)),
        data["skill_count"].to_numpy(dtype=float),
        *(data["level_clean"].eq(level).to_numpy(dtype=float) for level in LEVELS[1:]),
    ])


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "r2": float(1 - np.sum(residual**2) / total) if total else float("nan"),
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        "mae_log10": float(np.mean(np.abs(residual))),
    }


@dataclass
class EnrollmentModel:
    """OLS trên log10(enrollment); khoảng 95% hiệu chỉnh trên tập calibration."""

    coefficients: np.ndarray
    eligible_rows: int
    total_rows: int
    train_rows: int
    calibration_rows: int
    test_rows: int
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    interval_half_width: float
    test_actual_log: np.ndarray
    test_predicted_log: np.ndarray
    feature_ranges: dict[str, tuple[float, float]]

    def estimate(self, reviews: float, rating: float, hours: float,
                 skills: int, level: str) -> tuple[dict[str, float], list[str]]:
        """Dự đoán trung vị enrollment và khoảng thực nghiệm cho một course có review."""
        values = np.array([reviews, rating, hours, skills], dtype=float)
        if not np.isfinite(values).all() or reviews <= 0 or hours <= 0:
            raise ValueError("Lượt đánh giá và giờ học phải lớn hơn 0.")
        if not 0 <= rating <= 5 or skills < 0 or not float(skills).is_integer():
            raise ValueError("Điểm đánh giá từ 0 đến 5; số kỹ năng là số nguyên từ 0 trở lên.")
        if level not in LEVELS:
            raise ValueError("Trình độ chưa đúng.")

        row = pd.DataFrame([{
            "num_reviews": reviews, "rating_num": rating,
            "hours_to_complete": hours, "skill_count": skills,
            "level_clean": level,
        }])
        prediction_log = float((_enrollment_design(row) @ self.coefficients)[0])
        warnings = []
        for field, value, label in (
            ("num_reviews", reviews, "lượt đánh giá"),
            ("rating_num", rating, "điểm đánh giá"),
            ("hours_to_complete", hours, "giờ học"),
            ("skill_count", skills, "kỹ năng"),
        ):
            lower, upper = self.feature_ranges[field]
            if not lower <= value <= upper:
                warnings.append(f"{label} nằm ngoài khoảng {lower:g} đến {upper:g} của các khóa dùng để tính. Kết quả có thể kém chính xác.")
        return {
            "estimate": float(10**prediction_log),
            "lower": float(10**(prediction_log - self.interval_half_width)),
            "upper": float(10**(prediction_log + self.interval_half_width)),
        }, warnings


def build_enrollment_model(fact: pd.DataFrame, seed: int = 42) -> EnrollmentModel:
    missing = set(ENROLLMENT_COLUMNS).difference(fact.columns)
    if missing:
        raise ValueError(f"Thiếu cột mô hình: {sorted(missing)}")
    if fact["course_id"].isna().any() or fact["course_id"].duplicated().any():
        raise ValueError("course_id phải đầy đủ và duy nhất trước khi chia dữ liệu.")
    data = fact[ENROLLMENT_COLUMNS].copy()
    for column in ENROLLMENT_COLUMNS[1:-1]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=ENROLLMENT_COLUMNS)
    data = data[
        data["enrolled_num"].gt(0) & data["num_reviews"].gt(0)
        & data["hours_to_complete"].gt(0)
        & data["rating_num"].between(0, 5)
        & data["skill_count"].ge(0)
        & data["level_clean"].isin(LEVELS)
    ].reset_index(drop=True)
    if len(data) < 100:
        raise ValueError("Cần ít nhất 100 course đủ dữ liệu để chia train/calibration/test.")

    matrix = _enrollment_design(data)
    target = np.log10(data["enrolled_num"].to_numpy(dtype=float))
    order = np.random.default_rng(seed).permutation(len(data))
    train_end, calibration_end = int(0.6 * len(data)), int(0.8 * len(data))
    train, calibration, test = order[:train_end], order[train_end:calibration_end], order[calibration_end:]
    coefficients = np.linalg.lstsq(matrix[train], target[train], rcond=None)[0]
    calibration_error = np.abs(target[calibration] - matrix[calibration] @ coefficients)
    # Quantile split-conformal 95%; hữu hạn mẫu, không cần giả định phần dư chuẩn.
    rank = min(len(calibration_error), int(np.ceil((len(calibration_error) + 1) * 0.95)))
    half_width = float(np.partition(calibration_error, rank - 1)[rank - 1])
    test_pred = matrix[test] @ coefficients

    # Baseline công khai: chỉ reviews, với chính tập train/test của mô hình đầy đủ.
    baseline_columns = matrix[:, :2]
    baseline_beta = np.linalg.lstsq(baseline_columns[train], target[train], rcond=None)[0]
    baseline = _regression_metrics(target[test], baseline_columns[test] @ baseline_beta)
    ranges = {
        column: (float(data.loc[train, column].min()), float(data.loc[train, column].max()))
        for column in ("num_reviews", "rating_num", "hours_to_complete", "skill_count")
    }
    metrics = _regression_metrics(target[test], test_pred)
    metrics["interval_coverage"] = float(np.mean(np.abs(target[test] - test_pred) <= half_width))
    return EnrollmentModel(
        coefficients=coefficients, eligible_rows=len(data), total_rows=len(fact),
        train_rows=len(train), calibration_rows=len(calibration), test_rows=len(test),
        metrics=metrics, baseline_metrics=baseline, interval_half_width=half_width,
        test_actual_log=target[test], test_predicted_log=test_pred, feature_ranges=ranges,
    )


def enrollment_diagnostic_figures(model: EnrollmentModel) -> tuple[go.Figure, go.Figure]:
    """Biểu đồ test set: giá trị thực so với ước lượng và phần dư log10."""
    actual, predicted = model.test_actual_log, model.test_predicted_log
    lo, hi = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
    comparison = go.Figure()
    comparison.add_scatter(x=10**predicted, y=10**actual, mode="markers",
                           marker=dict(size=6, opacity=0.45, color=BLUE), name="Khóa dùng để kiểm tra")
    comparison.add_scatter(x=[10**lo, 10**hi], y=[10**lo, 10**hi], mode="lines",
                           line=dict(color="gray", dash="dash"), name="Trùng với số thật")
    _base_layout(comparison, "Số thật so với dự đoán", "Người học dự đoán (log10)", "Người học thực tế (log10)")
    comparison.update_xaxes(type="log")
    comparison.update_yaxes(type="log")

    residual = go.Figure()
    residual.add_scatter(x=predicted, y=actual - predicted, mode="markers",
                         marker=dict(size=6, opacity=0.45, color=BLUE), name="Khóa dùng để kiểm tra")
    residual.add_hline(y=0, line_dash="dash", line_color="gray")
    _base_layout(residual, "Mức chênh giữa số thật và dự đoán",
                 "Người học dự đoán (log10)", "Số thật trừ dự đoán (log10)")
    return comparison, residual


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
        raise ValueError("Trục ngang và trục dọc phải là hai chỉ số khác nhau.")
    if n_used < 5:
        raise ValueError(f"Chỉ còn {n_used} khóa có đủ hai chỉ số sau khi lọc; cần ít nhất 5 khóa.")

    fx = np.log10(d[x_col]) if log_x else d[x_col]
    fy = np.log10(d[y_col]) if log_y else d[y_col]
    trend = fit_linear_trend(fx, fy, confidence)

    grid_f = np.linspace(fx.min(), fx.max(), 200)
    grid_x = 10**grid_f if log_x else grid_f
    lo, hi = trend.interval(grid_f, "prediction")
    line = trend.predict(grid_f)
    back = (lambda v: 10**v) if log_y else (lambda v: v)

    warnings: list[str] = []
    if {x_col, y_col} in _SAME_SNAPSHOT_PAIRS:
        rank_corr = float(d[[x_col, y_col]].rank().corr().iloc[0, 1])
        warnings.append(f"Spearman = {rank_corr:.2f}. Hai chỉ số cùng tăng không có nghĩa cái này gây ra cái kia.")
    if trend.r2 < 0.1:
        warnings.append(f"R² = {trend.r2:.3f}: đường xu hướng chỉ giải thích được khoảng {trend.r2:.0%} chênh lệch giữa các khóa.")
    if n_used < len(df):
        warnings.append(f"Chỉ {n_used:,}/{len(df):,} khóa có đủ hai chỉ số để tính.")

    table = pd.DataFrame(columns=[x_col, f"{y_col}_pred", "lower", "upper"])
    if x_new:
        xn = np.asarray(x_new, dtype=float)
        if not np.isfinite(xn).all() or (log_x and (xn <= 0).any()):
            raise ValueError("Giá trị phải lớn hơn 0 khi dùng thang log.")
        if (xn < d[x_col].min()).any() or (xn > d[x_col].max()).any():
            warnings.append(f"Khoảng dữ liệu hiện có là {d[x_col].min():g} đến {d[x_col].max():g}; giá trị nằm ngoài khoảng này có thể kém chính xác.")
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
                             fillcolor=TREND_FILL, name=f"Khoảng dự đoán {pct}%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grid_x, y=back(line), mode="lines", name="Đường xu hướng",
                             line=dict(color=TREND, width=2), hoverinfo="skip"))
    if len(table):
        fig.add_trace(go.Scatter(x=table[x_col].to_numpy(), y=table[f"{y_col}_pred"].to_numpy(), mode="markers", name="Điểm ước lượng",
                                 marker=dict(symbol="diamond", size=11, color=TREND, line=dict(color=BLUE, width=1)),
                                 hovertemplate=f"{xl}: %{{x:,.4g}}<br>Ước lượng: %{{y:,.4g}}<extra></extra>"))
    _base_layout(fig, title or f"{yl} theo {xl}", xl + (" (log10)" if log_x else ""),
                 yl + (" (log10)" if log_y else ""))
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    _stat_box(fig, f"R² = {trend.r2:.3f} · n = {trend.n:,}")

    if log_x and log_y:
        meaning = f"Khi {xl} tăng 10%, {yl} thường thay đổi khoảng {((1.1 ** trend.slope) - 1) * 100:+.1f}% theo đường xu hướng"
    elif log_x:
        meaning = f"Khi {xl} tăng 10 lần, {yl} thường thay đổi {trend.slope:+.4g} theo đường xu hướng"
    elif log_y:
        meaning = f"Khi {xl} tăng 1, {yl} thường thay đổi khoảng {(10**trend.slope - 1) * 100:+.1f}% theo đường xu hướng"
    else:
        meaning = f"Khi {xl} tăng 1, {yl} thường thay đổi {trend.slope:+.4g} theo đường xu hướng"
    summary = (f"{meaning}. "
               f"R² = {trend.r2:.2f}: đường này giải thích được khoảng {trend.r2:.0%} chênh lệch của {yl} giữa {trend.n:,} khóa.")
    return TrendChart(fig, trend, table, summary, warnings)

# 4. Chạy thử: python -m Hoa.prediction
if __name__ == "__main__":
    model = build_enrollment_model(load_fact(ENROLLMENT_COLUMNS))
    print(f"Enrollment model: {model.eligible_rows:,}/{model.total_rows:,} course đủ dữ liệu")
    print(f"Train/calibration/test: {model.train_rows:,}/{model.calibration_rows:,}/{model.test_rows:,}")
    print("Test:", model.metrics)
    print("Baseline chỉ review:", model.baseline_metrics)

    out = PROJECT_DIR / "Hoa" / "outputs" / "trend"
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
