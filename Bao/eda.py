from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "Bao" / "outputs" / "eda"

FACT_PATH = PROCESSED_DIR / "fact_courses_eda_ready.csv"
SUBJECT_BRIDGE_PATH = PROCESSED_DIR / "bridge_course_subject.csv"
SKILL_BRIDGE_PATH = PROCESSED_DIR / "bridge_course_skill.csv"

# Format định dạng biểu đồ
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Tahoma", "DejaVu Sans"]

PLACEHOLDER_VALUES = {
    "",
    "[]",
    "['']",
    "none",
    "nan",
    "n/a",
    "not found",
    "organization not found",
    "enrollment number not found",
    "rating not found",
    "instructor not found",
}

# Kiểm tra cột bắt buộc
def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source.name} thiếu cột bắt buộc: {missing}")

# Tải dữ liệu đầu vào
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [FACT_PATH, SUBJECT_BRIDGE_PATH, SKILL_BRIDGE_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Chưa có {path}. Hãy chạy theo thứ tự trong README."
            )

    fact = pd.read_csv(FACT_PATH, low_memory=False)
    subjects = pd.read_csv(SUBJECT_BRIDGE_PATH)
    skills = pd.read_csv(SKILL_BRIDGE_PATH)

    require_columns(
        fact,
        {
            "course_id",
            "title",
            "URL",
            "Organization",
            "organization_hq_country",
            "enrolled_num",
            "rating_num",
            "num_reviews",
            "satisfaction_rate_num",
            "hours_to_complete",
            "duration_weeks",
            "level_clean",
            "popularity_score",
            "review_to_enrollment_ratio",
            "internet_usage_pct",
            "internet_usage_year",
        },
        FACT_PATH,
    )
    # Kiểm tra cột bắt buộc trong các bảng bridge
    require_columns(subjects, {"course_id", "subject", "subject_normalized"}, SUBJECT_BRIDGE_PATH)
    require_columns(skills, {"course_id", "skill", "skill_normalized"}, SKILL_BRIDGE_PATH)

    # Kiểm tra tính duy nhất và tồn tại của course_id
    if fact["course_id"].isna().any() or fact["course_id"].duplicated().any():
        raise ValueError("fact_courses_eda_ready.csv phải có course_id đầy đủ và duy nhất.")
    unknown_subject_ids = set(subjects["course_id"]).difference(fact["course_id"])
    unknown_skill_ids = set(skills["course_id"]).difference(fact["course_id"])
    if unknown_subject_ids or unknown_skill_ids:
        raise ValueError("Bridge chứa course_id không tồn tại trong fact.")
    if subjects.duplicated(["course_id", "subject_normalized"]).any():
        raise ValueError("bridge_course_subject có quan hệ trùng.")
    if skills.duplicated(["course_id", "skill_normalized"]).any():
        raise ValueError("bridge_course_skill có quan hệ trùng.")
    return fact, subjects, skills

# Kiểm tra placeholder trong cột
def placeholder_mask(series: pd.Series) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return pd.Series(False, index=series.index)
    normalized = series.astype("string").str.strip().str.casefold()
    return normalized.isin(PLACEHOLDER_VALUES).fillna(False)

# Kiểm tra giá trị invalid trong cột
def invalid_mask(df: pd.DataFrame, column: str) -> pd.Series:
    result = pd.Series(False, index=df.index)
    ranges = {
        "rating_num": (0, 5),
        "rating_normalized": (0, 1),
        "satisfaction_rate_num": (0, 100),
        "popularity_score": (0, 1),
        "review_to_enrollment_ratio": (0, 1),
        "internet_usage_pct": (0, 100),
    }
    nonnegative = {
        "enrolled_num",
        "num_reviews",
        "hours_to_complete",
        "hours_to_complete_reported",
        "hours_to_complete_estimated",
        "hours_per_week",
        "duration_weeks",
        "subject_count",
        "skill_count",
    }

    # Kiểm tra giá trị invalid dựa trên phạm vi hoặc điều kiện không âm
    if column in ranges:
        low, high = ranges[column]
        result = df[column].notna() & ~df[column].between(low, high)
    elif column in nonnegative:
        result = df[column].notna() & df[column].lt(0)
    elif column in {"URL", "course_url"}:
        present = df[column].notna()
        result = present & ~df[column].astype("string").str.startswith(
            ("http://", "https://"), na=False
        )
    return result

# Ghi chú cho cột dựa trên tên cột và tỷ lệ missing
def column_note(column: str, missing_pct: float) -> str:
    notes = {
        "organization_hq_country": "Headquarters only; not learner location.",
        "organization_hq_country_status": "Manual mapping verification status; citations are still required.",
        "internet_usage_pct": "Context for HQ country; observations use mixed years.",
        "internet_usage_year": "Always interpret together with internet_usage_pct.",
        "popularity_score": "Derived from rating and enrollment; not independent.",
        "duration_weeks": "Month schedules converted with 1 month = 4.345 weeks.",
        "hours_to_complete": "Mix of reported and explicitly flagged estimated values.",
        "Difficulty": "Supplemental matched subset only.",
        "Type": "Supplemental matched subset only.",
        "course_url": "Supplemental matched subset only; URL is the canonical URL.",
        "subjects_combined": "Use bridge_course_subject.csv for analysis.",
        "skills_combined": "Use bridge_course_skill.csv for analysis.",
    }
    note = notes.get(column, "")
    if missing_pct >= 50:
        note = (note + " " if note else "") + "High missingness; do not generalize without caveat."
    return note

# Xây dựng bảng chất lượng dữ liệu
def build_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in df.columns:
        null_count = int(df[column].isna().sum())
        placeholders = placeholder_mask(df[column]) & df[column].notna()
        placeholder_count = int(placeholders.sum())
        effective_missing = null_count + placeholder_count
        missing_pct = effective_missing / len(df) * 100
        rows.append(
            {
                "Column": column,
                "Data_Type": str(df[column].dtype),
                "Row_Count": len(df),
                "Null_Count": null_count,
                "Placeholder_Count": placeholder_count,
                "Effective_Missing_Count": effective_missing,
                "Effective_Missing_Percentage": missing_pct,
                "Coverage_Percentage": 100 - missing_pct,
                "Unique_Count": int(df[column].nunique(dropna=True)),
                "Invalid_Count": int(invalid_mask(df, column).sum()),
                "Notes": column_note(column, missing_pct),
            }
        )
    return pd.DataFrame(rows)

# Thống kê mô tả cho các biến số chính
def build_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "enrolled_num",
        "rating_num",
        "num_reviews",
        "satisfaction_rate_num",
        "hours_to_complete",
        "duration_weeks",
        "hours_per_week",
        "review_to_enrollment_ratio",
        "popularity_score",
        "internet_usage_pct",
    ]
    rows: list[dict[str, object]] = []

    # Tính toán các thống kê mô tả cho từng cột
    for column in columns:
        series = df[column].dropna()
        rows.append(
            {
                "Variable": column,
                "Count": int(series.count()),
                "Missing_Percentage": df[column].isna().mean() * 100,
                "Mean": series.mean(),
                "Median": series.median(),
                "Std": series.std(),
                "Min": series.min(),
                "Q1": series.quantile(0.25),
                "Q3": series.quantile(0.75),
                "P95": series.quantile(0.95),
                "P99": series.quantile(0.99),
                "Max": series.max(),
                "Skewness": series.skew(),
            }
        )
    return pd.DataFrame(rows)

# Thêm nhãn giá trị vào biểu đồ
def add_bar_labels(ax: plt.Axes, fmt: str = "{:,.0f}") -> None:
    for container in ax.containers:
        labels = [fmt.format(value) for value in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

def sample_title(label: str, series: pd.Series, total: int) -> str:
    count = int(series.notna().sum())
    missing = (1 - count / total) * 100
    return f"{label}\nn={count:,}; thiếu={missing:.1f}%"

# Phân phối các biến số chính
def save_numeric_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Phân phối các biến số chính", fontsize=17, fontweight="bold")
    total = len(df)

    plots = [
        ("enrolled_num", "Số học viên", "#4C72B0", True),
        ("num_reviews", "Số reviews", "#55A868", True),
        ("rating_num", "Điểm đánh giá", "#C44E52", False),
        ("satisfaction_rate_num", "Satisfaction rate (%)", "#8172B2", False),
        ("hours_to_complete", "Tổng số giờ", "#CCB974", True),
        ("popularity_score", "Popularity score", "#64B5CD", False),
    ]

    for ax, (column, label, color, log_scale) in zip(axes.flat, plots):
        values = df[column].dropna()
        sns.histplot(values, bins=35, kde=True, ax=ax, color=color, log_scale=log_scale)
        ax.set_title(sample_title(label, df[column], total), fontsize=11, fontweight="bold")
        ax.set_xlabel(label + (" (log scale)" if log_scale else ""))
        ax.set_ylabel("Tần suất")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "03_numeric_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Phân phối cấp độ khóa học
def save_level_distribution(df: pd.DataFrame) -> None:
    order = df["level_clean"].value_counts().index
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.countplot(data=df, y="level_clean", order=order, hue="level_clean", palette="pastel", legend=False, ax=ax)
    ax.set_title("Số lượng khóa học theo độ khó", fontweight="bold")
    ax.set_xlabel("Số lượng khóa học")
    ax.set_ylabel("Cấp độ")
    add_bar_labels(ax)
    fig.tight_layout()
    fig.text(0.99, 0.01, "* Ghi chú: 'Not specified' là Level bị thiếu",
             ha="right", va="bottom", fontsize=10, style="italic", color="dimgray")
    fig.savefig(OUTPUT_DIR / "04_level_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Phân phối Subject multi-label
def save_subject_distribution(df: pd.DataFrame, subjects: pd.DataFrame) -> None:
    # Đếm số khóa học theo subject và lấy top 10
    counts = (
        subjects.groupby(["subject_normalized", "subject"])["course_id"]
        .nunique()
        .reset_index(name="course_count")
        .sort_values("course_count", ascending=False)
        .head(10)
    )

    # Số khóa học có subject để hiển thị coverage
    covered = subjects["course_id"].nunique()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=counts, x="course_count", y="subject", hue="subject", palette="mako", legend=False, ax=ax)
    ax.set_title("Số khóa học theo Subject (multi-label)\n", fontweight="bold")
    ax.set_xlabel("Số khóa học có Subject")
    ax.set_ylabel("")
    add_bar_labels(ax)
    fig.tight_layout()
    note_text = f"* Ghi chú: Độ phủ: {covered:,}/{len(df):,} khóa học ({covered / len(df) * 100:.1f}%)"
    fig.text(0.99, 0.01, note_text, ha="right", va="bottom", fontsize=9.5, style="italic", color="dimgray")
    fig.savefig(OUTPUT_DIR / "05_subject_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Phân phối điểm đánh giá theo cấp độ khóa học
def save_rating_by_level(df: pd.DataFrame) -> None:
    order = ["Beginner", "Intermediate", "Advanced", "Not specified"]

    # Đếm số lượng khóa học theo level để hiển thị trên nhãn trục x
    counts = df.groupby("level_clean")["rating_num"].count().to_dict()

    labels = [f"{level}\n(n={counts.get(level, 0):,})" for level in order]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df, x="level_clean", y="rating_num", hue="level_clean", order=order, palette="Set2", legend=False, ax=ax)
    ax.set_xticks(range(len(order)), labels)
    ax.set_title("Phân bố rating theo mức độ khó", fontweight="bold")
    ax.set_xlabel("Cấp độ")
    ax.set_ylabel("Điểm đánh giá")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_rating_by_level.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Ma trận tương quan Spearman giữa các biến số chính
def save_correlation(df: pd.DataFrame) -> pd.DataFrame:
    # Tương quan Spearman giữa các biến số chính
    columns = [
        "enrolled_num",
        "rating_num",
        "num_reviews",
        "satisfaction_rate_num",
        "hours_to_complete",
    ]

    labels = ["Học viên", "Rating", "Reviews", "Hài lòng", "Số giờ"]
    data = df[columns]
    correlation = data.corr(method="spearman")
    present = data.notna().astype(int)
    pair_counts = present.T.dot(present)
    pair_counts.to_csv(OUTPUT_DIR / "07_correlation_sample_size.csv")

    # Heatmap tương quan và số mẫu từng cặp
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Tương quan Spearman", fontweight="bold")
    sns.heatmap(pair_counts, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Số quan sát dùng cho từng cặp", fontweight="bold")
    fig.suptitle("Phân tích tương quan giữa các chỉ số tương tác của khóa học", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    note_text = "* Ghi chú: Tương quan thống kê không hàm ý quan hệ nhân quả."
    fig.text(0.99, 0.01, note_text, ha="right", va="bottom", fontsize=11, style="italic", color="dimgray")
    fig.savefig(OUTPUT_DIR / "07_correlation_spearman.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return correlation

# Top 10 tổ chức có nhiều khóa học nhất
def save_top_organizations(df: pd.DataFrame) -> None:
    # Đếm số khóa học theo tổ chức và lấy top 10
    counts = df["Organization"].value_counts().head(10).rename_axis("Organization").reset_index(name="course_count")

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=counts, x="course_count", y="Organization", hue="Organization", palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 10 tổ chức có nhiều khóa học nhất", fontweight="bold")
    ax.set_xlabel("Số lượng khóa học")
    ax.set_ylabel("")
    add_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_top_organizations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Top 10 quốc gia trụ sở tổ chức theo số khóa học và số tổ chức duy nhất
def save_hq_countries(df: pd.DataFrame) -> None:
    # Số lượng khóa học theo quốc gia trụ sở
    course_counts = (df["organization_hq_country"]
                     .value_counts()
                     .head(10)
                     .rename_axis("country")
                     .reset_index(name="course_count"))

    # Số lượng tổ chức duy nhất theo quốc gia trụ sở
    org_counts = (
        df.dropna(subset=["organization_hq_country", "Organization"])
        .groupby("organization_hq_country")["Organization"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .rename_axis("country")
        .reset_index(name="organization_count")
    )

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    sns.barplot(data=course_counts, x="course_count", y="country", hue="country", palette="magma", legend=False, ax=axes[0])
    axes[0].set_title("Theo số khóa học", fontweight="bold")
    axes[0].set_xlabel("Số khóa học")
    axes[0].set_ylabel("")
    add_bar_labels(axes[0])
    sns.barplot(data=org_counts, x="organization_count", y="country", hue="country", palette="crest", legend=False, ax=axes[1])
    axes[1].set_title("Theo số tổ chức duy nhất", fontweight="bold")
    axes[1].set_xlabel("Số tổ chức")
    axes[1].set_ylabel("")
    add_bar_labels(axes[1])
    fig.suptitle("Quốc gia trụ sở tổ chức", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "09_organization_hq_countries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Top 25 cột có tỷ lệ missing cao nhất
def save_missing_values(quality: pd.DataFrame) -> None:
    # Lọc các cột có missing > 0 & sắp xếp theo tỷ lệ missing giảm dần
    # Lấy top 25
    missing = (
        quality.loc[quality["Effective_Missing_Count"].gt(0)]
        .sort_values("Effective_Missing_Percentage", ascending=False)
        .head(25)
    )

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.barplot(data=missing, x="Effective_Missing_Percentage", y="Column", hue="Column", palette="Reds_r", legend=False, ax=ax)
    ax.set_title("Top 25 cột có tỷ lệ thiếu hiệu dụng cao nhất", fontweight="bold")
    ax.set_xlabel("Phần trăm thiếu (%)")
    ax.set_ylabel("Tên cột")
    for index, value in enumerate(missing["Effective_Missing_Percentage"]):
        ax.text(value + 0.5, index, f"{value:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.text(0.99, 0.01, "* Chi tiết đầy đủ xem tại: 01_data_quality.csv",
             ha="right", va="bottom", fontsize=10, style="italic", color="dimgray")
    fig.savefig(OUTPUT_DIR / "10_missing_values.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Top 15 kỹ năng theo số khóa học, đã loại trùng trong từng khóa
def save_top_skills(df: pd.DataFrame, skills: pd.DataFrame) -> None:
    # Lấy tên hiển thị phổ biến nhất cho mỗi skill_normalized
    display_names = skills.groupby("skill_normalized")["skill"].agg(lambda values: values.mode().iloc[0])

    # Đếm số khóa học theo skill_normalized và lấy top 15
    counts = skills.groupby("skill_normalized")["course_id"].nunique().sort_values(ascending=False).head(15)

    top = counts.rename("course_count").reset_index()
    top["skill"] = top["skill_normalized"].map(display_names)
    covered = skills["course_id"].nunique()

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.barplot(data=top, x="course_count", y="skill", hue="skill", palette="mako", legend=False, ax=ax)
    ax.set_title("Top 15 kỹ năng theo số khóa học", fontweight="bold")
    ax.set_xlabel("Số khóa học")
    ax.set_ylabel("")
    add_bar_labels(ax)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    note_text = f"* Ghi chú: Mỗi kỹ năng đếm 1 lần/khóa | Độ phủ (coverage): {covered:,}/{len(df):,} khóa học ({covered / len(df) * 100:.1f}%)"
    fig.text(0.99, 0.01, note_text, ha="right", va="bottom", fontsize=10, style="italic", color="dimgray")
    fig.savefig(OUTPUT_DIR / "11_top_skills.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Bối cảnh internet tại quốc gia trụ sở tổ chức
def save_internet_context(df: pd.DataFrame) -> None:
    # Tóm tắt dữ liệu theo trụ sở quốc gia
    summary = (
        df.dropna(subset=["organization_hq_country", "internet_usage_pct"])
        .groupby("organization_hq_country", as_index=False)
        .agg(
            course_count=("course_id", "count"),
            organization_count=("Organization", "nunique"),
            internet_usage_pct=("internet_usage_pct", "first"),
            internet_usage_year=("internet_usage_year", "first"),
        )
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=summary,
        x="internet_usage_pct",
        y="course_count",
        size="organization_count",
        sizes=(50, 500),
        alpha=0.75,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Bối cảnh Internet tại quốc gia trụ sở", fontweight="bold")
    ax.set_xlabel("Tỷ lệ sử dụng Internet (%) – năm gần nhất trong nguồn cục bộ")
    ax.set_ylabel("Số khóa học")
    top_rows = summary.nlargest(5, "course_count")
    offsets = [(8, 10), (8, 8), (8, 8), (8, 12), (8, -14)]
    for row, offset in zip(top_rows.itertuples(index=False), offsets):
        ax.annotate(
            f"{row.organization_hq_country} ({int(row.internet_usage_year)})",
            (row.internet_usage_pct, row.course_count),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
        )
    ax.set_xlim(summary["internet_usage_pct"].min() - 3, 105)
    legend = ax.get_legend()
    if legend:
        legend.set_title("Số tổ chức")
        legend.set_bbox_to_anchor((1.01, 1))
        legend.set_loc("upper left")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    note_text = "* Ghi chú: Dữ liệu mang tính mô tả bối cảnh, không phải bằng chứng về hành vi người học."
    fig.text(0.99, 0.01, note_text, ha="right", va="bottom", fontsize=10, style="italic", color="dimgray")
    fig.savefig(OUTPUT_DIR / "12_internet_hq_context.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Tổng quan EDA
def write_report(
    df: pd.DataFrame,
    subjects: pd.DataFrame,
    skills: pd.DataFrame,
    quality: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    subject_courses = subjects["course_id"].nunique()
    skill_courses = skills["course_id"].nunique()
    complete_popularity = df["popularity_score"].notna().sum()
    estimated_hours = int(df["hours_to_complete_is_estimated"].fillna(False).sum())
    top_org = df["Organization"].value_counts().idxmax()
    top_org_count = int(df["Organization"].value_counts().max())
    corr_value = correlation.loc["enrolled_num", "num_reviews"]
    difficulty_coverage = (1 - df["Difficulty"].isna().mean()) * 100 if "Difficulty" in df else 0
    type_coverage = (1 - df["Type"].isna().mean()) * 100 if "Type" in df else 0
    manual_country_count = int(
        df["organization_hq_country_status"]
        .eq("manual_mapping_requires_citation")
        .sum()
    )

    report = f"""# EDA Report — Coursera Courses

## 1. Phạm vi và dữ liệu

- Bảng fact: `{FACT_PATH.relative_to(PROJECT_DIR)}`.
- Quy mô: **{len(df):,} khóa học**, **{len(df.columns):,} cột**; `course_id` duy nhất.
- Dữ liệu là snapshot, không phải dữ liệu Coursera thời gian thực.
- Fact giữ đủ **{len(df):,} URL duy nhất** từ nguồn chính; URL là khóa định danh, không tự xóa các URL chỉ vì trùng tên khóa học.
- Không sinh dữ liệu khóa học giả. Các cột duration quy đổi, percentile và popularity là biến dẫn xuất, có quy tắc/cờ nhận diện.
- Country trong phân tích là **quốc gia trụ sở tổ chức**, không phải quốc gia người học.

## 2. Kiểm định chất lượng

- Không còn placeholder chuẩn (`not found`, `[]`) trong các cột canonical; xem `01_data_quality.csv`.
- Popularity chỉ được tính khi có cả rating và enrollment: **{complete_popularity:,}/{len(df):,}** khóa.
- Duration tháng được đổi theo `1 tháng = 4.345 tuần`; **{estimated_hours:,}** giá trị tổng giờ là ước lượng và có cờ nhận diện.
- Subject coverage: **{subject_courses:,}/{len(df):,} ({subject_courses / len(df) * 100:.1f}%)**, lưu dưới dạng bridge multi-label.
- Skill coverage: **{skill_courses:,}/{len(df):,} ({skill_courses / len(df) * 100:.1f}%)**, đã chuẩn hóa hoa/thường và loại trùng trong từng khóa.
- Difficulty coverage: **{difficulty_coverage:.1f}%**; Type coverage: **{type_coverage:.1f}%**. Hai trường này chỉ đại diện tập con đã match.
- HQ country của **{manual_country_count:,}** khóa đến từ mapping thủ công; mapping chưa có nguồn riêng cho từng tổ chức nên chỉ dùng thăm dò.

## 3. Insight mô tả được dữ liệu hỗ trợ

- Enrollment và review lệch phải mạnh; biểu đồ sử dụng log scale.
- Tương quan hạng Spearman giữa enrollment và review là **{corr_value:.2f}**. Đây là tương quan, không chứng minh nhân quả.
- Tổ chức có nhiều khóa học nhất là **{top_org}** với **{top_org_count:,}** khóa.
- Rating giữa các level có thể so sánh mô tả bằng boxplot, nhưng cần kiểm định/effect size trước khi kết luận khác biệt thống kê.
- Subject và skill chỉ được kết luận trong phạm vi coverage được ghi trực tiếp trên biểu đồ.

## 4. Giới hạn bắt buộc khi diễn giải

1. Không suy luận quốc gia/ngữ cảnh Internet của người học từ quốc gia trụ sở tổ chức.
2. Internet Usage dùng năm gần nhất khác nhau giữa quốc gia; phải đọc cùng `internet_usage_year`.
3. `popularity_score` là proxy 50% rating chuẩn hóa + 50% percentile enrollment, không phải nhãn popularity quan sát được; không dùng các thành phần của nó làm feature để dự đoán chính nó.
4. Dataset không có chuỗi thời gian enrollment, nên không hỗ trợ time-series forecasting đúng nghĩa.
5. Subject, Difficulty, Type và Satisfaction có missing cao; mọi insight phải công bố coverage.
6. Các giá trị thời lượng quy đổi từ tháng là ước lượng, không phải tổng giờ do Coursera báo cáo.
7. Mapping quốc gia trụ sở là thủ công và chưa có citation theo từng tổ chức; không dùng biểu đồ HQ/Internet làm kết luận chính thức trước khi xác minh.

## 5. Mức sẵn sàng cho Insight và Prediction

- Có thể bắt đầu Insight từ `fact_courses_eda_ready.csv` và hai bảng bridge.
- Insight về quốc gia/Internet chỉ ở mức exploratory cho đến khi hoàn tất nguồn mapping HQ.
- Trước Prediction phải xác định target; loại mọi feature được tính trực tiếp từ target để tránh leakage.
- Nếu dự đoán enrollment: không dùng `enrolled_percentile`, `popularity_score` hoặc biến tạo từ enrollment.
- Nếu dự đoán rating: không dùng `rating_normalized` hoặc `popularity_score`.
- Nếu dự đoán popularity: không dùng rating/enrollment và các biến chuẩn hóa của chúng như feature.

## 6. Đầu ra EDA

- `00_pipeline_validation.csv`: PASS/FAIL kiểm định độc lập; được tạo bởi `validate_pipeline.py` sau EDA.
- `01_data_quality.csv`: chất lượng đầy đủ của mọi cột.
- `02_descriptive_statistics.csv`: thống kê mô tả mở rộng.
- `03_numeric_distributions.png`: phân phối biến số.
- `04_level_distribution.png`: phân bố level.
- `05_subject_distribution.png`: Subject multi-label và coverage.
- `06_rating_by_level.png`: rating theo level kèm cỡ mẫu.
- `07_correlation_spearman.png`: Spearman và số mẫu từng cặp.
- `07_correlation_sample_size.csv`: số mẫu tương quan.
- `08_top_organizations.png`: top tổ chức theo số khóa.
- `09_organization_hq_countries.png`: HQ theo số khóa và số tổ chức.
- `10_missing_values.png`: top missing; chi tiết đầy đủ nằm trong CSV.
- `11_top_skills.png`: skill theo số khóa, đã dedupe.
- `12_internet_hq_context.png`: bối cảnh Internet tại HQ với cảnh báo diễn giải.
"""
    (OUTPUT_DIR / "EDA_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, subjects, skills = load_inputs()

    quality = build_data_quality(df)
    descriptive = build_descriptive_statistics(df)
    quality.to_csv(OUTPUT_DIR / "01_data_quality.csv", index=False)
    descriptive.to_csv(OUTPUT_DIR / "02_descriptive_statistics.csv", index=False)

    save_numeric_distributions(df)
    save_level_distribution(df)
    save_subject_distribution(df, subjects)
    save_rating_by_level(df)
    correlation = save_correlation(df)
    save_top_organizations(df)
    save_hq_countries(df)
    save_missing_values(quality)
    save_top_skills(df, skills)
    save_internet_context(df)
    write_report(df, subjects, skills, quality, correlation)

    invalid_total = int(quality["Invalid_Count"].sum())
    placeholder_total = int(quality["Placeholder_Count"].sum())
    if invalid_total or placeholder_total:
        raise ValueError(
            f"EDA phát hiện {invalid_total} giá trị invalid và "
            f"{placeholder_total} placeholder còn sót. Xem 01_data_quality.csv."
        )
    print(f"EDA completed: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
