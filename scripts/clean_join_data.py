"""Bước 1: làm sạch và ghép các nguồn dữ liệu khóa học.

Đầu ra chính:
- Data/processed/fact_courses.csv: một dòng cho mỗi URL khóa học.
- Data/processed/bridge_course_subject.csv: quan hệ nhiều-nhiều khóa học–chủ đề.
- Data/processed/audit_same_title_organization.csv: các URL có cùng tên và tổ chức.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "Data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"

FACT_PATH = RAW_DIR / "coursera_course_2024.csv"
SUPPLEMENT_1_PATH = RAW_DIR / "Coursera.csv"
SUPPLEMENT_2_PATH = RAW_DIR / "coursera_course_dataset_v3.csv"

PLACEHOLDERS = {
    "",
    "[]",
    "['']",
    "none",
    "nan",
    "n/a",
    "organization not found",
    "enrollment number not found",
    "rating not found",
    "instructor not found",
}


def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    """Dừng sớm với thông báo rõ ràng nếu schema nguồn thay đổi."""
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source.name} thiếu các cột bắt buộc: {missing}")


def clean_text(series: pd.Series) -> pd.Series:
    """Chuẩn hóa placeholder chính xác thành NA, không xóa cụm từ hợp lệ trong mô tả."""
    result = series.astype("string").str.strip()
    return result.mask(result.str.casefold().isin(PLACEHOLDERS), pd.NA)


def normalize_key(series: pd.Series) -> pd.Series:
    """Tạo khóa join ổn định, giữ NA để tránh ghép nhầm các dòng thiếu khóa."""
    result = (
        series.astype("string")
        .str.normalize("NFKC")
        .str.casefold()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )
    return result.mask(series.isna(), pd.NA)


def parse_number(value: object) -> float:
    """Parse số có dấu phẩy hoặc phần trăm; trả NA nếu không chuyển đổi được."""
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip().replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def make_course_id(url: object) -> str:
    """Tạo mã khóa học ổn định từ URL, không phụ thuộc thứ tự dòng."""
    digest = hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:16]
    return f"course_{digest}"


def first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if not values.empty else pd.NA


def mode_non_null(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def combine_comma_separated(series: pd.Series) -> object:
    """Gộp kỹ năng từ các dòng Subject khác nhau, loại trùng không phân biệt hoa/thường."""
    combined: dict[str, str] = {}
    for value in series.dropna():
        for item in str(value).split(","):
            display = re.sub(r"\s+", " ", item).strip()
            if display:
                combined.setdefault(display.casefold(), display)
    return ", ".join(combined.values()) if combined else pd.NA


def load_fact() -> tuple[pd.DataFrame, pd.DataFrame]:
    fact = pd.read_csv(FACT_PATH)
    require_columns(
        fact,
        {
            "title",
            "enrolled",
            "rating",
            "num_reviews",
            "Instructor",
            "Organization",
            "Skills",
            "Description",
            "Modules/Courses",
            "Level",
            "Schedule",
            "URL",
            "Satisfaction Rate",
        },
        FACT_PATH,
    )
    fact = fact.drop(columns=["Unnamed: 0"], errors="ignore")

    for column in [
        "title",
        "enrolled",
        "rating",
        "Instructor",
        "Organization",
        "Skills",
        "Description",
        "Modules/Courses",
        "Level",
        "Schedule",
        "URL",
        "Satisfaction Rate",
    ]:
        fact[column] = clean_text(fact[column])

    fact["enrolled_num"] = fact["enrolled"].map(parse_number)
    fact["rating_num"] = fact["rating"].map(parse_number)
    fact["satisfaction_rate_num"] = fact["Satisfaction Rate"].map(parse_number)
    fact["num_reviews"] = pd.to_numeric(fact["num_reviews"], errors="coerce")

    fact["_title_key"] = normalize_key(fact["title"])
    fact["_org_key"] = normalize_key(fact["Organization"])
    if fact["URL"].isna().any():
        raise ValueError("URL phải tồn tại cho mọi dòng trong bảng fact.")
    if fact["URL"].duplicated().any():
        duplicates = fact.loc[fact["URL"].duplicated(keep=False), "URL"].unique()
        raise ValueError(
            "URL bị trùng trong nguồn chính; không tự động xóa vì có thể mất dữ liệu: "
            + ", ".join(map(str, duplicates[:10]))
        )

    # Title + organization is only a fuzzy join key. Distinct URLs are preserved
    # because equal names can represent different versions/listings of a course.
    same_name_mask = fact.duplicated(["_title_key", "_org_key"], keep=False)
    same_name_audit = fact.loc[same_name_mask].copy()
    same_name_audit["audit_reason"] = (
        "same normalized title and organization; retained because URL is distinct"
    )

    fact = fact.reset_index(drop=True)
    fact.insert(0, "course_id", fact["URL"].map(make_course_id))
    if fact["course_id"].duplicated().any():
        raise ValueError("course_id bị trùng; cần tăng độ dài hash.")

    return fact, same_name_audit


def load_supplement_1(fact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sup = pd.read_csv(SUPPLEMENT_1_PATH, skipinitialspace=True)
    require_columns(
        sup,
        {"Subject", "Title", "Institution", "Duration", "Gained Skills"},
        SUPPLEMENT_1_PATH,
    )
    for column in ["Subject", "Title", "Institution", "Duration", "Gained Skills"]:
        sup[column] = clean_text(sup[column])
    sup["_title_key"] = normalize_key(sup["Title"])
    sup["_org_key"] = normalize_key(sup["Institution"])

    subject_relations = (
        sup[["_title_key", "_org_key", "Subject"]]
        .dropna(subset=["_title_key", "_org_key", "Subject"])
        .drop_duplicates()
    )
    bridge = fact[["course_id", "title", "Organization", "_title_key", "_org_key"]].merge(
        subject_relations,
        on=["_title_key", "_org_key"],
        how="inner",
        validate="many_to_many",
    )
    bridge = bridge.rename(columns={"Subject": "subject"})
    bridge["subject_normalized"] = bridge["subject"].str.casefold()
    bridge = (
        bridge[
            [
                "course_id",
                "title",
                "Organization",
                "subject",
                "subject_normalized",
            ]
        ]
        .drop_duplicates(["course_id", "subject_normalized"])
        .sort_values(["course_id", "subject_normalized"])
        .reset_index(drop=True)
    )

    attributes = (
        sup.groupby(["_title_key", "_org_key"], dropna=False, as_index=False)
        .agg(
            duration_sup1=("Duration", mode_non_null),
            gained_skills_sup1=("Gained Skills", combine_comma_separated),
        )
    )
    return attributes, bridge


def load_supplement_2() -> pd.DataFrame:
    sup = pd.read_csv(SUPPLEMENT_2_PATH)
    require_columns(
        sup,
        {"Title", "Organization", "Difficulty", "Type", "course_url", "Duration"},
        SUPPLEMENT_2_PATH,
    )
    for column in ["Title", "Organization", "Difficulty", "Type", "course_url", "Duration"]:
        sup[column] = clean_text(sup[column])
    sup["_title_key"] = normalize_key(sup["Title"])
    sup["_org_key"] = normalize_key(sup["Organization"])

    return (
        sup.groupby(["_title_key", "_org_key"], dropna=False, as_index=False)
        .agg(
            Difficulty=("Difficulty", mode_non_null),
            Type=("Type", mode_non_null),
            course_url=("course_url", first_non_null),
            duration_sup2=("Duration", mode_non_null),
        )
    )


def add_subject_summary(fact: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    summary = (
        bridge.groupby("course_id")["subject"]
        .agg(lambda values: sorted(set(values)))
        .rename("_subjects")
    )
    result = fact.merge(summary, on="course_id", how="left", validate="one_to_one")
    result["subjects_combined"] = result["_subjects"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
        if isinstance(value, list)
        else pd.NA
    )
    result["subject_count"] = result["_subjects"].map(
        lambda value: len(value) if isinstance(value, list) else 0
    )
    return result.drop(columns="_subjects")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fact, same_name_audit = load_fact()
    initial_rows = len(fact)
    supplement_1, subject_bridge = load_supplement_1(fact)
    supplement_2 = load_supplement_2()

    merged = fact.merge(
        supplement_1,
        on=["_title_key", "_org_key"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        supplement_2,
        on=["_title_key", "_org_key"],
        how="left",
        validate="many_to_one",
    )
    merged = add_subject_summary(merged, subject_bridge)

    if len(merged) != initial_rows:
        raise AssertionError("Số dòng fact thay đổi sau khi join.")

    merged = merged.drop(columns=["_title_key", "_org_key"])
    same_name_audit = same_name_audit.drop(
        columns=["_title_key", "_org_key"], errors="ignore"
    )

    merged.to_csv(PROCESSED_DIR / "fact_courses.csv", index=False)
    subject_bridge.to_csv(PROCESSED_DIR / "bridge_course_subject.csv", index=False)
    same_name_audit.to_csv(
        PROCESSED_DIR / "audit_same_title_organization.csv", index=False
    )
    (PROCESSED_DIR / "audit_duplicate_courses_removed.csv").unlink(missing_ok=True)

    subject_courses = subject_bridge["course_id"].nunique()
    print(f"Fact rows after dedupe/join: {len(merged):,}")
    print(
        "Courses with subjects:",
        f"{subject_courses:,}/{len(merged):,}",
        f"({subject_courses / len(merged) * 100:.1f}%)",
    )
    print(f"Course-subject relationships: {len(subject_bridge):,}")
    print(f"Saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
