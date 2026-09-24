"""Bước 3: tạo các trường dẫn xuất và bảng bridge kỹ năng.

Các nguyên tắc quan trọng:
- Missing không được biến thành 0.
- Thời lượng quy đổi từ tháng được đánh dấu là ước lượng.
- Một kỹ năng chỉ xuất hiện tối đa một lần cho mỗi khóa học.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
MONTH_TO_WEEKS = 4.345

NUMBER = r"(\d+(?:\.\d+)?)"
HOURS_PATTERN = re.compile(rf"{NUMBER}\s*hours?\s+to\s+complete", re.IGNORECASE)
WEEKS_PATTERN = re.compile(rf"{NUMBER}\s*weeks?", re.IGNORECASE)
MONTHS_PATTERN = re.compile(rf"{NUMBER}\s*months?", re.IGNORECASE)
HOURS_PER_WEEK_PATTERN = re.compile(
    rf"at\s+{NUMBER}\s*hours?\s+a\s+week", re.IGNORECASE
)


def match_number(pattern: re.Pattern[str], text: str) -> float:
    match = pattern.search(text)
    return float(match.group(1)) if match else np.nan


def parse_schedule(value: object) -> pd.Series:
    """Parse lịch học và phân biệt rõ số liệu báo cáo với số liệu ước lượng."""
    result = {
        "hours_to_complete_reported": np.nan,
        "hours_to_complete_estimated": np.nan,
        "hours_to_complete": np.nan,
        "hours_per_week": np.nan,
        "duration_weeks": np.nan,
        "duration_weeks_is_estimated": pd.NA,
        "hours_to_complete_is_estimated": pd.NA,
        "schedule_parse_status": "missing",
    }
    if pd.isna(value):
        return pd.Series(result)

    text = str(value).strip()
    reported_hours = match_number(HOURS_PATTERN, text)
    reported_weeks = match_number(WEEKS_PATTERN, text)
    reported_months = match_number(MONTHS_PATTERN, text)
    hours_per_week = match_number(HOURS_PER_WEEK_PATTERN, text)

    result["hours_to_complete_reported"] = reported_hours
    result["hours_per_week"] = hours_per_week

    if pd.notna(reported_weeks):
        result["duration_weeks"] = reported_weeks
        result["duration_weeks_is_estimated"] = False
    elif pd.notna(reported_months):
        result["duration_weeks"] = reported_months * MONTH_TO_WEEKS
        result["duration_weeks_is_estimated"] = True

    if pd.notna(reported_hours):
        result["hours_to_complete"] = reported_hours
        result["hours_to_complete_is_estimated"] = False
    elif pd.notna(result["duration_weeks"]) and pd.notna(hours_per_week):
        estimated_hours = result["duration_weeks"] * hours_per_week
        result["hours_to_complete_estimated"] = estimated_hours
        result["hours_to_complete"] = estimated_hours
        result["hours_to_complete_is_estimated"] = True

    if pd.notna(reported_months):
        result["schedule_parse_status"] = "estimated_from_months"
    elif pd.notna(reported_hours) and pd.notna(reported_weeks):
        result["schedule_parse_status"] = "reported_hours_and_weeks"
    elif pd.notna(reported_hours):
        result["schedule_parse_status"] = "reported_hours_only"
    else:
        result["schedule_parse_status"] = "unparsed"

    return pd.Series(result)


def parse_skills(value: object) -> list[str]:
    """Parse danh sách Python hoặc chuỗi phân tách bằng dấu phẩy."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text in {"[]", "['']"}:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Không parse được danh sách skill: {text[:120]}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"Skill dạng ngoặc vuông nhưng không phải list: {text[:120]}")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def build_skill_bridge(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo bảng nhiều-nhiều course-skill và tóm tắt skill trong fact."""
    bridge_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for row in df[
        ["course_id", "title", "Skills", "gained_skills_sup1"]
    ].itertuples(index=False):
        skill_map: dict[str, dict[str, object]] = {}
        for source, raw_value in (
            ("primary", row.Skills),
            ("supplemental", row.gained_skills_sup1),
        ):
            for skill in parse_skills(raw_value):
                display = re.sub(r"\s+", " ", skill).strip()
                normalized = display.casefold()
                if normalized not in skill_map:
                    skill_map[normalized] = {"display": display, "sources": {source}}
                else:
                    skill_map[normalized]["sources"].add(source)
                    if source == "primary":
                        skill_map[normalized]["display"] = display

        displays: list[str] = []
        for normalized in sorted(skill_map):
            info = skill_map[normalized]
            display = str(info["display"])
            displays.append(display)
            bridge_rows.append(
                {
                    "course_id": row.course_id,
                    "title": row.title,
                    "skill": display,
                    "skill_normalized": normalized,
                    "skill_source": "|".join(sorted(info["sources"])),
                }
            )
        summaries.append(
            {
                "course_id": row.course_id,
                "skills_combined": json.dumps(displays, ensure_ascii=False)
                if displays
                else pd.NA,
                "skill_count": len(displays),
            }
        )

    bridge = pd.DataFrame(bridge_rows)
    if not bridge.empty:
        bridge = (
            bridge.drop_duplicates(["course_id", "skill_normalized"])
            .sort_values(["course_id", "skill_normalized"])
            .reset_index(drop=True)
        )
    summary = pd.DataFrame(summaries)
    return bridge, summary


def validate_ranges(df: pd.DataFrame) -> None:
    checks = {
        "rating_num": (0, 5),
        "rating_normalized": (0, 1),
        "satisfaction_rate_num": (0, 100),
        "popularity_score": (0, 1),
        "review_to_enrollment_ratio": (0, 1),
    }
    for column, (lower, upper) in checks.items():
        invalid = df[column].notna() & ~df[column].between(lower, upper)
        if invalid.any():
            raise ValueError(f"{column} có {invalid.sum()} giá trị ngoài [{lower}, {upper}].")


def main() -> None:
    input_path = PROCESSED_DIR / "fact_courses_with_country.csv"
    df = pd.read_csv(input_path)
    required = {
        "course_id",
        "title",
        "Schedule",
        "Level",
        "enrolled_num",
        "rating_num",
        "num_reviews",
        "Skills",
        "gained_skills_sup1",
        "Organization",
        "organization_hq_country",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{input_path.name} thiếu cột: {missing}")
    row_count = len(df)

    schedule_fields = df["Schedule"].apply(parse_schedule)
    df = pd.concat([df, schedule_fields], axis=1)
    df["duration_weeks_is_estimated"] = df[
        "duration_weeks_is_estimated"
    ].astype("boolean")
    df["hours_to_complete_is_estimated"] = df[
        "hours_to_complete_is_estimated"
    ].astype("boolean")

    df["level_clean"] = (
        df["Level"]
        .astype("string")
        .str.replace(r"\s+level$", "", regex=True, case=False)
        .fillna("Not specified")
    )

    valid_enrollment = df["enrolled_num"].notna() & df["enrolled_num"].gt(0)
    df["review_to_enrollment_ratio"] = np.where(
        valid_enrollment,
        df["num_reviews"] / df["enrolled_num"],
        np.nan,
    )
    df["enrolled_percentile"] = df["enrolled_num"].rank(pct=True)
    df["rating_normalized"] = df["rating_num"] / 5.0

    popularity_complete = df["rating_normalized"].notna() & df[
        "enrolled_percentile"
    ].notna()
    df["popularity_score"] = np.where(
        popularity_complete,
        0.5 * df["rating_normalized"] + 0.5 * df["enrolled_percentile"],
        np.nan,
    )
    df["popularity_score_status"] = np.select(
        [
            df["rating_num"].isna() & df["enrolled_num"].isna(),
            df["rating_num"].isna(),
            df["enrolled_num"].isna(),
        ],
        ["missing_both", "missing_rating", "missing_enrollment"],
        default="complete",
    )

    skill_bridge, skill_summary = build_skill_bridge(df)
    df = df.drop(columns=["skills_combined", "skill_count"], errors="ignore")
    df = df.merge(skill_summary, on="course_id", how="left", validate="one_to_one")

    df["courses_per_organization"] = df.groupby("Organization")["course_id"].transform(
        "count"
    )
    df["courses_per_hq_country"] = df.groupby("organization_hq_country")[
        "course_id"
    ].transform("count")

    if len(df) != row_count or df["course_id"].nunique() != row_count:
        raise AssertionError("Số dòng hoặc số course_id thay đổi khi tạo calculated fields.")
    validate_ranges(df)

    df.to_csv(PROCESSED_DIR / "fact_courses_final.csv", index=False)
    skill_bridge.to_csv(PROCESSED_DIR / "bridge_course_skill.csv", index=False)

    print(f"Fact calculated: {len(df):,} rows, {len(df.columns):,} columns")
    print("Schedule parse status:")
    print(df["schedule_parse_status"].value_counts(dropna=False).to_string())
    print(
        "Courses with skills:",
        f"{skill_bridge['course_id'].nunique():,}/{len(df):,}",
        f"({skill_bridge['course_id'].nunique() / len(df) * 100:.1f}%)",
    )


if __name__ == "__main__":
    main()
