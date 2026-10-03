"""Kiểm định độc lập dữ liệu đã xử lý và đầu ra EDA.

Chạy sau `eda.py`. Kết quả được lưu tại
`Thach/outputs/00_pipeline_validation.csv`; script trả exit code khác 0 nếu có
kiểm tra quan trọng không đạt.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "Data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
EDA_OUTPUT_DIR = PROJECT_DIR / "Bao" / "outputs" / "eda"
OUTPUT_DIR = PROJECT_DIR / "Thach" / "outputs"

PLACEHOLDERS = {
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

EXPECTED_EDA_OUTPUTS = [
    "01_data_quality.csv",
    "02_descriptive_statistics.csv",
    "03_numeric_distributions.png",
    "04_level_distribution.png",
    "05_subject_distribution.png",
    "06_rating_by_level.png",
    "07_correlation_spearman.png",
    "07_correlation_sample_size.csv",
    "08_top_organizations.png",
    "09_organization_hq_countries.png",
    "10_missing_values.png",
    "11_top_skills.png",
    "12_internet_hq_context.png",
    "EDA_REPORT.md",
]


def main() -> None:
    raw = pd.read_csv(RAW_DIR / "coursera_course_2024.csv", low_memory=False)
    fact = pd.read_csv(
        PROCESSED_DIR / "fact_courses_eda_ready.csv", low_memory=False
    )
    compatibility = pd.read_csv(
        PROCESSED_DIR / "fact_courses_FINAL_v2.csv", low_memory=False
    )
    subjects = pd.read_csv(PROCESSED_DIR / "bridge_course_subject.csv")
    skills = pd.read_csv(PROCESSED_DIR / "bridge_course_skill.csv")
    quality = pd.read_csv(EDA_OUTPUT_DIR / "01_data_quality.csv")

    checks: list[dict[str, object]] = []

    def record(
        check: str,
        passed: bool,
        expected: object,
        actual: object,
        details: str,
    ) -> None:
        checks.append(
            {
                "Check": check,
                "Status": "PASS" if passed else "FAIL",
                "Expected": expected,
                "Actual": actual,
                "Details": details,
            }
        )

    record(
        "raw_url_complete",
        bool(raw["URL"].notna().all()),
        "0 missing",
        int(raw["URL"].isna().sum()),
        "URL is the canonical course identity.",
    )
    record(
        "raw_url_unique",
        bool(raw["URL"].is_unique),
        "0 duplicates",
        int(raw["URL"].duplicated().sum()),
        "Distinct source URLs must never be collapsed by equal titles.",
    )
    record(
        "fact_row_count_matches_raw_urls",
        len(fact) == raw["URL"].nunique(),
        int(raw["URL"].nunique()),
        len(fact),
        "Confirms no source URL was lost or synthesized.",
    )
    missing_urls = set(raw["URL"]).difference(fact["URL"])
    extra_urls = set(fact["URL"]).difference(raw["URL"])
    record(
        "fact_url_set_matches_raw",
        not missing_urls and not extra_urls,
        "same URL set",
        f"missing={len(missing_urls)}; extra={len(extra_urls)}",
        "Set-level source lineage check.",
    )
    record(
        "course_id_primary_key",
        bool(fact["course_id"].notna().all() and fact["course_id"].is_unique),
        "complete and unique",
        f"missing={fact['course_id'].isna().sum()}; duplicates={fact['course_id'].duplicated().sum()}",
        "Fact grain is one row per course URL.",
    )
    expected_ids = fact["URL"].map(
        lambda value: "course_"
        + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    )
    record(
        "course_id_is_deterministic_url_hash",
        bool(expected_ids.equals(fact["course_id"])),
        "SHA-256 URL prefix",
        int(expected_ids.ne(fact["course_id"]).sum()),
        "IDs remain stable when row order changes.",
    )

    fact_ids = set(fact["course_id"])
    unknown_subject_ids = set(subjects["course_id"]).difference(fact_ids)
    unknown_skill_ids = set(skills["course_id"]).difference(fact_ids)
    record(
        "subject_bridge_foreign_keys",
        not unknown_subject_ids,
        0,
        len(unknown_subject_ids),
        "Every bridge row must resolve to the fact.",
    )
    record(
        "skill_bridge_foreign_keys",
        not unknown_skill_ids,
        0,
        len(unknown_skill_ids),
        "Every bridge row must resolve to the fact.",
    )
    duplicate_subjects = int(
        subjects.duplicated(["course_id", "subject_normalized"]).sum()
    )
    duplicate_skills = int(
        skills.duplicated(["course_id", "skill_normalized"]).sum()
    )
    record(
        "subject_bridge_unique_relations",
        duplicate_subjects == 0,
        0,
        duplicate_subjects,
        "A subject counts at most once per course.",
    )
    record(
        "skill_bridge_unique_relations",
        duplicate_skills == 0,
        0,
        duplicate_skills,
        "A case-normalized skill counts at most once per course.",
    )

    expected_subject_count = (
        fact["course_id"]
        .map(subjects.groupby("course_id").size())
        .fillna(0)
        .astype(int)
    )
    expected_skill_count = (
        fact["course_id"].map(skills.groupby("course_id").size()).fillna(0).astype(int)
    )
    subject_mismatches = int(expected_subject_count.ne(fact["subject_count"]).sum())
    skill_mismatches = int(expected_skill_count.ne(fact["skill_count"]).sum())
    record(
        "subject_count_matches_bridge",
        subject_mismatches == 0,
        0,
        subject_mismatches,
        "Fact summary and many-to-many bridge agree.",
    )
    record(
        "skill_count_matches_bridge",
        skill_mismatches == 0,
        0,
        skill_mismatches,
        "Fact summary and many-to-many bridge agree.",
    )

    range_rules = {
        "rating_num": (0, 5),
        "rating_normalized": (0, 1),
        "satisfaction_rate_num": (0, 100),
        "popularity_score": (0, 1),
        "review_to_enrollment_ratio": (0, 1),
        "internet_usage_pct": (0, 100),
    }
    invalid_range = 0
    for column, (low, high) in range_rules.items():
        invalid_range += int(
            (fact[column].notna() & ~fact[column].between(low, high)).sum()
        )
    nonnegative_columns = [
        "enrolled_num",
        "num_reviews",
        "hours_to_complete",
        "duration_weeks",
        "hours_per_week",
    ]
    invalid_nonnegative = sum(
        int((fact[column].notna() & fact[column].lt(0)).sum())
        for column in nonnegative_columns
    )
    record(
        "numeric_domains",
        invalid_range + invalid_nonnegative == 0,
        0,
        invalid_range + invalid_nonnegative,
        "Checks documented ranges and non-negative quantities.",
    )

    should_have_popularity = fact["rating_num"].notna() & fact[
        "enrolled_num"
    ].notna()
    popularity_mismatch = int(
        should_have_popularity.ne(fact["popularity_score"].notna()).sum()
    )
    record(
        "popularity_missingness_logic",
        popularity_mismatch == 0,
        0,
        popularity_mismatch,
        "Missing rating/enrollment is retained, never replaced by zero.",
    )
    expected_status = np.select(
        [
            fact["rating_num"].isna() & fact["enrolled_num"].isna(),
            fact["rating_num"].isna(),
            fact["enrolled_num"].isna(),
        ],
        ["missing_both", "missing_rating", "missing_enrollment"],
        default="complete",
    )
    status_mismatch = int((expected_status != fact["popularity_score_status"]).sum())
    record(
        "popularity_status_logic",
        status_mismatch == 0,
        0,
        status_mismatch,
        "Status explains every unavailable proxy score.",
    )

    schedule_mismatch = int(
        (
            fact["Schedule"].isna()
            != fact["schedule_parse_status"].eq("missing")
        ).sum()
    )
    unparsed = int(fact["schedule_parse_status"].eq("unparsed").sum())
    record(
        "schedule_parse_coverage",
        schedule_mismatch == 0 and unparsed == 0,
        "0 mismatches; 0 unparsed",
        f"mismatches={schedule_mismatch}; unparsed={unparsed}",
        "Every present schedule follows a documented parse rule.",
    )
    internet_pair_mismatch = int(
        fact["internet_usage_pct"].notna().ne(fact["internet_usage_year"].notna()).sum()
    )
    internet_sources = set(fact["internet_usage_source"].dropna().unique())
    record(
        "internet_value_year_pair",
        internet_pair_mismatch == 0,
        0,
        internet_pair_mismatch,
        "Never interpret Internet percentage without its observation year.",
    )
    record(
        "internet_source_is_documented_local_series",
        internet_sources == {"World Bank IT.NET.USER.ZS (source: ITU)"},
        "World Bank IT.NET.USER.ZS (source: ITU)",
        " | ".join(sorted(internet_sources)),
        "No undocumented manual Internet value is injected.",
    )

    placeholder_count = 0
    for column in fact.select_dtypes(include=["object", "string"]).columns:
        normalized = fact[column].astype("string").str.strip().str.casefold()
        placeholder_count += int(normalized.isin(PLACEHOLDERS).sum())
    record(
        "canonical_placeholders_removed",
        placeholder_count == 0,
        0,
        placeholder_count,
        "Exact placeholder tokens become NA; valid sentences are preserved.",
    )
    record(
        "eda_quality_has_no_invalid_values",
        int(quality["Invalid_Count"].sum()) == 0,
        0,
        int(quality["Invalid_Count"].sum()),
        "Independent check of EDA quality table.",
    )
    record(
        "eda_quality_has_no_placeholders",
        int(quality["Placeholder_Count"].sum()) == 0,
        0,
        int(quality["Placeholder_Count"].sum()),
        "Independent check of EDA quality table.",
    )
    record(
        "compatibility_file_matches_canonical",
        fact.equals(compatibility),
        "identical",
        "identical" if fact.equals(compatibility) else "different",
        "Legacy filename cannot silently diverge from EDA-ready data.",
    )
    missing_outputs = [
        name
        for name in EXPECTED_EDA_OUTPUTS
        if not (EDA_OUTPUT_DIR / name).is_file()
        or (EDA_OUTPUT_DIR / name).stat().st_size == 0
    ]
    record(
        "all_eda_outputs_exist",
        not missing_outputs,
        0,
        len(missing_outputs),
        ", ".join(missing_outputs) if missing_outputs else "All expected files are non-empty.",
    )

    result = pd.DataFrame(checks)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "00_pipeline_validation.csv", index=False)

    failures = result.loc[result["Status"].eq("FAIL")]
    print(f"Pipeline validation: {len(result) - len(failures)}/{len(result)} checks passed")
    print(f"Saved: {OUTPUT_DIR / '00_pipeline_validation.csv'}")
    if not failures.empty:
        raise AssertionError(
            "Pipeline validation failed: " + ", ".join(failures["Check"])
        )


if __name__ == "__main__":
    main()
