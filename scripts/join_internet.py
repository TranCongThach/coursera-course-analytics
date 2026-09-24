"""Bước 4: ghép Internet Usage vào quốc gia trụ sở tổ chức.

Internet Usage chỉ mô tả bối cảnh của quốc gia trụ sở tổ chức. Không được dùng
để suy luận vị trí hoặc hành vi của người học Coursera.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "Data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"

COUNTRY_NAME_FIX = {
    "South Korea": "Korea, Rep.",
    "Hong Kong": "Hong Kong SAR, China",
    "Czech Republic": "Czechia",
}

def prepare_internet_dimension() -> pd.DataFrame:
    source_path = RAW_DIR / "internet_usage.csv"
    net = pd.read_csv(
        source_path,
        na_values=["..", "N/A", "-", ""],
        keep_default_na=True,
    )
    required = {"Country Name", "Country Code"}
    missing = sorted(required.difference(net.columns))
    if missing:
        raise ValueError(f"{source_path.name} thiếu cột: {missing}")

    year_columns = sorted((c for c in net.columns if c.isdigit()), key=int)
    if not year_columns:
        raise ValueError("internet_usage.csv không có cột năm.")
    for column in year_columns:
        net[column] = pd.to_numeric(net[column], errors="coerce")

    descending = year_columns[::-1]
    net["internet_usage_pct"] = net[descending].bfill(axis=1).iloc[:, 0]

    def latest_year(row: pd.Series) -> int | pd._libs.missing.NAType:
        for year in descending:
            if pd.notna(row[year]):
                return int(year)
        return pd.NA

    net["internet_usage_year"] = net.apply(latest_year, axis=1).astype("Int64")
    net["internet_usage_source"] = "World Bank IT.NET.USER.ZS (source: ITU)"
    net["internet_observation_rule"] = "latest_available_in_local_2000_2023_file"

    dimension = net[
        [
            "Country Name",
            "Country Code",
            "internet_usage_pct",
            "internet_usage_year",
            "internet_usage_source",
            "internet_observation_rule",
        ]
    ].copy()
    if dimension["Country Name"].duplicated().any():
        raise ValueError("Country Name bị trùng trong internet_usage.csv.")
    invalid = dimension["internet_usage_pct"].notna() & ~dimension[
        "internet_usage_pct"
    ].between(0, 100)
    if invalid.any():
        raise ValueError("Internet Usage có giá trị ngoài [0, 100].")
    return dimension


def main() -> None:
    fact_path = PROCESSED_DIR / "fact_courses_final.csv"
    fact = pd.read_csv(fact_path)
    required = {"course_id", "organization_hq_country"}
    missing = sorted(required.difference(fact.columns))
    if missing:
        raise ValueError(f"{fact_path.name} thiếu cột: {missing}")

    dimension = prepare_internet_dimension()
    dimension.to_csv(PROCESSED_DIR / "dim_internet_usage.csv", index=False)

    row_count = len(fact)
    fact["_country_for_join"] = fact["organization_hq_country"].replace(
        COUNTRY_NAME_FIX
    )
    merged = fact.merge(
        dimension[
            [
                "Country Name",
                "internet_usage_pct",
                "internet_usage_year",
                "internet_usage_source",
                "internet_observation_rule",
            ]
        ],
        left_on="_country_for_join",
        right_on="Country Name",
        how="left",
        validate="many_to_one",
    ).drop(columns=["_country_for_join", "Country Name"])

    if len(merged) != row_count or merged["course_id"].nunique() != row_count:
        raise AssertionError("Số dòng/course_id thay đổi sau khi join Internet Usage.")
    invalid = merged["internet_usage_pct"].notna() & ~merged[
        "internet_usage_pct"
    ].between(0, 100)
    if invalid.any():
        raise ValueError("Internet Usage sau join có giá trị ngoài [0, 100].")

    unresolved_rows = merged.loc[
        merged["organization_hq_country"].notna()
        & merged["internet_usage_pct"].isna(),
        ["organization_hq_country", "course_id"],
    ]
    unresolved = (
        unresolved_rows.groupby("organization_hq_country", as_index=False)
        .agg(affected_courses=("course_id", "nunique"))
        .sort_values(["affected_courses", "organization_hq_country"], ascending=[False, True])
    )
    unresolved.to_csv(
        PROCESSED_DIR / "audit_unmatched_internet_countries.csv", index=False
    )

    ready_path = PROCESSED_DIR / "fact_courses_eda_ready.csv"
    compatibility_path = PROCESSED_DIR / "fact_courses_FINAL_v2.csv"
    merged.to_csv(ready_path, index=False)
    merged.to_csv(compatibility_path, index=False)

    coverage = merged["internet_usage_pct"].notna().mean() * 100
    print(f"EDA-ready: {len(merged):,} rows, {len(merged.columns):,} columns")
    print(f"Coverage Internet Usage: {coverage:.1f}%")
    print(f"Unmatched HQ countries: {len(unresolved):,}")
    print(f"Saved: {ready_path}")
    print("Warning: Internet Usage is joined by organization HQ, not learner country.")


if __name__ == "__main__":
    main()
