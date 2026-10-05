from __future__ import annotations
import ast
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"

def load_country_mapping() -> dict[str, str]:
    """Nạp file mapping có đuôi .PY viết hoa theo đường dẫn tuyệt đối."""
    path = SCRIPT_DIR / "org_country_mapping.PY"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mapping_node = next(
        (
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ORG_COUNTRY_MAP"
                for target in node.targets
            )
        ),
        None,
    )
    if not isinstance(mapping_node, ast.Dict):
        raise TypeError("ORG_COUNTRY_MAP phải được khai báo trực tiếp dưới dạng dict.")
    literal_keys = [ast.literal_eval(key) for key in mapping_node.keys]
    duplicate_keys = sorted(
        key for key, count in Counter(literal_keys).items() if count > 1
    )
    if duplicate_keys:
        raise ValueError(f"ORG_COUNTRY_MAP có key trùng: {duplicate_keys}")
    module = ModuleType("org_country_mapping")
    SourceFileLoader(module.__name__, str(path)).exec_module(module)
    mapping = getattr(module, "ORG_COUNTRY_MAP", None)
    if not isinstance(mapping, dict):
        raise TypeError("ORG_COUNTRY_MAP phải là dict.")
    return mapping

def main() -> None:
    input_path = PROCESSED_DIR / "fact_courses.csv"
    fact = pd.read_csv(input_path)
    required = {"course_id", "Organization", "title"}
    missing = sorted(required.difference(fact.columns))
    if missing:
        raise ValueError(f"{input_path.name} thiếu cột: {missing}")
    mapping = load_country_mapping()
    fact = fact.drop(columns=["Country"], errors="ignore")
    fact["organization_hq_country"] = fact["Organization"].map(mapping)
    fact["organization_hq_country"] = fact["organization_hq_country"].replace(
        "UNKNOWN", pd.NA
    )
    fact["organization_hq_country_status"] = "manual_mapping_requires_citation"
    fact.loc[
        fact["Organization"].isna(), "organization_hq_country_status"
    ] = "missing_organization"
    fact.loc[
        fact["Organization"].notna() & fact["organization_hq_country"].isna(),
        "organization_hq_country_status",
    ] = "unmapped"
    dim = pd.DataFrame(
        list(mapping.items()), columns=["Organization", "organization_hq_country"]
    )
    dim["organization_hq_country"] = dim["organization_hq_country"].replace(
        "UNKNOWN", pd.NA
    )
    dim["mapping_method"] = "manual_headquarters_mapping"
    dim["mapping_verification_status"] = "requires_per_organization_citation"
    dim["mapping_source"] = pd.NA
    dim["mapping_note"] = (
        "Country represents organization headquarters, not learner location."
    )
    dim = dim.drop_duplicates("Organization").sort_values("Organization")
    unmatched = (
        fact.loc[
            fact["Organization"].notna() & fact["organization_hq_country"].isna(),
            ["Organization", "course_id"],
        ]
        .groupby("Organization", as_index=False)
        .agg(affected_courses=("course_id", "nunique"))
        .sort_values(["affected_courses", "Organization"], ascending=[False, True])
    )
    unmatched.to_csv(
        PROCESSED_DIR / "audit_unmapped_organizations.csv", index=False
    )

    if len(fact) != fact["course_id"].nunique():
        raise AssertionError("Fact phải có đúng một dòng cho mỗi course_id.")

    fact.to_csv(PROCESSED_DIR / "fact_courses_with_country.csv", index=False)
    dim.to_csv(PROCESSED_DIR / "dim_organization_country.csv", index=False)
    mapped = fact["organization_hq_country"].notna().sum()
    print(
        "Coverage organization_hq_country:",
        f"{mapped:,}/{len(fact):,} ({mapped / len(fact) * 100:.1f}%)",
    )
    print("Note: this is organization HQ country, not learner country.")

if __name__ == "__main__":
    main()
