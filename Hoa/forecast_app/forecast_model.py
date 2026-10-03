"""Bước 1: Huấn luyện mô hình dự báo enrollment và lưu kết quả ra CSV.

Bài toán: hồi quy log10(enrolled_num) từ đặc trưng của khóa học.
Dataset là snapshot (không có chuỗi thời gian) nên đây là dự báo hồi quy
cắt ngang, không phải time-series forecasting.

Loại bỏ các biến gây leakage theo README: enrolled_percentile,
popularity_score, review_to_enrollment_ratio.

Chạy:  python -m Hoa.forecast_app.forecast_model
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "processed" / "fact_courses_eda_ready.csv"
OUT = ROOT / "Hoa" / "outputs" / "forecast"

NUMERIC = [
    "num_reviews", "rating_num", "hours_to_complete", "hours_per_week",
    "duration_weeks", "skill_count", "subject_count", "courses_per_organization",
]
TARGET = "enrolled_num"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].copy()
    X["num_reviews"] = np.log1p(X["num_reviews"])  # lệch phải mạnh
    X = X.join(pd.get_dummies(df["level_clean"], prefix="level", dtype=float))
    return X


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].reset_index(drop=True)

    X = build_features(df)
    y = np.log10(df[TARGET])  # log để tránh vài khóa cực lớn chi phối

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300, random_state=42
    )  # xử lý NaN nguyên bản nên không điền 0 giả
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    result = df.loc[idx_te, ["course_id", "title", "Organization", "level_clean"]].copy()
    result["actual_log10"] = y_te.values
    result["predicted_log10"] = pred
    result["actual_enrolled"] = 10 ** result["actual_log10"]
    result["predicted_enrolled"] = 10 ** result["predicted_log10"]
    result.to_csv(OUT / "forecast_results.csv", index=False)

    metrics = {
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "r2_log10": float(r2_score(y_te, pred)),
        "mae_log10": float(mean_absolute_error(y_te, pred)),
        "rmse_log10": float(np.sqrt(mean_squared_error(y_te, pred))),
        "features": list(X.columns),
    }
    (OUT / "forecast_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
