"""Bước 2: Dùng Plotly vẽ 1 biểu đồ duy nhất: thực tế vs dự báo + đường hồi quy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

OUT = Path(__file__).resolve().parents[1] / "outputs" / "forecast"


def load_results():
    df = pd.read_csv(OUT / "forecast_results.csv")
    metrics = json.loads((OUT / "forecast_metrics.json").read_text(encoding="utf-8"))
    return df, metrics


def build_figure() -> go.Figure:
    df, m = load_results()
    x, y = df["actual_log10"].to_numpy(), df["predicted_log10"].to_numpy()

    # Đường hồi quy OLS giữa dự báo và thực tế
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode="markers", name="Khóa học (tập test)",
        marker=dict(size=6, opacity=0.5, color="#2563eb"),
        customdata=np.stack([df["title"], df["Organization"].fillna("N/A"),
                             df["actual_enrolled"].round(), df["predicted_enrolled"].round()], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>Thực tế: %{customdata[2]:,.0f}<br>Dự báo: %{customdata[3]:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines",
        name=f"Đường hồi quy (y = {slope:.2f}x + {intercept:.2f})",
        line=dict(color="#dc2626", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=xs, mode="lines", name="Dự báo hoàn hảo (y = x)",
        line=dict(color="#6b7280", width=2, dash="dash"),
    ))

    ticks = [2, 3, 4, 5, 6]
    fig.update_layout(
        title=dict(text=f"Dự báo lượng đăng ký (enrollment): thực tế vs dự báo<br>"
                        f"<sup>R² = {m['r2_log10']:.3f} · MAE = {m['mae_log10']:.3f} log10 · "
                        f"n_test = {m['n_test']:,}</sup>"),
        xaxis=dict(title="Enrollment thực tế", tickvals=ticks, ticktext=[f"{10**t:,.0f}" for t in ticks]),
        yaxis=dict(title="Enrollment dự báo", tickvals=ticks, ticktext=[f"{10**t:,.0f}" for t in ticks]),
        template="plotly_white", height=650, legend=dict(orientation="h", y=-0.15),
    )
    return fig
