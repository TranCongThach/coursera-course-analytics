import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

OUT = Path(__file__).resolve().parents[2] / "Hoa" / "outputs" / "forecast"


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
        x=x, y=y, mode="markers", name="Khóa học",
        marker=dict(size=6, opacity=0.5, color="#2563eb"),
        customdata=np.stack([df["title"], df["Organization"].fillna("Chưa rõ"),
                             df["actual_enrolled"].round(), df["predicted_enrolled"].round()], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>Thực tế: %{customdata[2]:,.0f}<br>Dự đoán: %{customdata[3]:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines",
        name="Đường xu hướng",
        line=dict(color="#dc2626", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=xs, mode="lines", name="Dự đoán đúng hoàn toàn",
        line=dict(color="#6b7280", width=2, dash="dash"),
    ))

    ticks = [2, 3, 4, 5, 6]
    fig.update_layout(
        title=dict(text="Số người học: thực tế so với dự đoán<br>"
                        f"<sup>Độ chính xác R² = {m['r2_log10']:.3f} · Sai số MAE = {m['mae_log10']:.3f} · "
                        f"Số khóa kiểm tra: {m['n_test']:,}</sup>"),
        xaxis=dict(title="Số người học thực tế", tickvals=ticks, ticktext=[f"{10**t:,.0f}" for t in ticks]),
        yaxis=dict(title="Số người học dự đoán", tickvals=ticks, ticktext=[f"{10**t:,.0f}" for t in ticks]),
        template="plotly_white", height=650, legend=dict(orientation="h", y=-0.15),
    )
    return fig
