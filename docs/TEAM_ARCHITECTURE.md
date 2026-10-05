# Kiến trúc và phân công thành viên

Tài liệu này ánh xạ source code theo ba thành viên và bốn phần bắt buộc trong
barem: tiền xử lý dữ liệu, EDA, dashboard tương tác, insight và dự báo. Đây là
phân quyền sở hữu logic; các file vẫn giữ vị trí kỹ thuật hiện tại để không làm
hỏng import và lệnh chạy của dự án.

## Sơ đồ trách nhiệm

```text
Project_CK_TTDLTQ/
├── THẠCH — Data processing, filtering, dashboard integration
│   ├── dashboard.py
│   ├── scripts/
│   │   ├── clean_join_data.py
│   │   ├── country.py
│   │   ├── org_country_mapping.py
│   │   ├── calculated.py
│   │   ├── join_internet.py
│   │   ├── validate_pipeline.py
│   │   └── dashboard_utils.py
│   ├── assets/
│   │   ├── dashboard.css
│   │   ├── Logo HCM-UTE_ 1 (1).png
│   │   └── plotly-topojson/world_110m.json
│   ├── Data/processed/
│   └── tests/
│       ├── test_dashboard.py
│       └── dashboard_browser_smoke.mjs
│
├── BẢO — EDA and EDA charts
│   ├── scripts/eda.py
│   └── outputs/eda/
│       ├── 00_pipeline_validation.csv
│       ├── 01_data_quality.csv
│       ├── 02_descriptive_statistics.csv
│       ├── 03_numeric_distributions.png
│       ├── 04_level_distribution.png
│       ├── 05_subject_distribution.png
│       ├── 06_rating_by_level.png
│       ├── 07_correlation_spearman.png
│       ├── 07_correlation_sample_size.csv
│       ├── 08_top_organizations.png
│       ├── 09_organization_hq_countries.png
│       ├── 10_missing_values.png
│       ├── 11_top_skills.png
│       ├── 12_internet_hq_context.png
│       └── EDA_REPORT.md
│
└── HOA — Insight and prediction
    ├── scripts/
    │   ├── insights.py
    │   └── prediction.py
    ├── forecast_app/
    │   ├── forecast_model.py
    │   ├── forecast_figure.py
    │   └── dash_app.py
    ├── outputs/forecast/
    │   ├── forecast_results.csv
    │   └── forecast_metrics.json
    └── tests/test_prediction.py
```

## 1. Thạch — xử lý, lọc dữ liệu và tích hợp giao diện

### Tiền xử lý dữ liệu

| File                               | Trách nhiệm                                                                                                                       |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/clean_join_data.py`     | Đọc dữ liệu thô, chuẩn hóa chuỗi/số, xử lý placeholder, tạo`course_id`, join nguồn Coursera và tạo subject bridge. |
| `scripts/country.py`             | Ánh xạ tổ chức sang quốc gia trụ sở và tạo dimension.                                                                      |
| `scripts/org_country_mapping.py` | Danh sách mapping tổ chức–quốc gia được quản lý thủ công.                                                               |
| `scripts/calculated.py`          | Parse lịch học, tạo calculated fields và skill bridge.                                                                          |
| `scripts/join_internet.py`       | Chuẩn hóa và join Internet Usage vào fact.                                                                                      |
| `scripts/validate_pipeline.py`   | Kiểm tra grain, khóa, bridge, output EDA và tính nhất quán của pipeline.                                                     |

### Lọc và tích hợp dashboard

| File                                       | Trách nhiệm                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `Thach/dashboard_utils.py`                 | Đọc fact/bridge, lọc đa cấp, giữ grain, chuẩn hóa style và gom figure từ các module để dashboard sử dụng. |
| `Thach/dashboard.py`                       | Layout, sidebar, bốn trang, KPI, callback, filter, reset, cross-filter, drill-down, bảng và chi tiết khóa học. |
| `Thach/assets/dashboard.css`               | UI/UX, font Be Vietnam Pro, lưới biểu đồ và responsive layout. |
| `Thach/assets/*.png`                       | Logo HCMUTE và bốn icon KPI khóa học, người học, điểm đánh giá, tổ chức. |
| `Thach/assets/plotly-topojson/world_110m.json` | Nền bản đồ chạy cục bộ. |
| `Thach/test_dashboard.py`                  | Kiểm thử data grain, layout, callback và filter. |
| `Thach/dashboard_browser_smoke.mjs`        | Smoke test trên trình duyệt thật, gồm desktop, laptop, 125% scaling và mobile. |

Thạch không thay đổi công thức EDA, insight hoặc mô hình do thành viên khác bàn
giao. Khi tích hợp, Thạch chỉ truyền DataFrame đã lọc vào hàm và hiển thị
`go.Figure`, KPI hoặc câu diễn giải trả về.

## 2. Bảo — EDA và các biểu đồ EDA

File nguồn chính là `scripts/eda.py`. Phạm vi gồm:

- kiểm tra missing, placeholder, invalid range và coverage;
- thống kê mô tả và phân phối biến số;
- biểu đồ Matplotlib/Seaborn;
- ghi rõ cỡ mẫu và tỷ lệ coverage;
- tạo `outputs/eda/EDA_REPORT.md`.

Các biểu đồ dashboard mang tính EDA mà Bảo bàn giao cho Thạch gồm:

1. Bar — số khóa theo trình độ;
2. Donut — coverage lịch học;
3. Histogram — phân phối enrollment;
4. Boxplot — rating theo trình độ;
5. Treemap — subject/skill;
6. Choropleth — quốc gia trụ sở tổ chức;
7. Bar ngang — top tổ chức.

Các hình tĩnh chính thức nằm trong `outputs/eda/`. File
`00_pipeline_validation.csv` được sinh bởi script của Thạch nhưng đặt cùng output
EDA để Bảo kiểm tra trước khi rút kết luận.

## 3. Hoa — insight và prediction

| File                                | Trách nhiệm                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `scripts/insights.py`             | KPI phân tích, top organization/skill/subject, coverage và câu insight động theo bộ lọc.            |
| `scripts/prediction.py`           | Hồi quy tuyến tính, chia train/calibration/test, prediction interval, trend và biểu đồ chẩn đoán. |
| `forecast_app/forecast_model.py`  | Mô hình forecast thử nghiệm độc lập và lưu metric/kết quả.                                       |
| `forecast_app/forecast_figure.py` | Trực quan Actual vs Predicted của mini-app.                                                               |
| `forecast_app/dash_app.py`        | Mini-app hiển thị forecast độc lập.                                                                    |
| `tests/test_prediction.py`        | Kiểm thử tái lập, leakage, khoảng dự đoán và input.                                                |

Các biểu đồ/phần Hoa bàn giao cho Thạch tích hợp:

1. Scatter — lượt đánh giá và số người học;
2. Heatmap — tương quan Spearman kèm cỡ mẫu;
3. ECDF — phân phối tích lũy enrollment;
4. Actual vs Predicted;
5. Residual;
6. hệ số mô hình và câu giải thích giới hạn snapshot.

Dashboard hiện tích hợp tổng cộng **14 biểu đồ**: 7 biểu đồ Tổng quan do lớp EDA
cung cấp, 4 biểu đồ Phân tích (ba biểu đồ chính và một biểu đồ so sánh tùy chọn),
và 3 biểu đồ kiểm tra mô hình ở Dự đoán người học. Biểu đồ xếp hạng tổ chức chiếm
toàn bộ hàng; các biểu đồ còn lại dùng lưới hai cột trên màn hình rộng.

Mô hình trong dashboard chính sử dụng `scripts/prediction.py`. Thư mục
`forecast_app/` là phiên bản mini-app độc lập, không được `dashboard.py` import.
Để bám đúng barem yêu cầu hồi quy tuyến tính/logistic, phần được dùng khi chấm
và demo chính là hồi quy tuyến tính trong `scripts/prediction.py`;
`forecast_app/forecast_model.py` chỉ là thử nghiệm bổ sung, không thay thế mô
hình tuyến tính chính thức.

## 4. Ranh giới file dùng chung

| File/nhóm file                | Người quyết định | Người cung cấp nội dung                                                                       |
| ------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------- |
| `Thach/dashboard.py`         | Thạch                | Bảo cung cấp figure EDA; Hoa cung cấp insight và prediction figure.                           |
| `Thach/dashboard_utils.py`   | Thạch                | Chỉ chứa lớp tích hợp/lọc; công thức chuyên môn phải thống nhất với Bảo hoặc Hoa. |
| `Data/processed/*`           | Thạch                | Bảo và Hoa chỉ đọc, không ghi đè khi chạy EDA/dashboard.                                 |
| `outputs/eda/*`              | Bảo                  | Thạch cung cấp kết quả validation.                                                            |
| `outputs/forecast/*`         | Hoa                   | Thạch chỉ đọc khi tích hợp.                                                                 |
| `docs/DATA_DICTIONARY.md`    | Thạch                | Bảo và Hoa rà soát các cột họ sử dụng.                                                   |
| `docs/SOURCES.md`            | Thạch                | Cả nhóm bổ sung citation cho phần mình trình bày.                                          |

## 5. Luồng bàn giao

```text
Thạch: raw data → clean/join/calculated → fact + bridge + validation
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
Bảo: EDA tĩnh + EDA figures             Hoa: insight + regression
                         └────────────┬────────────┘
                                      ▼
                     Thạch: filter + callback + dashboard UI
```

Nguyên tắc tích hợp:

- Một dòng fact tương ứng một `course_id`; không merge bridge trực tiếp làm nhân bản fact.
- Bảo và Hoa nhận DataFrame đã qua bộ lọc của Thạch.
- Hàm biểu đồ trả về `plotly.graph_objects.Figure`; không tự sửa layout dashboard.
- Missing không được đổi thành `0` nếu không có căn cứ nghiệp vụ.
- Insight và biểu đồ phải công bố cỡ mẫu/coverage.
- Prediction không dùng các biến dẫn xuất trực tiếp từ target để tránh leakage.
