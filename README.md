# Project cuối kỳ — Phân tích dữ liệu khóa học Coursera

Pipeline làm sạch, ghép, kiểm định và EDA dữ liệu khóa học Coursera. Bảng chính
dùng cho Insight/Prediction là `Data/processed/fact_courses_eda_ready.csv`.

## Nguyên tắc dữ liệu

- Grain của fact: **một dòng cho mỗi URL khóa học**; không xóa URL chỉ vì trùng
  tên khóa học và tổ chức.
- Subject và skill là quan hệ nhiều-nhiều, được lưu ở hai bảng bridge để không
  đếm trùng hoặc làm mất nhãn.
- Missing được giữ là missing, không điền thành `0`.
- Không sinh khóa học giả. `duration`, percentile, ratio và popularity là biến
  dẫn xuất; các giá trị duration quy đổi đều có cờ nhận diện.
- `organization_hq_country` là quốc gia trụ sở của tổ chức, **không phải quốc
  gia người học**. Mapping hiện là thủ công và cần bổ sung nguồn theo từng tổ
  chức trước khi dùng cho kết luận chính thức.
- `internet_usage_pct` là bối cảnh của quốc gia trụ sở và dùng năm gần nhất có
  trong file cục bộ; luôn đọc cùng `internet_usage_year`.

## Cấu trúc chính

```text
Project_CK_TTDLTQ/
├── Thach/                            # Xử lý/lọc dữ liệu và tích hợp giao diện
│   ├── clean_join_data.py
│   ├── country.py
│   ├── org_country_mapping.py
│   ├── calculated.py
│   ├── join_internet.py
│   ├── validate_pipeline.py
│   ├── dashboard_utils.py
│   ├── dashboard.py
│   ├── assets/
│   ├── test_dashboard.py
│   ├── dashboard_browser_smoke.mjs
│   └── outputs/
├── Bao/                              # EDA và các biểu đồ EDA
│   ├── eda.py
│   └── outputs/eda/
├── Hoa/                              # Insight và prediction
│   ├── insights.py
│   ├── prediction.py
│   ├── forecast_app/
│   ├── test_prediction.py
│   └── outputs/
├── Data/
│   ├── raw/                         # Nguồn gốc, không chỉnh sửa (không push Git)
│   └── processed/
│       ├── fact_courses_eda_ready.csv
│       ├── bridge_course_subject.csv
│       ├── bridge_course_skill.csv
│       ├── dim_organization_country.csv
│       ├── dim_internet_usage.csv
│       └── audit_*.csv
├── docs/
│   ├── DATA_DICTIONARY.md
│   └── SOURCES.md
├── requirements.txt
└── README.md
```

`idea/` hoặc `.idea/` chỉ là cấu hình IDE, không phải nơi chứa output EDA.

## Cài đặt trên Windows PowerShell

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn `Activate.ps1`, chỉ mở quyền cho phiên terminal hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Không cần đổi execution policy cho toàn máy. Lỗi
`ModuleNotFoundError: No module named 'matplotlib'` được xử lý bằng lệnh cài
`requirements.txt` trong đúng virtual environment.

## Chạy toàn bộ pipeline

Chạy từ thư mục gốc dự án:

```powershell
python -m Thach.clean_join_data
python -m Thach.country
python -m Thach.calculated
python -m Thach.join_internet
python -m Bao.eda
python -m Thach.validate_pipeline
```

Các script tự xác định đường dẫn từ vị trí file nên không phụ thuộc current
working directory nội bộ. Matplotlib dùng backend `Agg`, không cần Tkinter và
không mở cửa sổ khi lưu ảnh.

## Bảng dữ liệu nên dùng

| File                                       | Vai trò                                                                               |
| ------------------------------------------ | -------------------------------------------------------------------------------------- |
| `fact_courses_eda_ready.csv`             | Fact canonical cuối pipeline; dùng cho Insight/Prediction                            |
| `fact_courses_FINAL_v2.csv`              | Bản tương thích tên cũ, nội dung phải giống file canonical                    |
| `bridge_course_subject.csv`              | Một dòng cho mỗi quan hệ course–subject                                           |
| `bridge_course_skill.csv`                | Một dòng cho mỗi quan hệ course–skill, đã dedupe hoa/thường trong từng khóa |
| `dim_organization_country.csv`           | Mapping HQ thủ công kèm trạng thái cần citation                                  |
| `dim_internet_usage.csv`                 | Internet Usage và năm gần nhất theo quốc gia                                      |
| `audit_same_title_organization.csv`      | Các URL khác nhau nhưng trùng title + organization; được giữ lại              |
| `audit_unmapped_organizations.csv`       | Tổ chức chưa map được HQ                                                         |
| `audit_unmatched_internet_countries.csv` | HQ chưa có dữ liệu Internet trong nguồn cục bộ                                  |

## Ý nghĩa output EDA

| File                                 | Ý nghĩa và cách dùng                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `00_pipeline_validation.csv`       | Kết quả PASS/FAIL của các bất biến dữ liệu và output                               |
| `01_data_quality.csv`              | Missing, placeholder, coverage, số giá trị duy nhất, invalid và ghi chú cho mọi cột |
| `02_descriptive_statistics.csv`    | Count, mean, median, độ lệch chuẩn, quartile, P95/P99, max và skewness                 |
| `03_numeric_distributions.png`     | Phân phối enrollment, review, rating, satisfaction, hours và popularity proxy            |
| `04_level_distribution.png`        | Số khóa theo level;`Not specified` là dữ liệu thiếu                                 |
| `05_subject_distribution.png`      | Subject multi-label theo số khóa và coverage của nguồn phụ                            |
| `06_rating_by_level.png`           | Boxplot rating theo level, kèm cỡ mẫu thực tế                                          |
| `07_correlation_spearman.png`      | Tương quan Spearman và số quan sát dùng cho từng cặp; không hàm ý nhân quả     |
| `07_correlation_sample_size.csv`   | Ma trận cỡ mẫu của tương quan để tránh đọc hệ số thiếu bối cảnh             |
| `08_top_organizations.png`         | Top tổ chức theo số URL khóa học                                                       |
| `09_organization_hq_countries.png` | HQ theo số khóa và số tổ chức; chỉ exploratory vì mapping thủ công                |
| `10_missing_values.png`            | Top 25 cột thiếu; chi tiết toàn bộ nằm trong`01_data_quality.csv`                   |
| `11_top_skills.png`                | Top skill theo số khóa, mỗi skill chỉ được đếm một lần/khóa                     |
| `12_internet_hq_context.png`       | Bối cảnh Internet ở HQ; không đại diện hành vi/vị trí người học                |
| `EDA_REPORT.md`                    | Báo cáo tự sinh: coverage, insight mô tả, giới hạn và cảnh báo leakage            |

## Lưu ý trước Insight và Prediction

- Subject chỉ phủ khoảng một phần tư số khóa; Difficulty/Type chỉ phủ một tập
  con rất nhỏ. Mọi kết luận phải công bố coverage.
- `popularity_score = 0.5 * rating_normalized + 0.5 * enrolled_percentile`; đây là proxy do nhóm định nghĩa, không phải nhãn
  popularity quan sát được.
- Dự đoán enrollment: loại `enrolled_percentile`, `popularity_score` và mọi
  biến tạo trực tiếp từ enrollment.
- Dự đoán rating: loại `rating_normalized` và `popularity_score`.
- Không có chuỗi thời gian enrollment nên không gọi bài toán là time-series
  forecasting.
- Chỉ chuyển sang Insight khi `Thach/outputs/00_pipeline_validation.csv` không
  có dòng FAIL.

Ý nghĩa đầy đủ của 49 cột được ghi tại `docs/DATA_DICTIONARY.md`; nguồn và các
khoảng trống provenance được ghi tại `docs/SOURCES.md`.

## Dashboard và mô hình enrollment

Dashboard là **một ứng dụng Dash với sidebar điều hướng bốn nhóm nội dung**.
Bộ lọc nằm trong vùng nội dung
chính, ngay dưới tiêu đề; bên dưới lần lượt là KPI, biểu đồ, khối prediction và
vùng drill-down/chi tiết khóa học.

Mục tiêu hiển thị:

- Tối ưu trước cho màn hình trình chiếu Full HD `1920 × 1080` giống dashboard
  tham khảo của nhóm.
- Mười biểu đồ chính dùng lưới hai cột: bar, donut, histogram, box, scatter,
  heatmap, treemap, choropleth HQ, ECDF và xếp hạng tổ chức.
- Dưới `1280 px`, lưới biểu đồ chuyển thành một cột; trên điện thoại sidebar
  đổi thành thanh điều hướng ngang và không tràn chiều rộng.
- Thanh lọc gồm organization, level, subject, skill, rating, số giờ và
  enrollment; nút reset nằm cùng vùng điều khiển phía trên.
- Bảng khóa, course detail và ghi chú nguồn nằm trong tab `Dữ liệu`, không làm
  vỡ lưới biểu đồ của các tab phân tích.

### Trạng thái hiện tại

`Thach/dashboard.py` hiện có giao diện HCMUTE responsive, sidebar bốn mục, toolbar bộ
lọc thu gọn và các phần tổng quan/insight/dự đoán. Khi chỉnh tiếp giao diện, phải giữ
lại các quy tắc đã kiểm thử:

- Một hàm lọc dùng chung cho KPI, biểu đồ và bảng khóa.
- Subject/skill lọc qua tập `course_id`, không merge làm nhân bản fact.
- Missing không tự động đổi thành `0`; biểu đồ luôn công bố cỡ mẫu/coverage.
- Bản đồ ghi rõ là HQ tổ chức, không phải vị trí người học.
- Nền bản đồ dùng `Thach/assets/plotly-topojson/world_110m.json`, không phụ thuộc CDN
  khi chạy hoặc triển khai dashboard.
- Mô hình giữ tập train/calibration/test cố định và không huấn luyện lại theo
  bộ lọc giao diện.

### Phân công cho ba thành viên

| Thành viên     | Trách nhiệm chính                                                                                             | File/module sở hữu          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Thạch** | Tiền xử lý, join, calculated fields, kiểm định; lọc, UI/UX, callback, drill-down và tích hợp dashboard | Toàn bộ thư mục`Thach/` |
| **Bảo**   | EDA bằng Matplotlib/Seaborn; thống kê mô tả, data quality và các biểu đồ EDA                           | Toàn bộ thư mục`Bao/`   |
| **Hoa**    | Storytelling/insight có cỡ mẫu; hồi quy, đánh giá mô hình và trực quan dự báo                       | Toàn bộ thư mục`Hoa/`   |

Thạch là người duy nhất tích hợp layout/callback trong `Thach/dashboard.py` và giao
diện trong `Thach/assets/dashboard.css`. Bảo và Hoa viết logic trong module mình phụ
trách; các hàm biểu đồ nhận DataFrame đã lọc và trả về `go.Figure` để Thạch nối
vào dashboard.

### Chạy dashboard

Chạy từ thư mục gốc:

```powershell
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m Thach.dashboard
```

Mở `http://127.0.0.1:8050`. Nếu dữ liệu processed đã có, không cần chạy lại
pipeline hoặc chạy riêng `insights.py`/`prediction.py` trước dashboard.

Trong một dropdown, nhiều giá trị được kết hợp theo OR; các nhóm bộ lọc khác
nhau kết hợp theo AND. `Xóa lựa chọn` giữ bộ lọc, còn `Đặt lại` xóa cả
bộ lọc và drill-down. Phương pháp, metric và giới hạn mô hình nằm trong
`Hoa/prediction.py`.

App chỉ đọc fact và bridge trong `Data/processed`; không ghi đè dữ liệu.
Dashboard dùng `Thach/dashboard.py`, `Thach/dashboard_utils.py` và
`Thach/assets/dashboard.css`.

Kiểm tra dữ liệu, callback và hồi quy:

```powershell
python -m unittest Thach.test_dashboard Hoa.test_prediction
```

Kiểm thử trình duyệt tùy chọn (Node >= 22, Chrome/Edge cài sẵn), chạy server
ở một terminal rồi chạy kiểm thử ở terminal khác:

```powershell
python -c "from Thach.dashboard import app; app.run(port=8097, debug=False)"
# Terminal thứ hai, cùng thư mục project:
node Thach/dashboard_browser_smoke.mjs
```

Ảnh kiểm tra giao diện được lưu trong `Thach/outputs/dashboard-preview/` (không đưa
vào Git). Cần chụp và kiểm tra lại ở `1920 × 1080`, `1366 × 768` và một kích
thước di động trước khi nộp.
