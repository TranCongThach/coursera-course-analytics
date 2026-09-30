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
├── Data/
│   ├── raw/                         # Nguồn gốc, không chỉnh sửa
│   └── processed/
│       ├── fact_courses_eda_ready.csv
│       ├── bridge_course_subject.csv
│       ├── bridge_course_skill.csv
│       ├── dim_organization_country.csv
│       ├── dim_internet_usage.csv
│       └── audit_*.csv
├── outputs/eda/                     # CSV, biểu đồ và báo cáo EDA
├── scripts/
│   ├── clean_join_data.py           # 1. Làm sạch, join, subject bridge
│   ├── country.py                   # 2. Mapping quốc gia trụ sở
│   ├── org_country_mapping.PY       # Mapping thủ công
│   ├── calculated.py                # 3. Trường dẫn xuất, skill bridge
│   ├── join_internet.py             # 4. Join Internet Usage
│   ├── eda.py                       # 5. EDA tái lập được
│   └── validate_pipeline.py         # 6. Kiểm định độc lập
├── requirements.txt
├── SOURCES.md
├── DATA_DICTIONARY.md
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
python scripts\clean_join_data.py
python scripts\country.py
python scripts\calculated.py
python scripts\join_internet.py
python scripts\eda.py
python scripts\validate_pipeline.py
```

Các script tự xác định đường dẫn từ vị trí file nên không phụ thuộc current
working directory nội bộ. Matplotlib dùng backend `Agg`, không cần Tkinter và
không mở cửa sổ khi lưu ảnh.

## Bảng dữ liệu nên dùng

| File | Vai trò |
|---|---|
| `fact_courses_eda_ready.csv` | Fact canonical cuối pipeline; dùng cho Insight/Prediction |
| `fact_courses_FINAL_v2.csv` | Bản tương thích tên cũ, nội dung phải giống file canonical |
| `bridge_course_subject.csv` | Một dòng cho mỗi quan hệ course–subject |
| `bridge_course_skill.csv` | Một dòng cho mỗi quan hệ course–skill, đã dedupe hoa/thường trong từng khóa |
| `dim_organization_country.csv` | Mapping HQ thủ công kèm trạng thái cần citation |
| `dim_internet_usage.csv` | Internet Usage và năm gần nhất theo quốc gia |
| `audit_same_title_organization.csv` | Các URL khác nhau nhưng trùng title + organization; được giữ lại |
| `audit_unmapped_organizations.csv` | Tổ chức chưa map được HQ |
| `audit_unmatched_internet_countries.csv` | HQ chưa có dữ liệu Internet trong nguồn cục bộ |

## Ý nghĩa output EDA

| File | Ý nghĩa và cách dùng |
|---|---|
| `00_pipeline_validation.csv` | Kết quả PASS/FAIL của các bất biến dữ liệu và output |
| `01_data_quality.csv` | Missing, placeholder, coverage, số giá trị duy nhất, invalid và ghi chú cho mọi cột |
| `02_descriptive_statistics.csv` | Count, mean, median, độ lệch chuẩn, quartile, P95/P99, max và skewness |
| `03_numeric_distributions.png` | Phân phối enrollment, review, rating, satisfaction, hours và popularity proxy |
| `04_level_distribution.png` | Số khóa theo level; `Not specified` là dữ liệu thiếu |
| `05_subject_distribution.png` | Subject multi-label theo số khóa và coverage của nguồn phụ |
| `06_rating_by_level.png` | Boxplot rating theo level, kèm cỡ mẫu thực tế |
| `07_correlation_spearman.png` | Tương quan Spearman và số quan sát dùng cho từng cặp; không hàm ý nhân quả |
| `07_correlation_sample_size.csv` | Ma trận cỡ mẫu của tương quan để tránh đọc hệ số thiếu bối cảnh |
| `08_top_organizations.png` | Top tổ chức theo số URL khóa học |
| `09_organization_hq_countries.png` | HQ theo số khóa và số tổ chức; chỉ exploratory vì mapping thủ công |
| `10_missing_values.png` | Top 25 cột thiếu; chi tiết toàn bộ nằm trong `01_data_quality.csv` |
| `11_top_skills.png` | Top skill theo số khóa, mỗi skill chỉ được đếm một lần/khóa |
| `12_internet_hq_context.png` | Bối cảnh Internet ở HQ; không đại diện hành vi/vị trí người học |
| `EDA_REPORT.md` | Báo cáo tự sinh: coverage, insight mô tả, giới hạn và cảnh báo leakage |

## Lưu ý trước Insight và Prediction

- Subject chỉ phủ khoảng một phần tư số khóa; Difficulty/Type chỉ phủ một tập
  con rất nhỏ. Mọi kết luận phải công bố coverage.
- `popularity_score = 0.5 * rating_normalized + 0.5 *
  enrolled_percentile`; đây là proxy do nhóm định nghĩa, không phải nhãn
  popularity quan sát được.
- Dự đoán enrollment: loại `enrolled_percentile`, `popularity_score` và mọi
  biến tạo trực tiếp từ enrollment.
- Dự đoán rating: loại `rating_normalized` và `popularity_score`.
- Không có chuỗi thời gian enrollment nên không gọi bài toán là time-series
  forecasting.
- Chỉ chuyển sang Insight khi `00_pipeline_validation.csv` không có dòng FAIL.

Ý nghĩa đầy đủ của 49 cột được ghi tại `DATA_DICTIONARY.md`; nguồn và các
khoảng trống provenance được ghi tại `SOURCES.md`.
