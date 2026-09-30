# EDA Report — Coursera Courses

## 1. Phạm vi và dữ liệu

- Bảng fact: `Data\processed\fact_courses_eda_ready.csv`.
- Quy mô: **6,645 khóa học**, **49 cột**; `course_id` duy nhất.
- Dữ liệu là snapshot, không phải dữ liệu Coursera thời gian thực.
- Fact giữ đủ **6,645 URL duy nhất** từ nguồn chính; URL là khóa định danh, không tự xóa các URL chỉ vì trùng tên khóa học.
- Không sinh dữ liệu khóa học giả. Các cột duration quy đổi, percentile và popularity là biến dẫn xuất, có quy tắc/cờ nhận diện.
- Country trong phân tích là **quốc gia trụ sở tổ chức**, không phải quốc gia người học.

## 2. Kiểm định chất lượng

- Không còn placeholder chuẩn (`not found`, `[]`) trong các cột canonical; xem `01_data_quality.csv`.
- Popularity chỉ được tính khi có cả rating và enrollment: **4,778/6,645** khóa.
- Duration tháng được đổi theo `1 tháng = 4.345 tuần`; **987** giá trị tổng giờ là ước lượng và có cờ nhận diện.
- Subject coverage: **1,654/6,645 (24.9%)**, lưu dưới dạng bridge multi-label.
- Skill coverage: **4,993/6,645 (75.1%)**, đã chuẩn hóa hoa/thường và loại trùng trong từng khóa.
- Difficulty coverage: **3.6%**; Type coverage: **3.6%**. Hai trường này chỉ đại diện tập con đã match.
- HQ country của **6,635** khóa đến từ mapping thủ công; mapping chưa có nguồn riêng cho từng tổ chức nên chỉ dùng thăm dò.

## 3. Insight mô tả được dữ liệu hỗ trợ

- Enrollment và review lệch phải mạnh; biểu đồ sử dụng log scale.
- Tương quan hạng Spearman giữa enrollment và review là **0.93**. Đây là tương quan, không chứng minh nhân quả.
- Tổ chức có nhiều khóa học nhất là **University of Colorado Boulder** với **353** khóa.
- Rating giữa các level có thể so sánh mô tả bằng boxplot, nhưng cần kiểm định/effect size trước khi kết luận khác biệt thống kê.
- Subject và skill chỉ được kết luận trong phạm vi coverage được ghi trực tiếp trên biểu đồ.

## 4. Giới hạn bắt buộc khi diễn giải

1. Không suy luận quốc gia/ngữ cảnh Internet của người học từ quốc gia trụ sở tổ chức.
2. Internet Usage dùng năm gần nhất khác nhau giữa quốc gia; phải đọc cùng `internet_usage_year`.
3. `popularity_score` là proxy 50% rating chuẩn hóa + 50% percentile enrollment, không phải nhãn popularity quan sát được; không dùng các thành phần của nó làm feature để dự đoán chính nó.
4. Dataset không có chuỗi thời gian enrollment, nên không hỗ trợ time-series forecasting đúng nghĩa.
5. Subject, Difficulty, Type và Satisfaction có missing cao; mọi insight phải công bố coverage.
6. Các giá trị thời lượng quy đổi từ tháng là ước lượng, không phải tổng giờ do Coursera báo cáo.
7. Mapping quốc gia trụ sở là thủ công và chưa có citation theo từng tổ chức; không dùng biểu đồ HQ/Internet làm kết luận chính thức trước khi xác minh.

## 5. Mức sẵn sàng cho Insight và Prediction

- Có thể bắt đầu Insight từ `fact_courses_eda_ready.csv` và hai bảng bridge.
- Insight về quốc gia/Internet chỉ ở mức exploratory cho đến khi hoàn tất nguồn mapping HQ.
- Trước Prediction phải xác định target; loại mọi feature được tính trực tiếp từ target để tránh leakage.
- Nếu dự đoán enrollment: không dùng `enrolled_percentile`, `popularity_score` hoặc biến tạo từ enrollment.
- Nếu dự đoán rating: không dùng `rating_normalized` hoặc `popularity_score`.
- Nếu dự đoán popularity: không dùng rating/enrollment và các biến chuẩn hóa của chúng như feature.

## 6. Đầu ra EDA

- `00_pipeline_validation.csv`: PASS/FAIL kiểm định độc lập; được tạo bởi `validate_pipeline.py` sau EDA.
- `01_data_quality.csv`: chất lượng đầy đủ của mọi cột.
- `02_descriptive_statistics.csv`: thống kê mô tả mở rộng.
- `03_numeric_distributions.png`: phân phối biến số.
- `04_level_distribution.png`: phân bố level.
- `05_subject_distribution.png`: Subject multi-label và coverage.
- `06_rating_by_level.png`: rating theo level kèm cỡ mẫu.
- `07_correlation_spearman.png`: Spearman và số mẫu từng cặp.
- `07_correlation_sample_size.csv`: số mẫu tương quan.
- `08_top_organizations.png`: top tổ chức theo số khóa.
- `09_organization_hq_countries.png`: HQ theo số khóa và số tổ chức.
- `10_missing_values.png`: top missing; chi tiết đầy đủ nằm trong CSV.
- `11_top_skills.png`: skill theo số khóa, đã dedupe.
- `12_internet_hq_context.png`: bối cảnh Internet tại HQ với cảnh báo diễn giải.
