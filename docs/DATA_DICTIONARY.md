# Data dictionary

Bảng canonical: `Data/processed/fact_courses_eda_ready.csv` — một dòng cho mỗi
URL trong `coursera_course_2024.csv`. Dùng `course_id` để nối hai bảng bridge.

## Định danh và nội dung nguồn chính

| Cột | Ý nghĩa | Cách dùng |
|---|---|---|
| `course_id` | ID ổn định tạo từ SHA-256 của `URL` | Khóa chính và khóa nối bridge |
| `title` | Tên course/listing | Dùng mô tả; không dùng một mình làm khóa |
| `Instructor` | Giảng viên từ nguồn chính | Có thể multi-value dạng text |
| `Organization` | Tổ chức cung cấp | Có 10 dòng thiếu organization |
| `Description` | Mô tả khóa | Text thô đã chuẩn hóa placeholder chính xác |
| `Modules/Courses` | Nội dung/module/courses | Text thô; có thể rất dài |
| `URL` | URL canonical từ nguồn chính | Khóa định danh course listing |

## Giá trị raw và numeric canonical

| Cột | Ý nghĩa | Cách dùng |
|---|---|---|
| `enrolled` | Enrollment dạng text gốc đã làm sạch | Chỉ để audit/hiển thị |
| `enrolled_num` | Enrollment đã parse số | Dùng cho tính toán/model |
| `rating` | Rating gốc đã làm sạch | Chỉ để audit/hiển thị |
| `rating_num` | Rating numeric, miền 0–5 | Dùng cho tính toán/model |
| `num_reviews` | Số review numeric | Missing được giữ nguyên |
| `Satisfaction Rate` | Satisfaction dạng text gốc | Chỉ để audit/hiển thị |
| `satisfaction_rate_num` | Satisfaction numeric, miền 0–100 | Coverage thấp, phải báo cỡ mẫu |
| `Skills` | Danh sách skill từ nguồn chính | Dùng bridge thay vì đếm trực tiếp |
| `Level` | Level gốc, thường có hậu tố `level` | Dùng `level_clean` cho phân tích |
| `Schedule` | Chuỗi thời lượng/lịch học gốc | Dùng các cột parse cho tính toán |

## Dữ liệu bổ sung và coverage

| Cột | Ý nghĩa | Cách dùng |
|---|---|---|
| `duration_sup1` | Duration từ `Coursera.csv` | Chỉ có ở tập match; không thay Schedule canonical |
| `gained_skills_sup1` | Skill bổ sung từ `Coursera.csv` | Đã đưa vào skill bridge |
| `Difficulty` | Difficulty từ dataset v3 | Coverage khoảng 3.6%, không khái quát toàn bộ |
| `Type` | Loại khóa từ dataset v3 | Coverage khoảng 3.6%, không khái quát toàn bộ |
| `course_url` | URL từ dataset v3 | URL phụ coverage thấp; canonical là `URL` |
| `duration_sup2` | Duration từ dataset v3 | Coverage thấp, dùng để audit |
| `subjects_combined` | Mảng JSON subject của mỗi course | Tóm tắt; phân tích bằng subject bridge |
| `subject_count` | Số subject phân biệt của course | Khớp số dòng tương ứng trong bridge |

## Quốc gia trụ sở

| Cột | Ý nghĩa | Cách dùng |
|---|---|---|
| `organization_hq_country` | Quốc gia trụ sở tổ chức theo mapping thủ công | Không phải quốc gia người học/course consumption |
| `organization_hq_country_status` | `manual_mapping_requires_citation`, `missing_organization` hoặc `unmapped` | Chặn kết luận chính thức khi mapping chưa có nguồn |

## Thời lượng đã parse

| Cột | Ý nghĩa | Cách dùng |
|---|---|---|
| `hours_to_complete_reported` | Tổng giờ được Schedule báo trực tiếp | Giá trị quan sát từ chuỗi nguồn |
| `hours_to_complete_estimated` | Tổng giờ ước lượng từ duration × hours/week | Chỉ có khi phải ước lượng |
| `hours_to_complete` | Ưu tiên reported, nếu thiếu thì dùng estimated | Luôn đọc cùng cờ estimated |
| `hours_per_week` | Số giờ/tuần parse từ Schedule | Missing nếu Schedule không nêu |
| `duration_weeks` | Số tuần báo cáo hoặc tháng × 4.345 | Luôn đọc cùng cờ estimated |
| `duration_weeks_is_estimated` | Cờ tuần được quy đổi từ tháng | NA khi không có duration |
| `hours_to_complete_is_estimated` | Cờ tổng giờ là ước lượng | NA khi không có tổng giờ |
| `schedule_parse_status` | `reported_hours_and_weeks`, `reported_hours_only`, `estimated_from_months`, `missing` | Giải thích nguồn gốc từng giá trị duration |

## Biến dẫn xuất

| Cột | Ý nghĩa | Cảnh báo |
|---|---|---|
| `level_clean` | Level bỏ hậu tố, missing → `Not specified` | Không coi `Not specified` là level thật |
| `review_to_enrollment_ratio` | reviews / enrollment khi enrollment > 0 | Không phải conversion rate |
| `enrolled_percentile` | Percentile rank của enrollment không thiếu | Leakage nếu dự đoán enrollment |
| `rating_normalized` | rating / 5 | Leakage nếu dự đoán rating |
| `popularity_score` | 50% rating normalized + 50% enrollment percentile | Proxy do nhóm định nghĩa, không phải ground truth |
| `popularity_score_status` | Lý do score complete/missing | Không có điền 0 giả cho thành phần thiếu |
| `skills_combined` | Mảng JSON skill từ nguồn chính + bổ sung | Phân tích bằng skill bridge |
| `skill_count` | Số skill phân biệt/course | Case-insensitive trong từng course |
| `courses_per_organization` | Số URL khóa của tổ chức | Dẫn xuất từ chính dataset snapshot |
| `courses_per_hq_country` | Số URL khóa tại HQ country | Phụ thuộc mapping HQ thủ công |

## Internet Usage

| Cột | Ý nghĩa | Cảnh báo |
|---|---|---|
| `internet_usage_pct` | Individuals using the Internet (% population) tại HQ country | Không đại diện người học Coursera |
| `internet_usage_year` | Năm của observation được dùng | Năm khác nhau giữa quốc gia |
| `internet_usage_source` | Tên indicator/source | Nguồn cục bộ World Bank/ITU |
| `internet_observation_rule` | Quy tắc chọn observation | Hiện là năm gần nhất có trong file 2000–2023 |

## Bảng bridge

- `bridge_course_subject.csv`: `course_id`, `title`, `Organization`, `subject`,
  `subject_normalized`. Một subject tối đa một lần/course.
- `bridge_course_skill.csv`: `course_id`, `title`, `skill`,
  `skill_normalized`, `skill_source`. `skill_source` cho biết đến từ primary,
  supplemental hoặc cả hai.

Không explode `subjects_combined`/`skills_combined` rồi đếm nếu đã có bridge;
làm như vậy dễ đếm trùng hoặc parse JSON sai.
