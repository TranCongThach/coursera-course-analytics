# Nguồn dữ liệu và provenance

Tài liệu này phân biệt dữ liệu nguồn, mapping thủ công và biến dẫn xuất. Không
được ghi “dữ liệu thật/official” nếu chưa có URL, tác giả, phiên bản và license.

## Dữ liệu raw

| File | Vai trò | Nguồn hiện biết | Trạng thái provenance |
|---|---|---|---|
| `Data/raw/coursera_course_2024.csv` | Nguồn fact chính, 1 URL/course listing | Nhóm ghi là Kaggle | **Thiếu** URL dataset, tác giả/uploader, version, ngày tải, license và ngày snapshot |
| `Data/raw/Coursera.csv` | Bổ sung Subject, Duration, Gained Skills | Nhóm ghi là Kaggle | **Thiếu** URL dataset, tác giả/uploader, version, ngày tải và license |
| `Data/raw/coursera_course_dataset_v3.csv` | Bổ sung Difficulty, Type, URL, Duration | Nhóm ghi là Kaggle | **Thiếu** URL dataset, tác giả/uploader, version, ngày tải và license |
| `Data/raw/internet_usage.csv` | Individuals using the Internet (% population), cột 2000–2023 | World Bank indicator `IT.NET.USER.ZS`; nguồn gốc chỉ tiêu: ITU | Trang chính thức: <https://data.worldbank.org/indicator/IT.NET.USER.ZS>; license hiển thị CC BY 4.0. Cần bổ sung ngày tải file cục bộ. |

Hai file `coursera_course_2024.xlsx` và
`coursera_course_dataset_v2_no_null.csv` hiện không được pipeline sử dụng. Không
xóa chúng khỏi raw nếu chưa xác nhận với thành viên cung cấp dữ liệu.

## Mapping quốc gia trụ sở

`Thach/org_country_mapping.py` là mapping thủ công Organization → HQ country,
không phải trường có sẵn trong nguồn Coursera. Hiện mapping chưa lưu URL chứng
minh theo từng tổ chức. Vì vậy:

1. Các phân tích HQ/Internet chỉ mang tính exploratory.
2. Trước khi nộp báo cáo, thêm `mapping_source` cho từng tổ chức từ website
   chính thức hoặc hồ sơ pháp lý đáng tin cậy.
3. Không đổi tên HQ thành learner country, course country hoặc market.
4. Taiwan không có trong file World Bank cục bộ; pipeline giữ Internet Usage là
   missing và không điền tay số không có citation.

## Biến dẫn xuất, không phải dữ liệu gốc

| Biến | Quy tắc |
|---|---|
| `course_id` | `course_` + 16 ký tự đầu của SHA-256(URL) |
| `duration_weeks` | Tuần được parse trực tiếp; tháng đổi theo `1 month = 4.345 weeks` và có cờ estimated |
| `hours_to_complete` | Ưu tiên giờ báo cáo; nếu chỉ có tháng/tuần và hours/week thì tính ước lượng, có cờ |
| `review_to_enrollment_ratio` | `num_reviews / enrolled_num`, chỉ khi enrollment > 0 |
| `enrolled_percentile` | Percentile rank trên các enrollment không thiếu |
| `rating_normalized` | `rating_num / 5` |
| `popularity_score` | `0.5 * rating_normalized + 0.5 * enrolled_percentile`, chỉ khi cả hai thành phần có dữ liệu |

Không coi `popularity_score` là ground truth nếu chưa có định nghĩa nghiệp vụ và
kiểm định độ nhạy cho trọng số 50/50.
