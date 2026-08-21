# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Tạ Đăng Đức |
| MSSV | 2A202601772 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/d2103/K4-Track2-Day21-2A202601772-TaDangDuc |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | **0.7149** | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần chạy 3 đạt `f1_score` cao nhất (0,7149) nên được chọn, đúng theo tiêu chí
của lab là F1 của lớp dương chứ không phải accuracy. Điểm đáng chú ý là lần chạy có
accuracy cao nhất lại **không** phải lần có F1 cao nhất: lần 1 dẫn đầu về accuracy
(0,8780) nhưng F1 chỉ 0,7109, thấp hơn lần 3. Điều này cho thấy accuracy gần như bão hòa
quanh 0,85 - 0,88 và không phân biệt được chất lượng thật giữa các mô hình, trong khi F1
dao động tới 0,11 giữa lần tốt nhất và lần tệ nhất. Lần 2 hạ đồng thời cả ba tham số
khiến mô hình quá yếu, F1 tụt xuống 0,6051 — dưới ngưỡng 0,65 nên sẽ bị quality gate
chặn, dù accuracy 0,8460 vẫn trông bình thường. Khi giảm `learning_rate` thì phải tăng
`n_estimators` để bù lại.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult mất cân bằng lớp: chỉ 24,8% số mẫu thuộc nhóm thu nhập trên 50K. Một mô hình
vô dụng luôn trả lời "thu nhập thấp" cho mọi đầu vào vẫn đạt accuracy 0,752 mà không bắt
được một trường hợp thu nhập cao nào. Tôi đã gặp đúng tình huống này khi cố tình hạ siêu
tham số xuống `n_estimators=5, max_depth=1`: kết quả là accuracy 0,7520 nhưng F1 bằng
0,0000, và quality gate đã chặn triển khai chính xác như thiết kế. Accuracy không phân
biệt được mô hình đó với một mô hình thật sự học được gì, còn F1 của lớp dương thì có,
vì nó là trung bình điều hòa của precision và recall tính riêng trên nhóm thiểu số. Cũng
vì vậy không được dùng `average="weighted"` hay `"macro"`: các giá trị này bị lớp đa số
chiếm 75% kéo lên cao, làm ngưỡng 0,65 mất hết ý nghĩa sàng lọc.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Không xuất được `sa-key.json` | Organization bật sẵn ràng buộc `iam.disableServiceAccountKeyCreation` | Tự cấp `roles/orgpolicy.policyAdmin` rồi `disable-enforce` ràng buộc ở cấp project |
| `import mlflow` thiếu `pkg_resources` | setuptools 84 đã gỡ module này, mlflow 2.13 vẫn dùng | Hạ về `setuptools<81` |
| `dvc pull` sẽ hỏng trên CI | `credentialpath` nằm trong `.dvc/config` được commit, nhưng `sa-key.json` bị gitignore nên runner không có file | Dùng `dvc remote modify --local` để CI tự dùng `GOOGLE_APPLICATION_CREDENTIALS` |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Gấp đôi dữ liệu chỉ làm F1 tăng 0,0205 và accuracy tăng 0,0080 — một mức
cải thiện nhỏ, đúng như dự đoán vì hai batch được chia ngẫu nhiên từ cùng một nguồn nên
có cùng phân phối và hầu như không mang thêm thông tin mới. Điều thực sự được kiểm chứng
ở Bước 3 không phải là chỉ số cao hơn, mà là dữ liệu mới đi trọn vòng từ commit đến
sản phẩm đang phục vụ mà không cần bất kỳ thao tác thủ công nào.
