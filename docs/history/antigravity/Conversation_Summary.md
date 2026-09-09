# TỔNG HỢP LỊCH SỬ TRÒ CHUYỆN: DỰ ÁN PHÂN LOẠI EMAIL SPAM (ANTIGRAVITY)

**Ngày cập nhật:** 13/08/2026  
**Môi trường:** Antigravity AI Assistant  
**Mục tiêu:** Xây dựng hoàn chỉnh quy trình phân loại email rác từ đầu (From-Scratch) bằng Python (NumPy, Pandas, Math) không dùng thư viện ML cấp cao.

---

## 1. Phân Tích Dữ Liệu Khám Phá (EDA)
- **Yêu cầu:** Lập kế hoạch và thực hiện phân tích dữ liệu email thô (`emails.csv`).
- **Hành động:**
  - Tạo file kế hoạch EDA chi tiết (`docs/List notes/EDA.md`).
  - Xây dựng 5 script Python thực thi EDA từ nạp dữ liệu, kiểm tra chất lượng, phân phối lớp, phân tích văn bản đến trực quan hóa (`step_1` đến `step_5` trong `training/train/EDA/`).
  - Phát hiện các đặc điểm dữ liệu quan trọng: 5,728 dòng, 33 dòng trùng lặp, tỷ lệ mất cân bằng lớp 3.19:1 (76.1% Ham / 23.9% Spam).
  - Xuất 8 biểu đồ trực quan hóa và tổng hợp báo cáo phát hiện (`eda_findings.md`).

---

## 2. Tiền Xử Lý Dữ Liệu (Preprocessing)
- **Yêu cầu:** Lập kế hoạch và thực hiện làm sạch dữ liệu, chia tập dữ liệu và vector hóa TF-IDF thuần toán học.
- **Hành động:**
  - Tạo file kế hoạch tiền xử lý (`docs/List notes/Preprocessing.md`).
  - Xóa 33 dòng trùng lặp, trích xuất đặc trưng `excl_count` và `word_count`, chuẩn hóa văn bản (lowercase, xóa HTML, URL, dấu câu, stop words) (`step_1_cleaning.py`).
  - Thực hiện chia phân tầng (Stratified Split) tỷ lệ 70% Train, 15% Validation, 15% Test (`step_2_split.py`).
  - Cài đặt thuật toán TF-IDF tùy chỉnh từ đầu bằng `numpy`. Khớp (fit) từ vựng chỉ trên tập Train với `min_df=5`, rút gọn từ vựng từ 28,222 từ xuống 7,825 đặc trưng (`step_3_tfidf.py`).
  - Xuất các ma trận dữ liệu nén `.npz` và file lưu từ vựng `tfidf_vocab.pkl`.

---

## 3. Thay Đổi Kiến Trúc & Huấn Luyện Mô Hình (Modeling)
- **Yêu cầu:** Chuyển đổi kiến trúc mô hình từ dạng Cây sang các mô hình Tuyến tính & Xác suất (Logistic Regression, Support Vector Machine, Naive Bayes) và viết từ đầu.
- **Hành động:**
  - Cập nhật lại tài liệu kế hoạch và quy tắc làm việc (`Define Problem.md`, `PLAN.md`, `WORKING_RULES.md`).
  - Lập trình 3 lớp mô hình thuần toán học với `numpy`: `LogisticRegression`, `SVM` (dùng SGD), và `MultinomialNaiveBayes` (`training/train/train_model/`).
  - Xây dựng script huấn luyện và đánh giá tập trung (`train_and_evaluate.py`).
  - Kết quả đánh giá trên tập Test:
    - **Multinomial Naive Bayes:** Accuracy 95.09%, Precision 100.00%, Recall 79.61%, **F1-Score 88.65%** (Vượt trội hoàn toàn).
    - **Linear SVM:** F1-Score 54.25%.
    - **Logistic Regression:** F1-Score 49.84%.
  - Tự động lưu trữ các trọng số mô hình đã huấn luyện vào thư mục `models/` dưới dạng các file `.pkl`.

---

## 4. Kiểm Thử End-to-End & Hoàn Thiện Báo Cáo
- **Yêu cầu:** Xây dựng script kiểm thử end-to-end, tạo báo cáo phân tích tiến độ, dịch tài liệu sang Tiếng Việt và cập nhật README.
- **Hành động:**
  - Viết script kiểm thử dự đoán End-to-End (`training/test_e2e.py`) nhận văn bản email thô và đưa ra kết quả phân loại thành công.
  - Tạo báo cáo phân tích đối chiếu tiến độ với kế hoạch ban đầu (`progress_analysis.md`).
  - Dịch toàn bộ các file Markdown trong thư mục `docs/` sang Tiếng Việt.
  - Cập nhật file `README.md` tại gốc dự án và trong `docs/` mô tả chi tiết cấu trúc thư mục và 4 bước chạy dự án (không sử dụng icon).
