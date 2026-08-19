# Báo Cáo Phân Tích Tiến Độ & Kiểm Tra Xác Nhận

**Ngày:** 12/08/2026  
**Mục tiêu:** Phân tích Đầu vào (Input) và Đầu ra (Output) của từng giai đoạn trong quy trình dự án để kiểm tra sự tuân thủ nghiêm ngặt theo `PLAN.md` và `WORKING_RULES.md`.

---

## 1. Giai Đoạn Phân Tích Dữ Liệu Khám Phá (EDA)

### Phân Tích Quy Trình
*   **Đầu vào (Input):** `emails.csv` (Dữ liệu thô: 5,728 dòng, 2 cột - `text`, `spam`).
*   **Xử lý (Processing):**
    *   Kiểm tra kiểu dữ liệu, giá trị thiếu (missing values) và các dòng trùng lặp (duplicates).
    *   Tính toán phân phối độ dài email (số ký tự và số từ).
    *   Trích xuất các từ xuất hiện nhiều nhất theo từng lớp.
    *   Phân tích các mẫu văn bản như ký tự đặc biệt (`$`, `!`).
*   **Đầu ra (Output):** 
    *   Tạo 8 biểu đồ trực quan hóa so sánh giữa các lớp.
    *   Tạo file `eda_step1_findings.md` tổng hợp các phát hiện chính.

### Kiểm Tra Xác Nhận Theo Kế Hoạch
*   **Đã tuân thủ kế hoạch?** ✅ **CÓ**
*   **Bằng chứng:** Giai đoạn EDA hoàn toàn đóng vai trò quan sát (không chỉnh sửa tập dữ liệu tại đây). Các chỉ số được tính toán (ví dụ: phát hiện tỷ lệ mất cân bằng lớp 3.19:1) đã trực tiếp định hướng cho giai đoạn Tiền xử lý (Phân chia tầng - Stratified split), đúng chính xác theo yêu cầu của `PLAN.md`.

---

## 2. Giai Đoạn Tiền Xử Lý (Preprocessing)

### Phân Tích Quy Trình
*   **Đầu vào (Input):** `emails.csv` (Dữ liệu thô) + Các phát hiện từ EDA.
*   **Xử lý (Processing):**
    *   **Bước 1:** Loại bỏ 33 dòng trùng lặp. Trích xuất features `excl_count` và `word_count`. Áp dụng làm sạch văn bản mạnh mẽ (chuyển chữ thường, xóa thẻ HTML, xóa dấu câu và từ dừng stop words tiếng Anh).
    *   **Bước 2:** Xáo trộn và phân chia tập dữ liệu thành Train (70%), Val (15%), Test (15%) trong khi vẫn duy trì chính xác tỷ lệ spam 24% (Stratified Split).
    *   **Bước 3:** Xây dựng thuật toán TF-IDF tùy chỉnh từ đầu bằng `numpy`. Khớp (Fit) từ vựng **chỉ trên tập Train** và loại bỏ các từ xuất hiện ít hơn 5 lần (`min_df=5`).
*   **Đầu ra (Output):** 
    *   `train.csv`, `val.csv`, `test.csv` (Các tập dữ liệu đã làm sạch).
    *   `train_data.npz`, `val_data.npz`, `test_data.npz` (Ma trận đặc trưng toán học cuối cùng).
    *   `tfidf_vocab.pkl` (Lưu từ vựng và trọng số IDF để dự đoán sau này).

### Kiểm Tra Xác Nhận Theo Kế Hoạch
*   **Đã tuân thủ kế hoạch?** ✅ **CÓ**
*   **Bằng chứng:** 
    1. **Không dùng thư viện cấp cao:** Toàn bộ được thực hiện bằng `numpy` và `pandas` (không dùng `scikit-learn`), tuân thủ nghiêm ngặt `WORKING_RULES.md`.
    2. **Không rò rỉ dữ liệu (No Data Leakage):** Từ điển TF-IDF được tạo duy nhất từ tập Training. Tập Validation và Test chỉ sử dụng hàm `transform`.
    3. **Phòng ngừa tràn bộ nhớ (OOM):** Từ vựng được cắt giảm hiệu quả từ 28k từ xuống còn ~7.8k từ.

---

## 3. Giai Đoạn Huấn Luyện Mô Hình (Modeling)

### Phân Tích Quy Trình
*   **Đầu vào (Input):** Các ma trận đặc trưng `.npz` (Kích thước: `N mẫu × 7,827 đặc trưng`).
*   **Xử lý (Processing):**
    *   Cài đặt 3 mô hình tùy chỉnh thuần toán học bằng các lớp OOP: `LogisticRegression`, `SVM`, và `MultinomialNaiveBayes`.
    *   Huấn luyện các mô hình trên tập Train.
    *   Đánh giá trên tập Test với các chỉ số Accuracy, Precision, Recall, và F1-Score.
*   **Đầu ra (Output):**
    *   Lưu trọng số mô hình: `logistic_regression.pkl`, `svm.pkl`, `naive_bayes.pkl`.
    *   Tạo `model_evaluation_report.md` chứng minh Naive Bayes vượt trội hoàn toàn so với các mô hình SGD (F1-Score: 0.88 so với ~0.50).

### Kiểm Tra Xác Nhận Theo Kế Hoạch
*   **Đã tuân thủ kế hoạch?** ✅ **CÓ** (với sự thay đổi kiến trúc đã được chấp thuận).
*   **Bằng chứng:** 
    1. Kế hoạch ban đầu yêu cầu các mô hình dạng Cây (Tree-based). Sự thay đổi sang các mô hình Tuyến tính/Xác suất đã được đề xuất, ghi nhận rõ ràng trong Implementation Plan và được chấp thuận.
    2. Các mô hình được xây dựng thành công từ đầu mà không cần `scikit-learn`.
    3. Tất cả các mô hình được đánh giá ưu tiên F1-Score và Recall, xử lý phù hợp với sự mất cân bằng lớp được phát hiện trong EDA.

---

## Kết Luận
Dự án đã xử lý thành công văn bản thô thành các vector toán học và phân loại chính xác Spam với Ham hoàn toàn bằng toán học nền tảng (tự viết từ đầu). Quy trình end-to-end hoạt động tốt, hoàn toàn mô-đun hóa và tuân thủ tỉ mỉ các ràng buộc trong `WORKING_RULES.md`.
