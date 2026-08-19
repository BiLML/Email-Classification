# Báo Cáo Định Nghĩa Bài Toán (Define Problem)

---

## 1. Bối Cảnh & Tính Cấp Thiết (Problem Context & Urgency)

Trong kỷ nguyên số hóa, email đã trở thành phương tiện giao tiếp công việc và cá nhân không thể thiếu. Tuy nhiên, sự bùng nổ của **email rác (Spam / Junk Email)** đang gây ra nhiều hệ lụy nghiêm trọng:
- **Tiêu tốn thời gian & Năng suất:** Người dùng phải tốn thời gian lọc và xóa các email quảng cáo, lừa đảo thủ công.
- **Rủi ro An ninh mạng:** Nhiều email spam chứa liên kết độc hại (Phishing URLs), mã độc (Malware), hoặc âm mưu lừa đảo tài chính.
- **Vấn đề của bộ lọc truyền thống:** Các bộ lọc dựa trên quy tắc cứng (Rule-based filters) dễ bị qua mặt khi spammer thay đổi cách viết. Ngược lại, nếu bộ lọc quá khắt khe, các email công việc quan trọng (**Ham**) dễ bị đánh nhầm thành Spam (hiện tượng **False Positive**), khiến người dùng bỏ lỡ thông tin quan trọng.

Do đó, việc xây dựng một **hệ thống phân loại email tự động bằng Machine Learning** có độ chính xác cao, khả năng tổng quát hóa tốt và giảm thiểu tối đa tỷ lệ đánh nhầm là một yêu cầu mang tính thực tiễn cao.

---

## 2. Định Nghĩa Bài Toán Machine Learning (ML Problem Formulation)

Bài toán phân loại email rác được định hình dưới dạng **Bài toán Phân loại Nhị phân có Giám sát (Supervised Binary Classification)**:

* **Tập dữ liệu huấn luyện:** $D = \{(X_1, y_1), (X_2, y_2), \dots, (X_N, y_N)\}$
* **Đầu vào ($X$):** Chuỗi văn bản thô của email (bao gồm tiêu đề và nội dung) kèm các đặc trưng số trích xuất từ văn bản.
* **Đầu ra ($y$):** Nhãn nhị phân thuộc tập $\{0, 1\}$:
  * $y = 0$: **Ham** (Email hợp lệ / công việc).
  * $y = 1$: **Spam** (Email rác / quảng cáo / lừa đảo).
* **Mục tiêu hàm toán học:** Học một hàm quyết định $f(X) \rightarrow \hat{y} \in \{0, 1\}$ hoặc ước lượng xác suất $P(y = 1 | X)$ sao cho tối thiểu hóa hàm mất mát (Loss Function) và tối đa hóa chỉ số đánh giá F1-Score trên tập dữ liệu chưa nhìn thấy (Test set).

---

## 3. Mục Tiêu Dự Án (Project Objectives)

### 3.1. Mục tiêu Kỹ thuật (Technical Objectives)
1. **Xây dựng Pipeline tự động hóa End-to-End:**
   - Xử lý dữ liệu thô từ file CSV (`emails.csv`).
   - Phân tích khám phá dữ liệu (EDA) và trực quan hóa 9 biểu đồ chuyên sâu.
   - Làm sạch văn bản, trích xuất đặc trưng số (`excl_count`, `word_count`), phân chia dữ liệu phân tầng (Stratified Split 70/15/15), và vector hóa văn bản bằng TF-IDF.
2. **Cài đặt Thuật toán Thuần NumPy (From Scratch):**
   - Tự lập trình 3 thuật toán nền tảng từ đầu theo mô hình hướng đối tượng (OOP) mà không sử dụng các thư viện Machine Learning high-level (như `scikit-learn`):
     - **Multinomial Naive Bayes (MNB):** Mô hình xác suất dựa trên định lý Bayes với Laplace Smoothing.
     - **Logistic Regression (LR):** Mô hình tuyến tính phân loại xác suất huấn luyện bằng Gradient Descent với hàm kích hoạt Sigmoid.
     - **Support Vector Machine (Linear SVM):** Mô hình tìm siêu phẳng tối đa hóa khoảng cách (Margin) huấn luyện bằng Stochastic Gradient Descent (SGD) với Hinge Loss.
3. **So sánh & Đánh giá Hiệu năng:**
   - So sánh trực quan và định lượng hiệu suất của 3 mô hình trên tập Test độc lập.
   - Đánh giá khả năng chống Overfitting và tính ổn định của từng thuật toán.

