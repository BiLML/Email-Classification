# KẾ HOẠCH DỰ ÁN: MÔ HÌNH PHÂN LOẠI EMAIL SPAM

## 1. Mục Tiêu & Mục Đích

*   **Vấn đề:** Phân loại tập dữ liệu email đầu vào thành hai nhóm: "Spam" (Thư rác) hoặc "Not Spam" (Ham - Thư hợp lệ).
*   **Mục tiêu:** Xây dựng một hệ thống phân loại văn bản nhị phân có độ chính xác cao.
*   **Ràng buộc nghiêm ngặt:** Không sử dụng các thư viện Học Máy cấp cao như `scikit-learn`. Toàn bộ quy trình, từ tiền xử lý dữ liệu (như TF-IDF) đến các thuật toán mô hình hóa, phải được lập trình thủ công từ đầu bằng các công thức toán học cơ bản và đại số tuyến tính (chỉ sử dụng các thư viện cốt lõi như `numpy`).

---

## 2. Dữ Liệu & Đặc Trưng

### 2.1. Nguồn Dữ Liệu
*   **Tập dữ liệu:** Sử dụng tập dữ liệu đã tải từ Kaggle được lưu trữ cục bộ.

### 2.2. Trích Xuất Đặc Trưng
Mỗi email sẽ được biểu diễn dưới dạng một vector toán học ($\mathbf{x} \in \mathbb{R}^n$), bao gồm:
*   **Tần suất từ:** Tần suất của các từ khóa nhạy cảm hoặc kích hoạt spam (ví dụ: "free", "winner", "urgent", "viagra").
*   **Tần suất ký tự:** Tỷ lệ ký tự đặc biệt (ví dụ: `$`, `!`, hoặc tỷ lệ phần trăm văn bản viết hoa toàn bộ).
*   **Đặc trưng chỉ số:** Thông tin người gửi (các tên miền nghi vấn), số lượng người nhận và độ dài của dòng tiêu đề.

---

## 3. Các Mô Hình Phân Loại

Để tuân thủ yêu cầu lập trình từ đầu bằng các công thức toán học, dự án sẽ triển khai các mô hình sau:

### 3.1. Hồi Quy Logistic (Logistic Regression)
Một mô hình tuyến tính sử dụng hàm sigmoid để xuất ra xác suất. Việc tối ưu hóa được thực hiện thông qua Gradient Descent để giảm thiểu Hàm mất mát Cross-Entropy nhị phân (Log Loss).
$$f(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

### 3.2. Máy Vector Hỗ Trợ (SVM - Support Vector Machine)
Một bộ phân loại tuyến tính tìm kiếm siêu phẳng có lề (margin) tối đa. Được cài đặt bằng Hinge Loss và Stochastic Gradient Descent (SGD) kết hợp với chuẩn hóa L2.
$$L = \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

### 3.3. Naive Bayes
Một bộ phân loại xác suất dựa trên Định lý Bayes. Cụ thể, chúng ta cài đặt Multinomial Naive Bayes thích ứng cho các giá trị đặc trưng liên tục TF-IDF, sử dụng log-xác suất để tránh lỗi tràn dưới số học (numerical underflow).
$$P(y|\mathbf{x}) \propto P(y) \prod_{i=1}^{n} P(x_i|y)$$

---

## 4. Quy Trình Chi Tiết

### Bước 1: Phân Tích Dữ Liệu Khám Phá (EDA)
*   Tải tập dữ liệu được chọn vào môi trường.
*   Thực hiện thống kê mô tả: Phân tích phân phối lớp (Spam vs. Not Spam), vẽ biểu đồ phân phối độ dài email, và xác định các từ vựng xuất hiện nhiều nhất trong mỗi lớp.

### Bước 2: Tiền Xử Lý Dữ Liệu
*   **Làm sạch văn bản:** Loại bỏ thẻ HTML, dấu câu, và từ dừng (từ phổ biến thiếu giá trị phân loại), và chuyển toàn bộ văn bản thành chữ thường.
*   **Vector hóa toán học:** Cài đặt thuật toán tùy chỉnh **TF-IDF** (Tần suất xuất hiện thuật ngữ - Tần suất nghịch đảo văn bản) để chuyển đổi văn bản thành ma trận số.
    *   *TF (Term Frequency):* Tần suất một thuật ngữ xuất hiện trong một văn bản cụ thể.
    *   *IDF (Inverse Document Frequency):* Trọng số nghịch đảo để phạt các thuật ngữ xuất hiện quá thường xuyên trên toàn bộ tập dữ liệu.
    $$IDF(t) = \log \left( \frac{N}{df_t + 1} \right)$$

### Bước 3: Chia Tập Train/Test
*   Viết hàm xáo trộn dữ liệu tùy chỉnh và cắt ma trận để chia tập dữ liệu theo tỷ lệ chuẩn (ví dụ: 80% Training - 20% Testing). Đảm bảo duy trì phân phối lớp (Phân tầng - Stratified split).

### Bước 4: Huấn Luyện Mô Hình
*   Lập trình kiến trúc (các lớp OOP) cho từng thuật toán từ đầu bằng `numpy`.
*   Cài đặt các vòng lặp tối ưu hóa sử dụng Gradient Descent/SGD để tìm trọng số tối ưu cho Logistic Regression và SVM.
*   Cài đặt các ma trận tính toán xác suất (priors và likelihoods) cho Naive Bayes.
*   Lưu trữ các tham số và trọng số mô hình đã huấn luyện thành công.

### Bước 5: Đánh Giá Mô Hình
Lập trình các hàm tùy chỉnh để tính Ma Trận Nhầm Lẫn (Confusion Matrix) và các chỉ số hiệu suất:
*   **Accuracy (Độ chính xác toàn cục):** Tổng dự đoán đúng / Tổng số mẫu.
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
*   **Precision (Độ chính xác lớp Spam):** Tỷ lệ dự đoán Spam đúng trên tổng số dự đoán là Spam.
    $$\text{Precision} = \frac{TP}{TP + FP}$$
*   **Recall (Độ gợi nhớ Spam):** Tỷ lệ Spam thực tế được nhận diện đúng.
    $$\text{Recall} = \frac{TP}{TP + FN}$$
*   **F1-Score:** Trung bình hài hòa của Precision và Recall.
    $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Bước 6: Tối Ưu Hóa & Dự Đoán (Inference)
*   **Tinh chỉnh siêu tham số:** Viết thuật toán Tìm kiếm Lưới (Grid Search) cơ bản để thử nghiệm với các giá trị như *learning rate* (cho SGD), hệ số chuẩn hóa ($\lambda$), và số lượng epoch.
*   **Quy trình dự đoán End-to-End:** Hoàn thiện quy trình dự đoán toàn diện: Nhận email mới (văn bản thô) $\rightarrow$ Làm sạch $\rightarrow$ Trích xuất vector TF-IDF $\rightarrow$ Đưa qua mô hình $\rightarrow$ Xuất kết quả dự đoán (1 = Spam, 0 = Not Spam).
