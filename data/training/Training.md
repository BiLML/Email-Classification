# Kế Hoạch & Ghi Chú Huấn Luyện (Training Plan)

## 1. Định Nghĩa (Define)

**Huấn luyện (Training)** trong ngữ cảnh bài toán Phân loại Email Nhị phân này bao gồm việc xây dựng các mô hình toán học nhằm học các đặc trưng ẩn phân biệt giữa email rác (Spam) và email hợp lệ (Ham). Theo các ràng buộc nghiêm ngặt của dự án, chúng ta không sử dụng các thư viện máy học cấp cao như `scikit-learn`. Thay vào đó, các mô hình tuyến tính và xác suất cốt lõi được cài đặt hoàn toàn từ đầu (from scratch) thông qua các phép toán ma trận của `numpy` và đại số tuyến tính / giải tích cơ bản.

## 2. Mục Đích (Purpose)

Các mục tiêu chính của giai đoạn huấn luyện bao gồm:
- **Cài đặt Mô hình (Model Implementation):** Lập trình từ đầu 3 mô hình riêng biệt: Logistic Regression, Support Vector Machine (SVM), và Multinomial Naive Bayes.
- **Tối ưu hóa (Optimization):** Sử dụng các kỹ thuật tối ưu hóa dựa trên độ dốc (ví dụ: Gradient Descent cho Logistic Regression và SVM) và khả năng xác suất (cho Naive Bayes) để tìm ra bộ tham số tối ưu.
- **Đánh giá (Evaluation):** Đo lường hiệu năng mô hình trên tập kiểm thử (test set) bằng các chỉ số phân loại nhị phân chuẩn (Accuracy, Precision, Recall, F1-Score).
- **So sánh (Comparison):** So sánh đặc tính hiệu năng giữa mô hình xác suất và mô hình tuyến tính, đặc biệt trong bối cảnh dữ liệu bị mất cân bằng lớp (class imbalance).

## 3. Quy Trình Thực Hiện (Workflow)

Pipeline huấn luyện được triển khai tuần tự theo các bước sau:

### Bước 1: Cài đặt Logistic Regression
- **Khái niệm:** Mô hình tuyến tính ánh xạ các dự đoán thành xác suất thông qua hàm Sigmoid.
- **Hàm mất mát (Loss Function):** Binary Cross-Entropy (Log Loss).
- **Tối ưu hóa:** Cài đặt batch Gradient Descent để cập nhật trọng số (weights) và độ lệch (bias) lặp đi lặp lại qua số lượng epoch nhất định.

### Bước 2: Cài đặt Support Vector Machine (SVM)
- **Khái niệm:** Ranh giới tuyến tính tìm cách tối đa hóa lề (margin) giữa các lớp.
- **Hàm mất mát (Loss Function):** Hinge Loss kết hợp với Chuẩn hóa L2 (L2 Regularization).
- **Tối ưu hóa:** Cài đặt Gradient Descent để cập nhật trọng số dựa trên các điểm vi phạm lề. Sử dụng nhãn $y \in \{-1, 1\}$ nội bộ.

### Bước 3: Cài đặt Multinomial Naive Bayes
- **Khái niệm:** Mô hình xác suất dựa trên Định lý Bayes. Cực kỳ hiệu quả đối với dữ liệu văn bản thưa và nhiều chiều như ma trận TF-IDF.
- **Huấn luyện:** Tính toán log-priors cho các lớp và log-likelihoods cho các đặc trưng (kèm theo kỹ thuật làm mịn Laplace - Laplace smoothing).
- **Suy luận (Inference):** Tính toán log-probabilities cho các mẫu test bằng tích vô hướng giữa ma trận đặc trưng và ma trận log-likelihood, tránh hiện tượng tràn dưới số thực (numerical underflow).

### Bước 4: Đánh Giá và Xác Nhận Mô Hình (Model Evaluation and Verification)
- Đánh giá từng mô hình trên tập kiểm thử chưa từng thấy (`test_data.npz`).
- Tính toán các chỉ số Accuracy, Precision, Recall, và F1-Score hoàn toàn từ đầu.
- Tạo báo cáo tổng hợp để đối chiếu hiệu năng giữa các mô hình.

## 4. Ghi Chú Kiểm Tra (Checknotes)

> ✅ = Đã kiểm tra và xác nhận vào ngày 2026-08-12 sau khi chạy script đánh giá.

### Logistic Regression
- [x] Đã cài đặt hàm Sigmoid kèm cơ chế chống tràn số (overflow protection).
- [x] Đã cài đặt các bước cập nhật Gradient Descent bằng kỹ thuật vectơ hóa với NumPy.
- [x] Đã huấn luyện và đánh giá mô hình. (F1-Score: ~49.84%)

### Support Vector Machine
- [x] Đã chuyển đổi nhãn sang $\{-1, 1\}$ ở xử lý nội bộ.
- [x] Đã cài đặt chính xác các điều kiện đạo hàm Hinge loss.
- [x] Đã bao gồm tham số chuẩn hóa ($\lambda$).
- [x] Đã huấn luyện và đánh giá mô hình. (F1-Score: ~54.25%)

### Multinomial Naive Bayes
- [x] Đã tính toán log-priors và log-likelihoods để tránh tràn dưới (underflow).
- [x] Đã áp dụng kỹ thuật làm mịn Laplace (Laplace smoothing).
- [x] Đã huấn luyện và đánh giá mô hình. Đạt hiệu năng vượt trội. (F1-Score: ~88.65%)

### Đánh giá (Evaluation)
- [x] Đã cài đặt thủ công tính toán các chỉ số (TP, TN, FP, FN).
- [x] Đã thực thi thành công script đánh giá `train_and_evaluate.py`.
- [x] Đã tạo và ghi nhận báo cáo hiệu năng.

---
> [!IMPORTANT]
> **Mất Cân Bằng Lớp (Class Imbalance):** Do tỷ lệ Ham : Spam là 3.19:1, các mô hình tuyến tính (Logistic Regression, SVM) gặp nhiều khó khăn ở chỉ số Recall. Mô hình xác suất Naive Bayes tỏ ra vượt trội rõ rệt ngay từ đầu đối với cấu hình tập dữ liệu này.
