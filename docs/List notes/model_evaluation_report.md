# Báo Cáo Đánh Giá Mô Hình

**Ngày:** 12/08/2026  
**Tập dữ liệu:** Tập Test (`856` dòng)  
**Nhiệm vụ:** Phân loại Spam nhị phân

---

## 1. Tổng Quan Hiệu Suất

Ba mô hình (Logistic Regression, Support Vector Machine, và Multinomial Naive Bayes) được xây dựng hoàn toàn từ đầu bằng `numpy` và được kiểm thử trên tập test độc lập.

| Mô hình | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Multinomial Naive Bayes** | **95.09%** | **100.00%** | **79.61%** | **88.65%** |
| **Linear SVM** (SGD) | 83.64% | 83.00% | 40.29% | 54.25% |
| **Logistic Regression** (SGD) | 81.89% | 74.76% | 37.38% | 49.84% |

---

## 2. Nhận Xét Chính

> 
> Mô hình đạt điểm Precision tuyệt đối (100% - không có âm tính giả nào bị phân loại nhầm), nghĩa là nó không bao giờ đánh dấu nhầm một email hợp lệ thành email rác. Điểm Recall (79.6%) cũng cao hơn đáng kể so với 2 mô hình còn lại, dẫn đến F1-Score xuất sắc đạt 0.88.

>
> **Logistic Regression & SVM đang bị underfit ở lớp thiểu số (Spam).** Với điểm Recall chỉ quanh mức 37-40%, các mô hình này gặp khó khăn trong việc nhận diện email Spam. Điều này xuất phát từ **tỷ lệ mất cân bằng lớp 3.19:1**.
> - Với Logistic Regression và SVM tiêu chuẩn được huấn luyện qua SGD, bộ tối ưu hóa tự nhiên ưu tiên dự đoán lớp đa số (Ham) để giảm thiểu tổng tổn thất (global loss).
> - **Giải pháp tiềm năng:** Cài đặt **trọng số lớp (class weights)** trong hàm loss để phạt việc phân loại sai email spam nặng hơn 3.19 lần so với email ham.

---

## 3. Chi Tiết Cài Đặt

- **Ràng buộc TF-IDF:** Các mô hình được huấn luyện trên ma trận đặc trưng được tinh gọn (7,827 đặc trưng).
- **Chiến lược tối ưu:** Logistic Regression và SVM sử dụng Stochastic Gradient Descent (SGD). Với tính thưa của TF-IDF, SGD yêu cầu tinh chỉnh siêu tham số chuyên sâu (lịch trình learning rate, số epoch cao) để hội tụ tối ưu.
- **Tại sao Naive Bayes thành công:** Các mô hình xác suất dạng đếm như Multinomial Naive Bayes rất mạnh trên các tập dữ liệu văn bản thưa và nhiều chiều. Chúng tính toán xác suất độc lập cho từng thuật ngữ và tự nhiên ít bị ảnh hưởng bởi vấn đề hội tụ hay mất cân bằng lớp hơn so với các phương pháp gradient descent không có trọng số.
