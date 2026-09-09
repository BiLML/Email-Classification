# QUY TẮC LÀM VIỆC (WORKING RULES)

## 1. Quy Trình Phù Hợp
Tuân thủ quy trình phát triển tuyến tính và được xác minh kỹ lưỡng ở từng bước:
*   **Phân nhánh Git:** Nếu quản lý mã nguồn bằng Git, hãy chia công việc thành các nhánh rõ ràng: `data-prep`, `model-linear`, `model-probabilistic`, và `evaluation`.
*   **Phát triển Hướng Kiểm Thử Toán Học (TDD for Mathematics):** 
    *   Trước khi đưa tập dữ liệu lớn vào mô hình, hãy viết các trường hợp kiểm thử nhỏ với dữ liệu giả lập để xác minh độ chính xác của các hàm toán học cốt lõi (ví dụ: Gradient, Log Loss, Hinge Loss, tính toán Prior/Likelihood).
*   **Thực Thi Quy Trình Pipeline:**
    1. Dữ liệu thô $\rightarrow$ 2. Tiền xử lý & Làm sạch $\rightarrow$ 3. Vector hóa (TF-IDF) $\rightarrow$ 4. Huấn luyện $\rightarrow$ 5. Đánh giá (Val/Test) $\rightarrow$ 6. Tinh chỉnh Siêu tham số.
*   **Lưu Trữ Điểm Kiểm Soát (Checkpoints):** Lưu trọng số hoặc tham số mô hình dưới dạng các file `.pkl` hoặc `.json` sau khi huấn luyện thành công để tránh phải huấn luyện lại từ đầu khi khởi động lại.

## 2. Quy Tắc Viết Mã Nguồn
Vì dự án phụ thuộc thuần túy vào việc cài đặt toán học qua Python, mã nguồn phải được tối ưu hóa cao và dễ đọc:
*   **Vector hóa (Vectorization):** Tối đa hóa việc sử dụng các phép toán ma trận của `numpy`. Nghiêm cấm sử dụng các vòng lặp `for` lồng nhau khi tính toán gradient, khoảng cách, hoặc tổng, vì chúng sẽ tạo ra nút thắt cổ chai về hiệu năng rất nghiêm trọng.
*   **Khai Báo Kiểu Dữ Liệu (Type Hinting):** Bắt buộc khai báo kiểu cho đầu vào và đầu ra của tất cả các hàm để hỗ trợ quá trình sửa lỗi (debug) ma trận.
    *   *Ví dụ:* `def calculate_loss(y: np.ndarray) -> float:`
*   **Lập Trình Hướng Đối Tượng (OOP):** Cấu trúc các mô hình thành các Lớp (ví dụ: `class LogisticRegression`, `class SVM`, `class NaiveBayes`) với các phương thức cốt lõi như `.fit(X, y)` và `.predict(X)`.
*   **Chú Thích Chức Năng (Docstrings):** Mọi hàm tính toán phải đi kèm docstring chi tiết về đầu vào, đầu ra và các công thức toán học nền tảng được áp dụng.

## 3. Các Giới Hạn Nghiêm Ngặt (Strict Boundaries)
*   **Không Sử Dụng Thư Viện ML Cấp Cao:** Tuyệt đối không import `scikit-learn`, `xgboost`, `lightgbm`, hay bất kỳ thư viện dựng sẵn mô hình nào. Chỉ cho phép `numpy` (đại số tuyến tính), `pandas` (đọc/ghi dữ liệu), và `math`.
*   **Không Rò Rỉ Dữ Liệu (No Data Leakage):** Thuật toán TF-IDF phải được `.fit()` **chỉ trên tập Training**. Sử dụng các tham số học được (từ vựng, trọng số IDF) để `.transform()` tập Testing. Không bao giờ `.fit()` TF-IDF trên toàn bộ tập dữ liệu trước khi chia.
*   **Không Bỏ Qua Tiền Xử Lý:** Không đưa văn bản thô (chứa thẻ HTML hoặc dấu câu hỗn loạn) vào bước vector hóa mà không đi qua các hàm làm sạch trước.

## 4. Các Mẫu Lỗi Cần Tránh
Khi cài đặt thuật toán từ đầu, hãy hết sức cẩn trọng với các bẫy kỹ thuật sau:
*   **Bùng Nổ Gradients trong SGD:** Nếu `learning_rate` quá cao, trọng số sẽ dao động hoặc bùng nổ đến vô cùng trong quá trình huấn luyện Logistic Regression hoặc SVM. Luôn ngắt (clip) gradient hoặc tinh chỉnh learning rate cẩn thận.
*   **Tràn Dưới Số Học (Numerical Underflow) trong Naive Bayes:** Nhân nhiều xác suất nhỏ lại với nhau sẽ khiến biến bị tràn dưới về 0. Luôn sử dụng log-xác suất và cộng chúng lại.
*   **Tràn Bộ Nhớ (Out of Memory - OOM):** Tạo ma trận TF-IDF dày đặc với bộ từ vựng khổng lồ sẽ làm sập chương trình. Luôn lọc từ vựng TF-IDF bằng cách áp dụng tần suất xuất hiện tối thiểu (`min_df`).
*   **Lỗi Phép Toán Quảng Bá (Broadcasting Errors):** Xảy ra trong `numpy` khi nhân ma trận đặc trưng $X$ với một vector hoặc tính toán phần dư với hình dạng chiều không tương thích (ví dụ: `(N, )` so với `(N, 1)`). Luôn sử dụng `.reshape(-1, 1)` khi cần thiết để đảm bảo tính nhất quán về chiều.
*   **Underfitting do Chuẩn Hóa Quá Mức:** Đặt hình phạt $\lambda$ quá cao trong SVM sẽ khiến đường biên lề bỏ qua hoàn toàn sự phân bố dữ liệu. Tinh chỉnh $\lambda$ cẩn thận trên tập validation.
