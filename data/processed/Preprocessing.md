# Kế Hoạch Tiền Xử Lý Dữ Liệu (Preprocessing Plan)

## 1. Định Nghĩa

**Tiền xử lý (Preprocessing)** trong ngữ cảnh của quy trình Xử lý Ngôn ngữ Tự nhiên (NLP) này bao gồm việc biến đổi văn bản email thô, không cấu trúc thành các vector đặc trưng số có cấu trúc mà các thuật toán học máy có thể hiểu được. Vì chúng ta xây dựng các mô hình từ đầu mà không dùng `scikit-learn` hay các thư viện tương tự, tất cả các phép biến đổi dữ liệu, chuyển đổi vector (như TF-IDF) và chia tập dữ liệu đều phải được xử lý thủ công bằng `numpy`, `pandas` và các thư viện chuẩn của Python.

## 2. Mục Đích

Các mục tiêu chính của giai đoạn tiền xử lý bao gồm:
- **Làm sạch dữ liệu (Data Cleaning):** Loại bỏ các dòng trùng lặp, xóa các thẻ HTML, URL, ký tự đặc biệt và số không đóng góp vào việc phân biệt spam/ham.
- **Chuẩn hóa văn bản (Text Normalization):** Chuyển văn bản thành chữ thường và loại bỏ các từ dừng (stop words) tiếng Anh phổ biến để giảm nhiễu và kích thước từ vựng.
- **Kỹ thuật trích xuất đặc trưng (Feature Engineering):** Trích xuất các tín hiệu phi văn bản được phát hiện trong EDA (ví dụ: số lượng dấu chấm cảm) để làm phong phú tập dữ liệu.
- **Chia tập dữ liệu (Data Splitting):** Chia tập dữ liệu một cách hợp lý thành các tập Train, Validation và Test bằng phương pháp phân chia tầng (stratified) để xử lý sự mất cân bằng lớp vừa phải (3.19:1).
- **Vector hóa (TF-IDF):** Chuyển đổi văn bản đã làm sạch thành ma trận Tần suất xuất hiện thuật ngữ - Tần suất nghịch đảo văn bản (TF-IDF), cẩn thận khớp (fit) từ vựng **chỉ trên tập huấn luyện** để tránh rò rỉ dữ liệu (data leakage).

## 3. Quy Trình Thực Hiện (Workflow)

Quy trình tiền xử lý sẽ được triển khai tuần tự theo các bước sau:

### Bước 1: Làm Sạch Dữ Liệu & Kỹ Thuật Đặc Trưng
- Tải file `emails.csv`.
- Loại bỏ 33 dòng trùng lặp được phát hiện trong EDA.
- Trích xuất đặc trưng: Tính toán `excl_count` (số lượng dấu chấm cảm) và `word_count` trước khi làm sạch sâu văn bản, vì các đặc trưng này có tương quan với nhãn spam.
- Làm sạch văn bản:
  - Chuyển tất cả văn bản thành chữ thường.
  - Xóa URL, thẻ HTML và địa chỉ email.
  - Xóa dấu câu và chữ số.
  - Xóa các từ dừng tiếng Anh phổ biến.
  - Xử lý các khoảng trắng thừa.

### Bước 2: Chia Tập Dữ Liệu Phân Tầng (Stratified Train/Val/Test Split)
- Xáo trộn tập dữ liệu kỹ lưỡng (vì dữ liệu ban đầu đã bị sắp xếp).
- Chia dữ liệu: **70% Train, 15% Validation, 15% Test**.
- Đảm bảo việc chia là **phân tầng (stratified)**, nghĩa là mỗi tập phân chia đều duy trì tỷ lệ ~76% Ham / 24% Spam.
- Lưu các tập dữ liệu đã chia (ví dụ: `train.csv`, `val.csv`, `test.csv`) vào thư mục lưu trữ để đảm bảo tính tái lập.

### Bước 3: Cài Đặt Bộ Vector Hóa TF-IDF Tự Viết (Từ Đầu)
- **Khái niệm:** Xây dựng một lớp vector hóa TF-IDF tùy chỉnh bằng `numpy` và dictionary (không dùng `sklearn.feature_extraction`).
- **Fit:** Xây dựng từ điển từ vựng và Tính toán Tần suất nghịch đảo văn bản (IDF) **chỉ trên tập Train**.
  - Áp dụng `min_df` (ví dụ: 2 đến 5) để loại bỏ các từ rất hiếm và giữ cho bộ nhớ từ vựng hiệu quả.
- **Transform:** Tính toán Tần suất xuất hiện thuật ngữ (TF) và tạo ma trận TF-IDF cho tập Train, Validation và Test bằng từ vựng đã fit.

### Bước 4: Lắp Ráp Ma Trận Đặc Trưng Cuối Cùng
- Kết hợp các vector TF-IDF với các đặc trưng số đã trích xuất (ví dụ: `excl_count`).
- Lưu các ma trận cuối cùng `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` ở định dạng nhị phân hiệu quả (ví dụ: `.npy` hoặc `.npz`) sẵn sàng cho giai đoạn mô hình hóa.

## 4. Ghi Chú Kiểm Tra (Checknotes)

> ✅ = Đã xác nhận vào ngày 12/08/2026 sau khi chạy các bước 1-3. Tất cả các bước đều tuân thủ chính xác kế hoạch (Không sử dụng thư viện bị cấm).

### Làm Sạch Dữ Liệu
- [x] Đã xóa thành công các dòng trùng lặp (Đã xác nhận giảm kích thước). (Đã xóa 33 trùng lặp)
- [x] Chuẩn hóa văn bản: chữ thường, không HTML, không dấu câu.
- [x] Đã xóa thành công các từ dừng (stop words) khỏi tập dữ liệu.
- [x] Các đặc trưng (`excl_count` và `word_count`) được tính toán và nối vào chính xác.

### Chia Tập Dữ Liệu
- [x] Tập dữ liệu được xáo trộn thành công.
- [x] Đã tạo các tập Train, Val, và Test. (70/15/15)
- [x] Đã xác nhận phân tầng: Kiểm tra tỷ lệ Spam/Ham trong cả 3 tập chia (đạt ~24% Spam). (Đã xác nhận: 24.0%, 24.0%, 24.1%)
- [x] Đã lưu các tập chia vào đĩa. (`train.csv`, `val.csv`, `test.csv`)

### Vector Hóa TF-IDF Tùy Chỉnh
- [x] TF-IDF tùy chỉnh được cài đặt không dùng thư viện bị cấm. (Viết hoàn toàn từ đầu bằng `numpy`)
- [x] Từ vựng được fit **CHỈ** trên tập Train (không rò rỉ dữ liệu).
- [x] Tham số `min_df` được áp dụng thành công để loại bỏ từ hiếm. (min_df=5 đã lọc 28,222 từ xuống còn 7,825 từ)
- [x] Kích thước từ vựng cuối cùng nằm trong tầm kiểm soát (không gây lỗi tràn bộ nhớ OOM).
- [x] Tập Val và Test được biến đổi (transform) thành công bằng từ vựng của tập Train (bỏ qua các từ chưa từng xuất hiện).

### Chuẩn Bị Đầu Ra Cuối Cùng
- [x] Gộp các đặc trưng số với ma trận TF-IDF. (Xếp chồng ngang văn bản và các đặc trưng số)
- [x] Xuất các ma trận cuối cùng sang định dạng `.npz` để huấn luyện.
- [x] Kiểm tra hợp lệ (Sanity check): Đảm bảo số dòng `X_train` khớp với `y_train`, v.v. (Hình dạng tập Train khớp chính xác ở 3985 dòng)

---
> [!IMPORTANT]
> **Phòng Ngừa Rò Rỉ Dữ Liệu:** Không tính toán tần suất văn bản TF-IDF (IDF) hoặc xây dựng từ vựng trên toàn bộ tập dữ liệu. Nó phải được rút ra một cách nghiêm ngặt từ tập huấn luyện.

> [!NOTE]
> **Các Thư Viện Được Phép Dùng Cho Tiền Xử Lý:** `pandas` (thao tác bảng dữ liệu), `numpy` (cho các phép toán số học và ma trận), `re` (làm sạch văn bản bằng regex), `math`. **Bị cấm:** `scikit-learn` (`train_test_split`, `TfidfVectorizer`, v.v.), `nltk`, `spacy`.
