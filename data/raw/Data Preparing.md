# Báo Cáo Chuẩn Bị Dữ Liệu (Data Preparing)

---

## 1. Tổng Quan Về Bộ Dữ Liệu (Dataset Overview)

Chất lượng của dữ liệu đầu vào đóng vai trò quyết định đối với hiệu năng của mọi mô hình Machine Learning. Trong dự án này, bộ dữ liệu được sử dụng là **Email Classification Dataset** được thu thập từ Kaggle (nguồn gốc từ tập dữ liệu Enron Email Dataset).

* **Kích thước file thô:** `data/emails.csv` (~8.5 MB).
* **Tổng số dòng (Mẫu dữ liệu):** 5,728 dòng.
* **Cấu trúc cột thô:** 2 cột chính:
  1. `text`: Chuỗi văn bản thô chứa toàn bộ nội dung email (bao gồm cả tiêu đề `Subject:`).
  2. `spam`: Nhãn phân loại nhị phân ($0$ = Ham, $1$ = Spam).
* **Phân bố nhãn ban đầu:**
  - **Ham (0):** 4,360 email (~76.1%).
  - **Spam (1):** 1,368 email (~23.9%).
  - **Imbalance Ratio:** $\approx 3.19 : 1$ (Lớp Ham gấp hơn 3 lần lớp Spam).

---

## 2. Quy Trình 4 Bước Chuẩn Bị Dữ Liệu (Data Preparation Pipeline)

Quy trình chuẩn bị dữ liệu được thiết kế thành một Pipeline khép kín gồm 4 bước nối tiếp nhau, đảm bảo dữ liệu sạch, không rò rỉ thông tin (Data Leakage) và sẵn sàng cho mô hình huấn luyện:

```
[Dữ liệu thô emails.csv]
       │
       ▼
 ┌───────────┐   1. Kiểm tra cấu trúc, ô khuyết, dòng trùng, email rỗng,
 │ Bước 1:   │   2. Phân tích mẫu nhiễu (HTML, URL, email, $, !),
 │ EDA &     │   3. Thống kê độ dài văn bản, từ vựng & IQR Outliers,
 │ Quality   │   4. Trực quan hóa Dashboard 9 biểu đồ (evals/eda_plots/).
 └─────┬─────┘
       │
       ▼
 ┌───────────┐   1. Trích xuất đặc trưng gốc TRƯỚC làm sạch (excl_count, word_count),
 │ Bước 2:   │   2. Loại bỏ 33 dòng trùng lặp hoàn toàn (Dedup),
 │ Cleaning  │   3. Làm sạch văn bản: Lowercase -> Lọc HTML/URL/Email ->
 │ & Feature │      Lọc tiền tố subject: -> Lọc dấu câu/số -> Lọc Stop words & words len <= 1,
 │ Extraction│   4. Xuất file: data/processed/emails_cleaned.csv.
 └─────┬─────┘
       │
       ▼
 ┌───────────┐   1. Áp dụng Stratified Split giữ nguyên tỷ lệ nhãn (76.1% Ham / 23.9% Spam),
 │ Bước 3:   │   2. Tỷ lệ chia: Train (70%), Validation (15%), Test (15%),
 │ Split     │   3. Cố định ngẫu nhiên với seed = 42 (tính tái lập),
 │ Datasets  │   4. Xuất file: train.csv (3,986 dòng), val.csv (853 dòng), test.csv (856 dòng).
 └─────┬─────┘
       │
       ▼
 ┌───────────┐   1. Xây dựng Vocabulary (min_df = 5) CHỈ từ tập Train,
 │ Bước 4:   │   2. Tính hằng số Smooth IDF trên tập Train,
 │ TF-IDF &  │   3. Biến đổi văn bản thành ma trận TF-IDF & Chuẩn hóa L2 (L2 Normalization),
 │ Matrix    │   4. Biến đổi log1p cho excl_count và word_count,
 │ Assembly  │   5. Ghép ma trận & Xuất: train_data.npz, val_data.npz, test_data.npz, tfidf_vocab.pkl.
 └───────────┘
```

---

### Bước 1: Phân Tích Khám Phá & Đánh Giá Chất Lượng Dữ Liệu (EDA & Quality Assessment)
Thực hiện thông qua 5 script Python trong thư mục `training/train/EDA/`:
- **Đánh giá thiếu hụt dữ liệu (Missing Values):** Xác nhận 0 cell bị null/missing trong toàn bộ tập dữ liệu.
- **Phát hiện dòng trùng lặp (Duplicates):** Phát hiện 33 dòng bị trùng lặp hoàn toàn (30 dòng ở lớp Spam, 3 dòng ở lớp Ham).
- **Kiểm tra email rỗng / quá ngắn:** Phát hiện một số email quá ngắn (< 20 ký tự hoặc < 5 từ).
- **Phát hiện các mẫu nhiễu (Pattern Checks):**
  - Thẻ HTML xuất hiện ở một số mẫu.
  - Các liên kết URL (`http://`, `https://`, `www.`) và địa chỉ email xuất hiện rải rác.
  - Dấu chấm cảm `!` và từ viết hoa ALL-CAPS xuất hiện với mật độ cao vượt trội trong các email Spam.
- **Trực quan hóa EDA:** Sinh tự động 9 file biểu đồ `.png` lưu tại `evals/eda_plots/` hỗ trợ đánh giá hình dạng phân bố và tương quan đặc trưng.

### Bước 2: Làm Sạch Văn Bản & Trích Xuất Đặc Trưng (Cleaning & Feature Extraction)
Thực hiện tại file `training/train/Preprocessing/step_1_cleaning.py`:
1. **Trích xuất đặc trưng trước khi làm sạch:**
   - `excl_count`: Số lượng dấu chấm cảm `!` trong email thô.
   - `word_count`: Số lượng từ trong email thô.
   > **Lưu ý kỹ thuật:** Phải thực hiện trích xuất 2 chỉ số này *trước* khi xóa dấu câu và xóa từ dừng, nếu không thông tin đặc trưng gốc sẽ bị biến mất.
2. **Loại bỏ trùng lặp:** Loại bỏ 33 dòng trùng lặp, đưa kích thước tập dữ liệu về **5,695 dòng sạch**.
3. **Chuẩn hóa văn bản (`clean_text`):**
   - Chuyển toàn bộ về chữ thường (Lowercasing).
   - Xóa thẻ HTML bằng Regex `<[^>]+>`.
   - Xóa liên kết URL bằng Regex `http[s]?://\S+|www\.\S+`.
   - Xóa địa chỉ email bằng Regex.
   - Xóa tiền tố `subject:` xuất hiện ở đầu email.
   - Xóa toàn bộ dấu câu và chữ số, chỉ giữ lại chữ cái `a-z` và khoảng trắng.
   - Loại bỏ danh sách Stop words mở rộng (bao gồm từ dừng tiếng Anh phổ biến và các từ nhiễu xuất hiện quá nhiều như `subject`, `re`, `fw`, `please`...) và các từ có độ dài $\le 1$.

### Bước 3: Phân Chia Tập Dữ Liệu Phân Tầng (Stratified Dataset Splitting)
Thực hiện tại file `training/train/Preprocessing/step_2_split.py`:
- Do dữ liệu mất cân bằng lớp (3.19 : 1), bắt buộc áp dụng **Stratified Split** để đảm bảo cả 3 tập đều có chính xác tỷ lệ 76.1% Ham và 23.9% Spam.
- Tỷ lệ phân chia:
  - **Tập Train (70%):** 3,986 mẫu (3,052 Ham, 934 Spam) — Dùng để học từ điển, IDF và huấn luyện mô hình.
  - **Tập Validation (15%):** 853 mẫu (654 Ham, 199 Spam) — Dùng để tinh chỉnh siêu tham số và đánh giá trung gian.
  - **Tập Test (15%):** 856 mẫu (654 Ham, 202 Spam) — Tập độc lập dùng để đánh giá hiệu năng cuối cùng.
- Cố định `seed = 42` đảm bảo tính tái lập (reproducibility) 100%.

### Bước 4: Vector Hóa TF-IDF & Lắp Ráp Ma Trận Đặc Trưng (TF-IDF & Matrix Assembly)
Thực hiện tại file `training/train/Preprocessing/step_3_tfidf.py`:
1. **Xây dựng Vocabulary:**
   - Đếm tần suất xuất hiện của các từ **chỉ trên tập Train**.
   - Áp dụng ngưỡng `min_df = 5` (loại bỏ các từ xuất hiện ít hơn 5 lần) để giảm nhiễu và kiểm soát kích thước từ điển. Từ điển cuối cùng thu được kích thước khoảng 10,000+ từ.
2. **Tính Hằng Số IDF (Inverse Document Frequency):**
   - Sử dụng công thức Smooth IDF:
     $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
3. **Biến đổi TF-IDF & Chuẩn hóa L2 (L2 Normalization):**
   - Chuyển đổi văn bản của tập Train, Val, Test thành ma trận TF-IDF.
   - Chuẩn hóa L2 theo chiều ngang từng vector email để triệt tiêu ảnh hưởng của độ dài email:
     $$\mathbf{x}_{\text{normalized}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$
4. **Biến đổi log1p cho Đặc trưng Số:**
   - Áp dụng phép biến đổi $\ln(1 + x)$ (`np.log1p`) cho `excl_count` và `word_count` để nén biên độ giá trị về khoảng $[0, 1]$, đồng nhất thang đo với các giá trị TF-IDF.
5. **Ghép nối ma trận (Matrix Assembly):**
   - Ghép ma trận văn bản TF-IDF và 2 đặc trưng số bằng `np.hstack()`.
   - Lưu trữ các ma trận hoàn chỉnh dưới dạng nhị phân nén `.npz` và file pickle `.pkl` phục vụ bước huấn luyện.

---

## 3. Các Sản Phẩm Đầu Ra Của Bước Chuẩn Bị Dữ Liệu (Output Artifacts)

| Tên File Artifact | Đường Dẫn Thư Mục | Mô Tả & Nội Dung |
| :--- | :--- | :--- |
| `emails_cleaned.csv` | `data/processed/` | File CSV chứa 5,695 dòng đã làm sạch văn bản và trích xuất đặc trưng số. |
| `train.csv` | `data/processed/` | Tập dữ liệu huấn luyện dạng CSV (3,986 dòng). |
| `val.csv` | `data/processed/` | Tập dữ liệu kiểm định dạng CSV (853 dòng). |
| `test.csv` | `data/processed/` | Tập dữ liệu kiểm thử dạng CSV (856 dòng). |
| `train_data.npz` | `data/processed/` | Ma trận đặc trưng nén NumPy cho tập Train ($X$ và $y$). |
| `val_data.npz` | `data/processed/` | Ma trận đặc trưng nén NumPy cho tập Val ($X$ và $y$). |
| `test_data.npz` | `data/processed/` | Ma trận đặc trưng nén NumPy cho tập Test ($X$ và $y$). |
| `tfidf_vocab.pkl` | `data/processed/` | File lưu từ điển Vocabulary và vector trọng số IDF để phục vụ suy luận (Inference). |
| `eda_plots/*.png` | `evals/eda_plots/` | 9 file ảnh biểu đồ trực quan hóa dữ liệu từ bước EDA. |

---

## 4. Các Nguyên Tắc Đảm Bảo Tính Toàn Vẹn Dữ Liệu (Data Integrity Principles)

1. **Ngăn chặn Rò rỉ Dữ liệu (Strict Data Leakage Prevention):**
   - Bộ từ điển (Vocabulary) và trọng số IDF **chỉ được học duy nhất từ tập Train**. Tập Val và Test chỉ áp dụng bộ từ điển và IDF đã học mà không tham gia vào quá trình tính toán thống kê.
2. **Đồng nhất Pipeline Huấn luyện và Suy luận (Training-Serving Consistency):**
   - Hàm làm sạch `clean_text()` và quy trình biến đổi TF-IDF được giữ nguyên 100% khi chạy kiểm thử End-to-End (`test_e2e.py`).
3. **Triệt tiêu nhiễu thang đo (Feature Scale Normalization):**
   - Phép biến đổi `log1p` giúp cân bằng tầm ảnh hưởng giữa các đặc trưng đếm (`word_count`, `excl_count`) với ma trận ma trận thưa TF-IDF đã qua chuẩn hóa L2.