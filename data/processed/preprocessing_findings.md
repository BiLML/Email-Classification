# ⚙️ Báo Cáo Phân Tích Kết Quả Sau Quá Trình Tiền Xử Lý (Preprocessing Output Findings)

> **Ngày thực hiện:** 12/08/2026  
> **Thư mục đầu ra:** `data/processed/`  
> **Các script thực thi:** [step_1_cleaning.py](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/Preprocessing/step_1_cleaning.py) · [step_2_split.py](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/Preprocessing/step_2_split.py) · [step_3_tfidf.py](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/Preprocessing/step_3_tfidf.py)

---

## 1. Tổng Quan Quy Trình Tiền Xử Lý (Pipeline Overview)

Dựa trên các phát hiện từ [Báo cáo EDA](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/EDA/eda_findings.md), quy trình Tiền xử lý dữ liệu được thiết kế thành một đường ống (pipeline) gồm 3 bước độc lập, tuyến tính và bảo tồn nguyên tắc chống rò rỉ dữ liệu (Data Leakage):

```
Dữ liệu thô (emails.csv)
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│ Bước 1: làm sạch văn bản & Trích xuất đặc trưng số        │
│ ➔ Loại bỏ 33 trùng lặp (dữ liệu còn 5,695 dòng)           │
│ ➔ Trích xuất excl_count, word_count                      │
│ ➔ Lọc HTML, URL, Email, Punctuation, Stop words           │
│ ➔ Đầu ra: data/processed/emails_cleaned.csv               │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Bước 2: Phân chia dữ liệu phân tầng (Stratified Split)     │
│ ➔ Xáo trộn ngẫu nhiên độc lập (Seed = 42)                 │
│ ➔ Chia tỷ lệ 70% Train / 15% Validation / 15% Test        │
│ ➔ Đầu ra: train.csv, val.csv, test.csv                     │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Bước 3: Biến đổi TF-IDF thủ công & Ghép ma trận đặc trưng  │
│ ➔ Học Vocabulary (min_df=5) và IDF CHỈ TRÊN TẬP TRAIN     │
│ ➔ Biến đổi TF-IDF L2-normalized cho cả Train/Val/Test     │
│ ➔ Chuẩn hóa log1p đặc trưng số (excl_count, word_count)   │
│ ➔ Ghép ma trận (7,825 cột TF-IDF + 2 cột đặc trưng số)    │
│ ➔ Đầu ra: train_data.npz, val_data.npz, test_data.npz    │
│            và tfidf_vocab.pkl                             │
└───────────────────────────────────────────────────────────┘
```

---

## 2. Kết Quả Bước 1: Làm Sạch Dữ Liệu & Trích Xuất Đặc Trưng (Step 1 Findings)

### 2.1 Loại Bỏ Dòng Trùng Lặp (Deduplication)
- **Số lượng dòng ban đầu:** 5,728 dòng.
- **Số dòng trùng lặp hoàn toàn:** 33 dòng (toàn bộ 33 dòng trùng đều thuộc nhãn Ham `spam=0`).
- **Số lượng dòng còn lại:** **5,695 dòng**.
- **Đánh giá:** Việc loại bỏ trùng lặp ngăn ngừa hiện tượng mô hình học thuộc lòng các mẫu trùng, đồng thời loại bỏ nguy cơ rò rỉ dữ liệu khi trùng lặp xuất hiện ở cả tập Train và Test.

### 2.2 Trích Xuất Đặc Trưng Số Kỹ Thuật (Engineered Numerical Features)
Trước khi làm sạch sâu văn bản, 2 đặc trưng số thô quan trọng được trích xuất:
1. `excl_count`: Số lượng dấu chấm cảm (`!`) có trong văn bản thô. (Có tương quan $+0.30$ với nhãn Spam).
2. `word_count`: Tổng số từ trong văn bản thô (phân tách theo khoảng trắng).

### 2.3 Chuẩn Hóa & Làm Sạch Văn Bản (Text Cleaning & Normalization)
Hàm `clean_text()` áp dụng các quy tắc tiền xử lý theo thứ tự:
- **Chuyển chữ thường (Lowercasing):** Đưa toàn bộ ký tự về dạng chữ thường.
- **Lọc thẻ HTML:** Sử dụng Regex `r'<[^>]+>'` loại bỏ các thẻ định dạng web.
- **Lọc liên kết URL:** Sử dụng Regex `r'http[s]?://\S+|www\.\S+'` loại bỏ các đường dẫn web.
- **Lọc địa chỉ Email:** Sử dụng Regex loại bỏ các chuỗi dạng `user@domain.com`.
- **Lọc Tiền Tố Dữ Liệu:** Loại bỏ tiền tố `subject:` xuất hiện ở đầu đa số email trong tập dữ liệu.
- **Lọc Dấu Câu & Chữ Số:** Sử dụng Regex `r'[^a-z\s]'` chỉ giữ lại các ký tự chữ cái tiếng Anh từ `a` đến `z` và khoảng trắng.
- **Lọc Từ Dừng (Stop Words):** Sử dụng tập hợp 65 từ dừng mở rộng (`STOP_WORDS`) bao gồm các đại từ, giới từ, từ nối phổ biến (`the`, `and`, `is`, `subject`, `re`, `fw`, `please`, `knowledge`...).
- **Lọc Từ Quá Ngắn:** Loại bỏ tất cả các từ có độ dài $\le 1$ ký tự.

### 2.4 Đánh Giá Đầu Ra Bước 1
- **File lưu trữ:** `data/processed/emails_cleaned.csv` (Dung lượng: 14.37 MB).
- **Cột dữ liệu:** `text`, `spam`, `excl_count`, `word_count`, `clean_text`.
- **Hiện tượng Email rỗng:** Sau khi làm sạch, một số ít email quá ngắn có thể trở thành chuỗi rỗng (`""`). Các email này được giữ lại vì vẫn đóng góp thông tin qua các đặc trưng số (`excl_count`, `word_count`).

---

## 3. Kết Quả Bước 2: Phân Chia Tập Dữ Liệu Phân Tầng (Step 2 Findings)

### 3.1 Cấu Trúc Phân Chia Dữ Liệu (Stratified 70 / 15 / 15 Split)
Do tập dữ liệu gốc bị sắp xếp (Spam đứng trước, Ham đứng sau), script thực hiện xáo trộn ngẫu nhiên (Shuffle) với `random_state=42` độc lập trên từng lớp trước khi thực hiện chia tập.

| Tập dữ liệu (Split) | Số dòng (Rows) | Số lượng Ham (0) | Số lượng Spam (1) | Tỷ lệ Spam (%) | Tỷ lệ tập dữ liệu |
|----------------------|----------------|------------------|-------------------|----------------|-------------------|
| **Train (Huấn luyện)** | **3,985** | 3,028 | 957 | **24.0%** | 70.0% |
| **Validation (Kiểm định)** | **854** | 649 | 205 | **24.0%** | 15.0% |
| **Test (Thử nghiệm)** | **856** | 650 | 206 | **24.1%** | 15.0% |
| **Tổng cộng** | **5,695** | **4,327** | **1,368** | **24.0%** | **100.0%** |

### 3.2 Đánh Giá Tính Phân Tầng (Stratification Verification)
- Tỷ lệ Spam ở tập dữ liệu sau deduplicate là **24.02%** ($1,368 / 5,695$).
- Tỷ lệ Spam ở cả 3 tập Train (24.0%), Validation (24.0%) và Test (24.1%) đều **khớp hoàn hảo với tỷ lệ tổng thể**.
- **File đầu ra:** `train.csv` (10.07 MB), `val.csv` (2.15 MB), `test.csv` (2.15 MB).

---

## 4. Kết Quả Bước 3: Biến Đổi TF-IDF & Ghép Ma Trận (Step 3 Findings)

### 4.1 Xây Dựng Từ Vựng (Vocabulary Construction)
- Từ vựng (Vocabulary) và chỉ số IDF **chỉ được xây dựng trên tập Train** (`train.csv`) để tránh rò rỉ dữ liệu từ tập Validation và Test.
- Áp dụng ngưỡng tần suất tài liệu tối thiểu `min_df = 5` (từ phải xuất hiện ít nhất trong 5 email tập Train).
- **Kết quả lọc từ vựng:**
  - Tổng số từ duy nhất ban đầu trong tập Train: **>25,000 từ**.
  - Số lượng từ vựng sau khi lọc (`min_df=5`): **7,825 từ**.
  - Giúp giảm tới **~70% số chiều không cần thiết**, loại bỏ nhiễu từ các từ hiếm/gõ sai.

### 4.2 Công Thức TF-IDF & Chuẩn Hóa
- **Inverse Document Frequency (IDF) có làm mượt (Smoothing):**
  $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
  Trong đó $N = 3,985$ (số tài liệu tập Train), $\text{df}(t)$ là số tài liệu chứa từ $t$.
- **Chuẩn hóa Vector L2 (L2 Normalization):**
  Mỗi vector văn bản sau khi tính TF-IDF được chia cho độ dài L2-norm của chính nó:
  $$v_{\text{norm}} = \frac{v}{\|v\|_2}$$
  Đảm bảo độ dài văn bản không làm lệch giá trị trọng số TF-IDF.

### 4.3 Xử Lý Đặc Trưng Số & Biến Đổi `log1p`
Hai đặc trưng số `excl_count` và `word_count` có dải giá trị từ $0$ đến hàng chục ngàn. Để đưa về cùng thang đo với giá trị TF-IDF (nằm trong khoảng $[0, 1]$), biến đổi $\log(1 + x)$ được áp dụng:
$$x_{\text{transformed}} = \log(1 + x)$$
Sau đó, 2 đặc trưng này được ghép nối (concatenate) vào cuối ma trận TF-IDF.

### 4.4 Kích Thước Ma Trận Đầu Ra Chi Tiết

| Tập dữ liệu | Kích thước ma trận $X$ (Rows $\times$ Cols) | Kích thước vector nhãn $y$ | Dung lượng lưu trữ `.npz` |
|-------------|-------------------------------------------|----------------------------|--------------------------|
| **Train** | `(3,985, 7,827)` | `(3,985,)` | 1.97 MB |
| **Validation** | `(854, 7827)` | `(854,)` | 0.43 MB |
| **Test** | `(856, 7827)` | `(856,)` | 0.43 MB |

> **Cấu trúc 7,827 cột đặc trưng bao gồm:**  
> - **Cột 0 đến 7,824 (7,825 cột):** Giá trị TF-IDF đại diện cho các từ trong từ vựng.  
> - **Cột 7,825:** Đặc trưng số `log1p(excl_count)`.  
> - **Cột 7,826:** Đặc trưng số `log1p(word_count)`.  
> - **Kiểu dữ liệu:** Ma trận $X$ dạng `float32`, Nhãn $y$ dạng `int8`.

### 4.5 Lưu Trữ File Đầu Ra
- `data/processed/train_data.npz` (Nén NPZ cho NumPy matrix $X$ và vector $y$).
- `data/processed/val_data.npz`.
- `data/processed/test_data.npz`.
- `data/processed/tfidf_vocab.pkl` (File Pickle chứa từ điển `vocab` 7,825 từ và mảng `idf`).

---

## 5. So Sánh Bảng Đối Chiếu Dữ Liệu Trước & Sau Tiền Xử Lý

| Tiêu chí so sánh | Dữ liệu thô ban đầu (`emails.csv`) | Dữ liệu sau Preprocessing (`.npz`) |
|------------------|------------------------------------|-----------------------------------|
| **Tổng số dòng** | 5,728 dòng | 5,695 dòng (đã trừ 33 dòng trùng) |
| **Số cột đặc trưng** | 1 cột văn bản thô (`text`) | **7,827 cột** (`float32`) |
| **Giá trị khuyết** | 0 | 0 |
| **Trùng lặp** | 33 dòng | 0 dòng (100% sạch) |
| **Thứ tự dữ liệu** | Sắp xếp (Spam trước, Ham sau) | Xáo trộn ngẫu nhiên (Seed 42) |
| **Phân chia tập** | Chưa phân chia | Train (3,985), Val (854), Test (856) |
| **Tỷ lệ lớp Spam** | 23.9% | Đồng nhất 24.0% ở cả Train/Val/Test |
| **Đặc trưng bổ sung** | Không có | `excl_count` & `word_count` ($\log(1+x)$) |
| **Lọc từ rác/từ hiếm** | Chứa HTML, URL, từ dừng, từ hiếm | Đã lọc triệt để (`min_df=5`) |

---

## 6. Đánh Giá Tính Sẵn Sàng Cho Bước Huấn Luyện Mô Hình (Readiness for Model Training)

> [!IMPORTANT]
> **Các lưu ý quan trọng khi đưa ma trận đặc trưng vào bước Huấn luyện mô hình:**
> 1. **Mô hình Cây quyết định (Decision Tree / Random Forest / XGBoost):** Ma trận `X` kích thước 7,827 cột là ma trận dày (dense format trong `.npz`). Với tập Train 3,985 dòng, ma trận chiếm khoảng ~124 MB RAM. Cần chú ý tham số `max_depth` (khuyến nghị 10–15) để tránh tràn bộ nhớ OOM và quá bớp (overfitting).
> 2. **Chỉ số đánh giá chính:** Tuyệt đối không dùng Accuracy làm thước đo chính. Sử dụng **F1-Score (Spam class)** và **Recall (Spam class)** trên tập Validation để chọn Hyperparameters và ngưỡng phân loại.
> 3. **Tương quan đặc trưng số:** 2 đặc trưng số ở vị trí cuối (`excl_count` và `word_count`) đã được chuẩn hóa Log-transform. Mô hình cây có thể trực tiếp chọn các cột này làm điểm cắt nhánh (split node) mà không bị lệch thang đo so với các cột TF-IDF.
