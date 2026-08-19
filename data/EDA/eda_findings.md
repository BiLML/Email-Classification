# 📊 Báo Cáo Tổng Hợp Kết Quả Phân Tích EDA — Phân Loại Email Spam

> **Ngày thực hiện:** 12/08/2026  
> **Tập dữ liệu:** `emails.csv` (8.9 MB) — Kaggle  
> **Các script EDA:** [step_1](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/EDA/step_1_data_loading.py) · [step_2](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/EDA/step_2_data_quality.py) · [step_3](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/EDA/step_3_class_distribution.py) · [step_4](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/EDA/step_4_text_analysis.py) · [step_5](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/training/train/EDA/step_5_visualization.py)

---

## 1. Tổng Quan Tập Dữ Liệu (Dataset Overview)

| Thuộc tính | Giá trị |
|------------|---------|
| **File dữ liệu** | `emails.csv` |
| **Dung lượng trên đĩa** | 8.9 MB |
| **Dung lượng trên bộ nhớ** | 8.86 MB |
| **Số dòng (Rows)** | **5,728** |
| **Số cột (Columns)** | **2** — `text` (object/chuỗi văn bản), `spam` (int64/nhãn nhị phân) |
| **Giá trị khuyết (Missing values)** | **0** — Hoàn thành 100% |
| **Dòng trùng lặp (Duplicates)** | **33** dòng trùng (tổng cộng 66 dòng liên quan) |
| **Số lượng văn bản duy nhất** | 5,695 |

---

## 2. Phân Tích Chất Lượng Dữ Liệu (Data Quality)

### Giá Trị Khuyết (Missing Values)
- Cột `text`: 0 giá trị khuyết (100% đầy đủ)
- Cột `spam`: 0 giá trị khuyết (100% đầy đủ)
- **Chiến lược:** Không cần xử lý khuyết dữ liệu.

### Dữ Liệu Trùng Lặp (Duplicates)
- Tìm thấy **33 dòng trùng lặp hoàn toàn** — tất cả các dòng mẫu trùng lặp đều thuộc nhãn Ham (`spam=0`).
- Tổng số dòng liên quan là 66 dòng (mỗi dòng lặp lại đúng 2 lần).
- **Chiến lược:** Loại bỏ trùng lặp trong bước Tiền xử lý (Preprocessing) trước khi chia tập train/test (theo `WORKING_RULES.md`).

### Email Rỗng Hoặc Gần Như Rỗng (Empty / Near-Empty Emails)
- Số email null: **0**
- Số email chỉ chứa khoảng trắng: **0**
- Số email < 20 ký tự: Số lượng rất nhỏ — đã đánh dấu để kiểm tra.
- Số email < 5 từ: Số lượng nhỏ — có thể chỉ chứa dòng tiêu đề (header).

### Phát Hiện Các Mô Hình Nội Dung Special (Content Pattern Issues)
- Email chứa thẻ HTML: Có xuất hiện → **cần lọc bỏ thẻ HTML trong bước Preprocessing**.
- Email chứa liên kết URL: Có xuất hiện → **cần làm sạch hoặc loại bỏ URL**.
- Email chứa địa chỉ email: Xuất hiện ở cả 2 lớp Ham và Spam.
- Email chứa ký tự `$` (tiền tệ): Xuất hiện phổ biến hơn ở lớp Spam.
- Email chứa ký tự `!!` hoặc nhiều hơn: Xuất hiện phổ biến hơn ở lớp Spam.
- Email chứa các từ VIẾT HOA HOÀN TOÀN: Xuất hiện phổ biến hơn ở lớp Spam.

### Tính Toàn Vẹn Của Nhãn (Label Integrity)
- Danh sách giá trị nhãn duy nhất: `[0, 1]` ✅
- Số dòng chứa nhãn không phải nhị phân: **0** ✅

### Thứ Tự Tập Dữ Liệu (Dataset Order)
- Tập dữ liệu **đã bị sắp xếp theo thứ tự** — tất cả email Spam xuất hiện ở phần đầu, các email Ham xuất hiện ở phần sau.
- **Bắt buộc phải xáo trộn (shuffle) dữ liệu trước khi thực hiện chia tập train/val/test**.

---

## 3. Phân Bố Lớp (Class Distribution)

| Nhãn (Label) | Ý nghĩa | Số lượng | Tỷ lệ % |
|--------------|---------|----------|---------|
| **0** | Ham (Email thường) | **4,360** | **76.1%** |
| **1** | Spam (Email rác) | **1,368** | **23.9%** |
| | **Tổng cộng** | **5,728** | **100%** |

### Đánh Giá Mức Độ Mất Cân Bằng (Imbalance Assessment)
- Tỷ lệ mất cân bằng: **3.19 : 1** (Ham so với Spam)
- Mức độ: **MẤT CÂN BẰNG TRUNG BÌNH (MODERATE IMBALANCE)**
- Chỉ số Accuracy (độ chính xác) đơn thuần sẽ **gây hiểu nhầm** — bắt buộc phải dùng **F1-Score** và **Recall** để đánh giá mô hình.
- **Bắt buộc phải dùng Stratified Split (chia tập phân tầng)** (theo `WORKING_RULES.md`).

### Sau Khi Loại Bỏ Trùng Lặp (After Deduplication)
- Số lượng Ham: ~4,327 | Số lượng Spam: ~1,368 → Tỷ lệ giữ nguyên mức ~3.16 : 1.
- Tổng số dòng sau khi loại bỏ trùng lặp: ~5,695 dòng.

---

## 4. Phân Tích Nội Dung Văn Bản (Text Content Analysis)

### Phân Bố Độ Dài Email (Email Length Distribution)

| Chỉ số Thống kê | Tổng thể | Lớp Ham | Lớp Spam |
|-----------------|----------|---------|----------|
| **Số ký tự trung bình (Mean chars)** | 1,556.8 | ~1,700 | ~1,100 |
| **Trung vị ký tự (Median chars)** | 979.0 | Cao hơn | Thấp hơn |
| **Số từ trung bình (Mean words)** | 326.8 | ~370 | ~190 |
| **Trung vị số từ (Median words)** | 210.0 | Cao hơn | Thấp hơn |
| **Ký tự nhỏ nhất (Min chars)** | 13 | — | — |
| **Ký tự lớn nhất (Max chars)** | 43,952 | — | — |
| **Số từ nhỏ nhất (Min words)** | 2 | — | — |
| **Số từ lớn nhất (Max words)** | 8,477 | — | — |

> **Nhận xét quan trọng:** Email Ham trung bình **dài gấp khoảng 2 lần** so với email Spam.

### Top 20 Từ Phổ Biến Nhất Theo Lớp (Đã loại bỏ từ dừng)

**Từ khóa phổ biến nhất ở lớp SPAM:**
| Hạng | Từ khóa | Tần suất xuất hiện |
|------|---------|-------------------|
| 1 | com | 998 |
| 2 | business | 844 |
| 3 | company | 805 |
| 4 | email | 804 |
| 5 | here | 770 |
| 6 | information | 740 |
| 7 | money | 662 |
| 8 | free | 606 |
| 9 | http | 600 |
| 10 | mail | 586 |
| 11 | now | 575 |
| 12 | 000 | 560 |
| 13 | click | 531 |
| 14 | time | 521 |
| 15 | make | 496 |
| 16 | only | 474 |
| 17 | website | 465 |
| 18 | adobe | 462 |
| 19 | over | 450 |
| 20 | software | 438 |

**Từ khóa phổ biến nhất ở lớp HAM:**
| Hạng | Từ khóa | Tần suất xuất hiện |
|------|---------|-------------------|
| 1 | enron | 13,382 |
| 2 | ect | 11,417 |
| 3 | vince | 8,531 |
| 4 | hou | 5,569 |
| 5 | 2000 | 4,935 |
| 6 | kaminski | 4,770 |
| 7 | com | 4,444 |
| 8 | 2001 | 3,060 |
| 9 | research | 2,670 |
| 10 | thanks | 2,523 |
| 11 | group | 2,255 |
| 12 | time | 2,212 |
| 13 | energy | 2,115 |
| 14 | risk | 1,984 |
| 15 | power | 1,916 |
| 16 | let | 1,822 |
| 17 | meeting | 1,772 |
| 18 | shirley | 1,679 |
| 19 | corp | 1,643 |
| 20 | edu | 1,620 |

> **Insight cốt lõi:** Các email Ham thuộc về **bộ dữ liệu Enron Email** — chiếm ưu thế bởi các thuật ngữ nội bộ công ty (enron, ect, vince, kaminski). Trong khi đó, các từ khóa Spam mang tính thương mại/quảng cáo/tiếp thị (business, free, money, click).

### Phân Tích Các Từ Khóa Kích Hoạt Spam (Spam Trigger Keywords)
- Đã phân tích 23 từ kích hoạt với tỷ lệ tần suất Spam/Ham.
- Các từ có chỉ số nhận diện Spam mạnh nhất: `free`, `click`, `money`, `offer`, `winner`.
- Các từ này xuất hiện nhiều hơn đáng kể (> 2 lần) trong Spam so với Ham.

### Phân Tích Ký Tự Đặc Biệt & Mô Hình Ký Tự (Special Character & Pattern Analysis)

| Đặc trưng | Trung bình Spam | Trung bình Ham | Tỷ lệ Spam/Ham |
|-----------|------------------|----------------|----------------|
| Tỷ lệ ký tự đặc biệt (Special char ratio) | Hơi cao hơn | Chuẩn (baseline) | ~1.0x |
| Tỷ lệ chữ viết hoa (Uppercase ratio) | Hơi cao hơn | Chuẩn (baseline) | ~1.3x |
| Tỷ lệ chữ số (Digit ratio) | Hơi cao hơn | Chuẩn (baseline) | ~1.1x |
| Số lượng dấu chấm cảm (`!`) | **Cao hơn vượt trội** | Thấp hơn | **~5.6x** |

> **Số lượng dấu chấm cảm (`!`) là tín hiệu phi văn bản mạnh nhất** giúp phân biệt Spam và Ham.

---

## 5. Thống Kê Mô Tả & Phát Hiện Ngoại Lệ (Statistical Summary & Outlier Detection)

### Thống Kê Mô Tả Các Biến

| Thống kê | `char_count` | `word_count` | `special_char_ratio` | `upper_ratio` | `excl_count` |
|----------|-------------|--------------|----------------------|---------------|--------------|
| Trung bình (Mean) | 1,556.8 | 326.8 | ~0.04 | ~0.002 | ~0.7 |
| Độ lệch chuẩn (Std) | Lớn | Lớn | Nhỏ | Nhỏ | Biến động cao |
| Nhỏ nhất (Min) | 13 | 2 | 0 | 0 | 0 |
| Lớn nhất (Max) | 43,952 | 8,477 | ~0.4 | ~0.08 | Cao |

### Phát Hiện Ngoại Lệ (Phương pháp IQR)

**Độ dài ký tự (`char_count`):**
- Hầu hết các email đều dưới ~4,000 ký tự.
- Ngoại lệ trên (Upper outliers): Xuất hiện các email >10,000 ký tự ở cả 2 lớp.
- Giá trị cực đoan: Có 1 email dài tới 43,952 ký tự.

**Độ dài số từ (`word_count`):**
- Hầu hết các email đều dưới ~800 từ.
- Ngoại lệ trên: Xuất hiện các email >2,000 từ.
- Giá trị cực đoan: Có 1 email chứa 8,477 từ.

**Chiến lược xử lý ngoại lệ:** **Giữ nguyên** — các mô hình dạng cây (Decision Tree, Random Forest, XGBoost) có khả năng chống chịu tốt tự nhiên đối với ngoại lệ. Việc giới hạn độ sâu `max_depth` sẽ giúp kiểm soát quá mức (overfitting).

### Hình Dạng Phân Bố (Distribution Shape)
- Phân bố của cả 2 lớp đều **lệch phải (right-skewed)** với phần đuôi dài gồm các email rất dài.
- Giá trị Trung vị (Median) << Giá trị Trung bình (Mean) ở cả 2 chỉ số ký tự và số từ.

---

## 6. Phân Tích Tương Quan (Correlation Insights)

| Đặc trưng (Feature) | Hệ số tương quan với nhãn Spam |
|---------------------|--------------------------------|
| **Số dấu chấm cảm (`excl_count`)** | **+0.30** (tín hiệu mạnh nhất) |
| Tỷ lệ chữ viết hoa (`upper_ratio`) | +0.09 |
| Độ dài ký tự (`char_count`) | -0.07 |
| Độ dài số từ (`word_count`) | -0.08 |
| Tỷ lệ ký tự đặc biệt (`special_char_ratio`) | -0.04 |

### Các Tương Quan Đáng Chú Ý
- `char_count` ↔ `word_count`: **0.99** (gần như tương quan hoàn hảo) → chỉ cần chọn 1 trong 2 làm đặc trưng số.
- `excl_count` ↔ `spam`: **0.30** → đặc trưng được trích xuất hữu ích nhất.
- `upper_ratio` ↔ `char_count`: **-0.31** → các email ngắn có xu hướng sở hữu tỷ lệ chữ viết hoa cao hơn.

---

## 7. Danh Mục Biểu Đồ Đã Tạo (Generated Visualizations)

Tất cả 9 biểu đồ ảnh đã được tạo và lưu tại thư mục [evals/eda_plots/](file:///d:/Machine%20Learning/máy%20học/PRE1%20-%20Email%20Classfication/evals/eda_plots/)

| STT | Tên biểu đồ | File lưu trữ |
|-----|-------------|--------------|
| 1 | Phân bố lớp (Cột + Tròn) | `step3_class_distribution.png` |
| 2 | Độ dài ký tự — Histogram | `plot2_char_length_hist.png` |
| 3 | Độ dài số từ — Histogram | `plot3_word_length_hist.png` |
| 4 | Độ dài email — Boxplot theo lớp | `plot4_length_boxplot.png` |
| 5 | Top 20 từ khóa — Lớp Spam | `plot5_top_words_spam.png` |
| 6 | Top 20 từ khóa — Lớp Ham | `plot6_top_words_ham.png` |
| 7 | Tỷ lệ ký tự đặc biệt & Viết hoa | `plot7_special_chars.png` |
| 8 | So sánh các đặc trưng (Thang Log) | `plot8_feature_comparison.png` |
| 9 | Ma trận tương quan các đặc trưng | `plot9_correlation_matrix.png` |

---

## 8. Các Quyết Định Cho Bước Tiền Xử Lý (Decisions for Preprocessing)

| Quyết định kỹ thuật | Giá trị / Thao tác | Lý do & Căn cứ thực nghiệm |
|----------------------|-------------------|----------------------------|
| **Loại bỏ trùng lặp** | Có — 33 dòng | Tránh rò rỉ dữ liệu (data leakage) và lệch mô hình |
| **Xáo trộn dữ liệu (Shuffle)** | Có | Tập dữ liệu gốc đã bị sắp xếp (Spam lên trước, Ham theo sau) |
| **Chiến lược chia tập** | Stratified 70/15/15 | Bảo toàn tỷ lệ lớp 76.0% / 24.0% ở cả 3 tập Train, Val, Test |
| **Chỉ số đánh giá chính** | F1-Score + Recall | Tỷ lệ mất cân bằng 3.16:1 khiến Accuracy trở nên sai lệch |
| **Lọc thẻ HTML** | Có | Phát hiện thẻ HTML trong một số email |
| **Loại bỏ từ dừng (Stop words)** | Có | Từ dừng chiếm ưu thế trong tần suất từ thô |
| **Làm sạch dấu câu** | Có | Ký tự đặc biệt và dấu `!` có sự khác biệt rõ rệt giữa 2 lớp |
| **Loại bỏ URL & Email** | Có | Phát hiện nhiều URL và địa chỉ email; không hỗ trợ phân nhánh cây tốt |
| **Chuyển chữ thường (Lowercase)** | Có | Chuẩn hóa văn bản NLP tiêu chuẩn |
| **TF-IDF min_df** | 5 | Loại bỏ từ hiếm, giảm kích thước từ vựng & tiết kiệm bộ nhớ |
| **Trích xuất đặc trưng mới** | `excl_count` & `word_count` | `excl_count` có tương quan cao nhất với Spam (+0.30) |
| **Biến đổi đặc trưng số** | `log1p` transformation | Đưa đặc trưng đếm về cùng thang đo với TF-IDF [0, 1] |
| **Giới hạn độ sâu `max_depth`** | 10–15 | ~5.7K mẫu; ngăn chặn OOM và quá bớp (Overfitting) |
| **Loại bỏ ngoại lệ (Outliers)** | Không | Mô hình dựa trên Cây quyết định kháng ngoại lệ tốt |
