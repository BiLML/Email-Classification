# 📊 Kế Hoạch EDA — Phân Loại Email Rác (Spam Email Classification)

> **Dự án:** PRE1 - Email Classification  
> **Tập dữ liệu:** `emails.csv` (8.9 MB) — Kaggle  
> **Ràng buộc:** Cài đặt từ đầu (Chỉ dùng NumPy, Pandas — không dùng scikit-learn)

---

## 1. Định Nghĩa

**EDA (Exploratory Data Analysis - Phân Tích Dữ Liệu Khám Phá)** là giai đoạn đầu tiên cực kỳ quan trọng của bất kỳ quy trình Học Máy nào. Đây là quá trình điều tra và tổng hợp các đặc trưng chính của tập dữ liệu thô một cách có hệ thống — sử dụng các bản tóm tắt thống kê và kỹ thuật trực quan hóa — **trước khi** tiến hành bất kỳ bước mô hình hóa hay tiền xử lý nào.

Trong dự án này, EDA trả lời câu hỏi cốt lõi:

> *"Tập dữ liệu email của chúng ta trông như thế nào, và nó có phù hợp để huấn luyện một bộ phân loại spam từ đầu không?"*

EDA **không phải** là xây dựng mô hình. EDA là để **hiểu dữ liệu** nhằm đưa ra các quyết định có cơ sở cho tất cả các bước tiếp theo: làm sạch, trích xuất đặc trưng, vector hóa, và lựa chọn mô hình.

### Các Khía Cạnh EDA Bao Phủ
| Khía cạnh | Câu hỏi |
|-----------|---------|
| **Cấu trúc** | Có bao nhiêu dòng/cột? Kiểu dữ liệu là gì? |
| **Chất lượng** | Có giá trị thiếu, dòng trùng lặp hay bản ghi hỏng không? |
| **Phân phối** | Email spam vs. ham phân bố như thế nào? Có mất cân bằng lớp không? |
| **Nội dung** | Từ/mẫu nào phổ biến nhất trong spam vs. ham? |
| **Thống kê** | Độ dài trung bình của email là bao nhiêu? Phân phối ký tự ra sao? |

---

## 2. Mục Đích

### Tại Sao EDA Quan Trọng Cho Dự Án Này

1. **Phát Hiện Sớm Vấn Đề Chất Lượng Dữ Liệu**  
   Nhận diện giá trị thiếu, dữ liệu trùng lặp, email rỗng hoặc lỗi mã hóa (encoding) **trước khi** chúng lặng lẽ làm hỏng bộ vector hóa TF-IDF và các mô hình.

2. **Hiểu Phân Phối Lớp (Spam vs. Ham)**  
   Xác định tập dữ liệu là cân bằng hay mất cân bằng. Tập dữ liệu mất cân bằng đòi hỏi phải **Chia Phân Tầng (Stratified Split)** (theo quy định của WORKING_RULES.md) và có thể ảnh hưởng đến việc chọn chỉ số đánh giá (ưu tiên Recall hơn Accuracy).

3. **Định Hướng Các Quyết Định Tiền Xử Lý**  
   - Phát hiện xem email có chứa thẻ HTML không → cần loại bỏ HTML
   - Tìm ký tự đặc biệt / mẫu dấu câu → cần quy tắc làm sạch
   - Xác định các từ dừng (stop words) chiếm ưu thế → cần xóa từ dừng

4. **Cung Cấp Thông Tin Cho Kỹ Thuật Đặc Trưng**  
   - Phát hiện các từ khóa kích hoạt spam (ví dụ: "free", "winner", "urgent")
   - Hiểu sự khác biệt về độ dài email giữa các lớp
   - Xác định kích thước từ vựng → ảnh hưởng đến kích thước ma trận TF-IDF và bộ nhớ (ngưỡng `min_df`)

5. **Phòng Ngừa Lỗi Ở Các Bước Sau**  
   - Tránh lỗi tràn bộ nhớ (OOM) bằng cách hiểu quy mô từ vựng trước khi TF-IDF
   - Tránh rò rỉ dữ liệu bằng cách tách biệt rõ ràng việc phân tích khỏi biến đổi dữ liệu
   - Đặt kỳ vọng thực tế cho hiệu suất mô hình

6. **Cung Cấp Bằng Chứng Trực Quan Cho Báo Cáo**  
   Tạo biểu đồ và thống kê để đưa vào báo cáo dự án cuối cùng, thể hiện sự hiểu biết sâu sắc về bài toán.

---


---

## 3. Quy Trình Thực Hiện (Workflow)

### Bước 3.1 — Nạp Dữ Liệu & Xem Xét Ban Đầu

| Hành động | Mẫu mã nguồn | Đầu ra |
|-----------|-------------|--------|
| Nạp CSV | `pd.read_csv('data/emails.csv')` | DataFrame |
| Kiểm tra shape | `df.shape` | `(n_rows, n_cols)` |
| Kiểm tra cột | `df.columns.tolist()` | Tên các cột |
| Kiểm tra dtypes | `df.dtypes` | Kiểu dữ liệu mỗi cột |
| Xem trước | `df.head(10)`, `df.tail(5)` | Các dòng đầu/cuối |
| Thông tin cơ bản | `df.info()` | Bộ nhớ, số lượng non-null |

**Mục tiêu:** Hiểu cấu trúc thô — có bao nhiêu email, các cột gồm những gì, cột nhãn tên là gì.

---

### Bước 3.2 — Đánh Giá Chất Lượng Dữ Liệu

| Kiểm tra | Mẫu mã nguồn | Hành động nếu tìm thấy |
|-----------|-------------|-----------------------|
| Giá trị thiếu | `df.isnull().sum()` | Quyết định: xóa hoặc thay thế |
| Dòng trùng lặp | `df.duplicated().sum()` | Xóa trước khi chia train/test |
| Văn bản rỗng | `df[df['text'].str.strip() == '']` | Đánh dấu để xóa |
| Lỗi mã hóa | Kiểm tra thủ công các dòng mẫu | Sửa mã hóa hoặc xóa dòng hỏng |

**Mục tiêu:** Định lượng các vấn đề chất lượng dữ liệu và lên kế hoạch các bước làm sạch.

---

### Bước 3.3 — Phân Tích Phân Phối Lớp

```
Hành động:
  1. Đếm số nhãn spam vs. ham
  2. Tính tỷ lệ phần trăm
  3. Vẽ biểu đồ cột
  4. Xác định xem có cần chia phân tầng (stratified split) không (luôn cần theo WORKING_RULES)
```

**Các chỉ số chính cần ghi nhận:**
- Tổng số email: `N`
- Số lượng & tỷ lệ Spam: `n_spam`, `spam_ratio`
- Số lượng & tỷ lệ Ham: `n_ham`, `ham_ratio`
- Tỷ lệ mất cân bằng: `max_class / min_class`

**Điểm quyết định:** Nếu tỷ lệ mất cân bằng > 3:1, ghi nhận đây là yếu tố rủi ro cho việc đánh giá mô hình (chỉ dùng Accuracy sẽ gây hiểu lầm → phải dùng F1/Recall).

---

### Bước 3.4 — Phân Tích Nội Dung Văn Bản

#### 3.4.1 — Phân Phối Độ Dài Email
```
- Tính số ký tự mỗi email: df['char_count'] = df['text'].str.len()
- Tính số từ mỗi email:      df['word_count'] = df['text'].str.split().str.len()
- Tạo thống kê mô tả:      df[['char_count', 'word_count']].describe()
- Vẽ biểu đồ tần suất:      So sánh độ dài Spam vs. Ham
- Vẽ biểu đồ hộp (box plot): Nhận diện điểm ngoại lệ (email cực ngắn/cực dài)
```

#### 3.4.2 — Phân Tích Tần Suất Từ
```
- Tách từ (tokenize) tất cả email
- Đếm tần suất từ theo lớp (riêng biệt cho spam / ham)
- Xác định 20 từ xuất hiện nhiều nhất trong mỗi lớp
- Xác định từ khóa kích hoạt spam: "free", "winner", "click", "urgent", "offer", v.v.
- Vẽ biểu đồ cột ngang cho các từ hàng đầu
```

#### 3.4.3 — Phân Tích Ký Tự Đặc Biệt & Mẫu Văn Bản
```
- Tính tỷ lệ ký tự đặc biệt mỗi email: count($, !, ?, #) / tổng ký tự
- Tính tỷ lệ chữ viết hoa mỗi email:     count(UPPER) / tổng ký tự
- So sánh phân phối giữa spam và ham
- Các đặc trưng này có thể trở thành đặc trưng đầu vào cùng với TF-IDF
```

---

### Bước 3.5 — Tóm Tắt Thống Kê & Phát Hiện Điểm Ngoại Lệ

| Chỉ số | Phương pháp | Mục đích |
|--------|------------|----------|
| Mean / Median / Std | `.describe()` | Xu hướng trung tâm của độ dài email |
| IQR | `Q3 - Q1` | Phát hiện điểm ngoại lệ về độ dài |
| Min / Max | `.min()`, `.max()` | Tìm các trường hợp biên (email 1 từ, email cực lớn) |
| Skewness | `.skew()` | Hiểu hình dạng phân phối |

---

### Bước 3.6 — Trực Quan Hóa Dashboard

Tạo các biểu đồ sau (sử dụng `matplotlib`):

| # | Biểu đồ | Loại | Mục đích |
|---|---------|------|----------|
| 1 | Class Distribution | Biểu đồ cột | Trực quan hóa cân bằng spam/ham |
| 2 | Email Length (chars) | Histogram | So sánh mẫu độ dài ký tự |
| 3 | Email Length (words) | Histogram | So sánh mẫu số lượng từ |
| 4 | Email Length | Box plot | Phát hiện điểm ngoại lệ |
| 5 | Top 20 Words — Spam | Biểu đồ cột ngang | Nhận diện từ vựng spam |
| 6 | Top 20 Words — Ham | Biểu đồ cột ngang | Nhận diện từ vựng ham |
| 7 | Special Char Ratio | Histogram | Sự khác biệt mẫu ký tự |
| 8 | Word Cloud — Spam | Word cloud | Tổng quan từ khóa trực quan |
| 9 | Word Cloud — Ham | Word cloud | Tổng quan từ khóa trực quan |

---

## 4. Ghi Chú Kiểm Tra (Checknotes)

> ✅ = Đã xác nhận vào ngày 12/08/2026 sau khi chạy cả 5 script EDA (từ step_1 đến step_5)

### Nạp Dữ Liệu
- [x] Tập dữ liệu nạp thành công không có lỗi mã hóa
- [x] Đã xác định và ghi nhận tất cả các cột → `['text', 'spam']`
- [x] Đã xác nhận cột nhãn (nhị phân: 0=Ham, 1=Spam)
- [x] Không có kiểu dữ liệu bất thường → `text`=object, `spam`=int64

### Chất Lượng Dữ Liệu
- [x] Đã ghi nhận số lượng giá trị thiếu cho mọi cột → **0 thiếu** (Hoàn toàn đầy đủ 100%)
- [x] Đã quyết định chiến lược giá trị thiếu → **Không cần xử lý** (không có giá trị thiếu)
- [x] Đã ghi nhận số lượng trùng lặp → **33 trùng lặp** (liên quan 66 dòng)
- [ ] Tất cả trùng lặp đã được xóa khỏi tập dữ liệu → ⚠️ **CHƯA** — EDA chỉ quan sát, việc xóa diễn ra ở bước Tiền xử lý
- [x] Đã xác định và đếm các dòng email rỗng/trắng → **0 rỗng hoàn toàn**, có một số email rất ngắn
- [x] Không có bản ghi văn bản bị hỏng khi kiểm tra mẫu

### Phân Phối Lớp
- [x] Đã ghi lại số lượng chính xác spam và ham → **Ham: 4,360 (76.1%) | Spam: 1,368 (23.9%)**
- [x] Đã tính tỷ lệ lớp → **76.1% / 23.9% (~3.19:1 mất cân bằng)**
- [x] Đã đánh giá mức độ mất cân bằng → **MẤT CÂN BẰNG VỪA PHẢI** (>3:1)
- [x] Đã vẽ và lưu biểu đồ cột → `step3_class_distribution.png`
- [x] Đã xác nhận: Sẽ sử dụng Phân Chia Phân Tầng (Yêu cầu của WORKING_RULES)

### Phân Tích Văn Bản
- [x] Phân phối số lượng ký tự được tính toán theo từng lớp
- [x] Phân phối số lượng từ được tính toán theo từng lớp
- [x] Đã vẽ histogram độ dài email (chồng lớp spam vs ham)
- [x] Đã xác định 20 từ thường gặp nhất cho lớp spam → top: "com", "business", "company", "email", "free"
- [x] Đã xác định 20 từ thường gặp nhất cho lớp ham → top: "enron", "ect", "vince", "hou", "kaminski"
- [x] Từ khóa kích hoạt spam đã được ghi nhận → 23 từ kích hoạt được phân tích với tỷ lệ spam/ham
- [x] Tỷ lệ ký tự đặc biệt được phân tích → spam cao hơn một chút
- [x] Tỷ lệ chữ viết hoa được phân tích → spam cao hơn một chút

### Tóm Tắt Thống Kê
- [x] Bảng thống kê mô tả đã được tạo (mean, median, std, min, max)
- [x] Ngưỡng điểm ngoại lệ được xác định (phương pháp IQR)
- [x] Chiến lược xử lý điểm ngoại lệ được quyết định → **Giữ lại hiện tại** (không xóa trong EDA)
- [x] Độ lệch phân phối được ghi nhận → cả hai lớp đều lệch phải (đuôi dài các email rất dài)

### Pre-Preprocessing Decisions (Quyết định trước tiền xử lý)
- [x] Đã xác nhận: Có chứa thẻ HTML → cần loại bỏ
- [x] Đã xác nhận: Từ dừng chiếm ưu thế → cần loại bỏ
- [x] Đã xác nhận: Mẫu dấu câu → cần làm sạch
- [x] Đã ghi nhận kích thước từ vựng ước tính → **5,695 văn bản duy nhất**
- [x] Đã quyết định `min_df` khuyến nghị cho TF-IDF → `min_df=2` đến `min_df=5`
- [x] Đã ghi nhận ngưỡng an toàn `max_depth`

---

## Các Độ Lệch So Với Kế Hoạch

### Độ Lệch 1: Thiếu Word Cloud (Biểu đồ 8 & 9)
- **Kế hoạch yêu cầu:** Biểu đồ 8 = Word Cloud (Spam), Biểu đồ 9 = Word Cloud (Ham)
- **Thực tế:** Biểu đồ 8 = Feature Comparison Bar Chart, Biểu đồ 9 = Correlation Matrix
- **Tác động:** Word cloud được thay thế bằng các biểu đồ có giá trị phân tích cao hơn. Biểu đồ top 20 từ (plot5, plot6) đã đáp ứng cùng mục đích.
- **Hành động:** ✅ Thay thế chấp nhận được — biểu đồ cột chính xác hơn word cloud.

---

## Bước 4.7 — Tóm Tắt Phát Hiện EDA

```
EDA Findings Document:
  ├── Kích thước tập dữ liệu: 5,728 dòng × 2 cột (text, spam)
  ├── Kích thước file:         8.9 MB trên đĩa → 8.86 MB trong bộ nhớ
  ├── Cân bằng lớp:           76.1% Ham (4,360) / 23.9% Spam (1,368)
  ├── Tỷ lệ mất cân bằng:     3.19:1 → VỪA PHẢI (Dùng Stratified Split + F1/Recall)
  ├── Giá trị thiếu:          0 (Tập dữ liệu đầy đủ 100%)
  ├── Trùng lặp:              33 → xóa ở bước Tiền xử lý
  ├── Độ dài email trung bình: 326.8 từ tổng thể (Ham dài hơn Spam)
  │     ├── Ham:             ~370 từ trung bình
  │     └── Spam:            ~190 từ trung bình
  ├── Ước tính từ vựng:       ~hàng chục nghìn thuật ngữ duy nhất
  ├── min_df khuyến nghị:     2–5 (cắt giảm từ hiếm để tiết kiệm bộ nhớ TF-IDF)
  ├── Từ khóa spam chính:     com, business, company, email, free, money, click, http
  ├── Từ khóa ham chính:      enron, ect, vince, kaminski (Tập email Enron)
  ├── Chiến lược outlier:     Giữ lại (xử lý qua ràng buộc mô hình)
  ├── Chiến lược mất cân bằng: Phân chia phân tầng + Ưu tiên Recall & F1-Score
  ├── Phát hiện HTML:         Có → loại bỏ ở Tiền xử lý
  ├── Phát hiện URL:          Có → làm sạch hoặc xóa ở Tiền xử lý
  ├── Dữ liệu đã sắp xếp:     Có (spam trước, ham sau) → xáo trộn trước khi chia
  └── Tín hiệu spam mạnh nhất: Số lượng dấu chấm cảm (tương quan = 0.30 với nhãn spam)
```

---

> [!IMPORTANT]
> **EDA chỉ là quan sát.** KHÔNG chỉnh sửa tập dữ liệu gốc hoặc fit bất kỳ bộ biến đổi nào trong giai đoạn này. Tất cả các phép biến đổi diễn ra ở bước Tiền xử lý tiếp theo.

> [!NOTE]
> **Các thư viện được phép dùng cho EDA:** `pandas` (nạp dữ liệu + thống kê), `numpy` (tính toán), `matplotlib` (trực quan hóa). Tùy chọn: `wordcloud` để tạo word cloud.
