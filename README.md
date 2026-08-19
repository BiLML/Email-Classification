# PRE1 - Email Classification

## Cấu trúc thư mục dự án

```text
PRE1 - Email Classfication/
├── README.md                       # Tài liệu hướng dẫn sử dụng và cấu trúc dự án
├── data/                           # Thư mục lưu trữ dữ liệu và báo cáo theo từng công đoạn
│   ├── raw/                        # Dữ liệu thô ban đầu (emails.csv và tài liệu chuẩn bị)
│   ├── EDA/                        # Báo cáo phân tích EDA và biểu đồ trực quan hóa (eda_plots/)
│   ├── processed/                  # Dữ liệu đã qua làm sạch, phân chia và ma trận TF-IDF (.npz, .pkl)
│   └── training/                   # Ghi chú quy trình huấn luyện (Training.md)
├── docs/                           # Tài liệu tổng thể dự án, quy tắc và lịch sử
│   ├── PLAN.md                     # Kế hoạch tổng thể dự án
│   ├── List notes/                 # Định nghĩa bài toán, báo cáo đánh giá và phân tích tiến độ
│   ├── behavioral/                 # Quy tắc làm việc và các ràng buộc kỹ thuật
│   ├── history/                    # Lịch sử phát triển dự án
│   └── knowledge base/             # Cơ sở kiến thức
├── models/                         # Thư mục lưu trữ các trọng số mô hình đã huấn luyện (.pkl)
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
└── src/                            # Mã nguồn thực thi dự án
    ├── EDA/                        # Scripts phân tích dữ liệu khám phá (Step 1 - 5)
    │   ├── step_1_data_loading.py
    │   ├── step_2_data_quality.py
    │   ├── step_3_class_distribution.py
    │   ├── step_4_text_analysis.py
    │   └── step_5_visualization.py
    ├── Preprocessing/              # Scripts thực thi tiền xử lý dữ liệu (Cleaning, Split, TF-IDF)
    │   ├── step_1_cleaning.py
    │   ├── step_2_split.py
    │   └── step_3_tfidf.py
    ├── training/                   # Các lớp mô hình học máy tự viết từ đầu (OOP)
    │   ├── logistic_regression.py
    │   ├── naive_bayes.py
    │   └── svm.py
    └── evalute/                    # Scripts huấn luyện, đánh giá mô hình và kiểm thử End-to-End
        ├── train_and_evaluate.py   # Script chạy huấn luyện và đánh giá 3 mô hình
        └── test_e2e.py             # Script kiểm thử End-to-End toàn bộ pipeline dự đoán
```

## Cách chạy dự án

### Bước 1: Chạy phân tích dữ liệu khám phá (EDA)
Để thực hiện phân tích thống kê và tạo các biểu đồ trực quan hóa dữ liệu:
```bash
cd src/EDA
python step_1_data_loading.py
python step_2_data_quality.py
python step_3_class_distribution.py
python step_4_text_analysis.py
python step_5_visualization.py
```

### Bước 2: Thực hiện tiền xử lý dữ liệu
Chạy lần lượt các bước làm sạch văn bản, phân chia dữ liệu tầng (Stratified Split) và vector hóa TF-IDF:
```bash
cd src/Preprocessing
python step_1_cleaning.py
python step_2_split.py
python step_3_tfidf.py
```
Sau bước này, ma trận dữ liệu sẽ được lưu tại thư mục `data/processed/`.

### Bước 3: Huấn luyện và đánh giá mô hình
Chạy script huấn luyện 3 mô hình (Logistic Regression, Linear SVM, Multinomial Naive Bayes) và lưu kết quả vào thư mục `models/`:
```bash
cd src/evalute
python train_and_evaluate.py
```

### Bước 4: Kiểm thử End-to-End (E2E)
Chạy script kiểm thử toàn bộ pipeline với văn bản email thô mới:
```bash
cd src/evalute
python test_e2e.py
```


---

## Kết quả đánh giá mô hình

Kết quả thử nghiệm trên tập Test (856 mẫu):
- Multinomial Naive Bayes: Accuracy 95.09%, Precision 100.00%, Recall 79.61%, F1-Score 88.65%
- Linear SVM (SGD): Accuracy 83.64%, Precision 83.00%, Recall 40.29%, F1-Score 54.25%
- Logistic Regression (SGD): Accuracy 81.89%, Precision 74.76%, Recall 37.38%, F1-Score 49.84%
