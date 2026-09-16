import json
import os

def create_markdown_cell(source_text):
    lines = [line + '\n' for line in source_text.split('\n')]
    # Remove the trailing newline on the last line to follow standard notebook style
    if lines:
        lines[-1] = lines[-1].strip('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def main():
    notebook_path = 'src/lab1.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    mark_dict = {
        0: "# Lab 1: Phân loại Email (Spam/Ham) - Từ cơ bản đến nâng cao\n\n**Mục tiêu:**\n- Xây dựng một pipeline hoàn chỉnh để phân loại email.\n- Tự xây dựng các thuật toán TF-IDF, Naive Bayes, Logistic Regression, SVM từ đầu (from scratch).\n\n## Phần 1: Khởi tạo và Import thư viện",
        
        1: "## Phần 2: Nạp dữ liệu và Khám phá ban đầu (EDA)\n\nỞ phần này, chúng ta sẽ đọc dữ liệu, xem xét một số mẫu, và vẽ biểu đồ phân bố nhãn `Spam` và `Ham`.",
        
        2: "## Phần 3: Tiền xử lý và Làm sạch dữ liệu\n\nBắt đầu bằng việc kiểm tra các giá trị bị thiếu (Missing Values) để đảm bảo chất lượng dữ liệu.",
        
        3: "### Tạo đặc trưng mới: Độ dài văn bản\n\nChúng ta thử tạo thêm cột `char_len` để phân tích độ dài các email Spam so với Ham.",
        
        5: "### Xử lý Dữ liệu trùng lặp\n\nKiểm tra và xử lý các dòng trùng lặp, xem xét sự mâu thuẫn về nhãn cho cùng một nội dung.",
        
        7: "## Phần 4: Phân tích N-gram (Bigrams)\n\nSử dụng kỹ thuật Bigram để xem cụm 2 từ nào xuất hiện nhiều nhất trong thư rác và thư thường.",
        
        11: "### Loại bỏ các dòng trùng lặp cuối cùng\n\nThực hiện xóa các dữ liệu trùng sau khi đã phân tích xong để tránh rò rỉ dữ liệu.",
        
        13: "## Phần 5: Xây dựng TF-IDF Vectorizer từ đầu (From Scratch)\n\nTF-IDF là một phương pháp cốt lõi trong NLP. Chúng ta sẽ xây dựng lớp `TfidfVectorizerScratch` để tự tính toán trọng số này.",
        
        15: "## Phần 6: Xây dựng các Mô hình Học Máy từ đầu\n\nThay vì dùng thư viện có sẵn, chúng ta sẽ cài đặt ba thuật toán classification phổ biến:\n1. **Multinomial Naive Bayes**\n2. **Logistic Regression**\n3. **Linear SVM**",
        
        18: "## Phần 7: Hàm Đánh giá (Evaluation Metrics)\n\nViết hàm để dễ dàng tính toán Accuracy, Precision, Recall và F1-Score.",
        
        19: "### Đánh giá các mô hình trên tập Validation\n\nHuấn luyện và so sánh xem thuật toán nào hoạt động tốt nhất trước khi tinh chỉnh.",
        
        20: "## Phần 8: Tinh chỉnh ngưỡng quyết định (Threshold Tuning)\n\nCác mô hình thường dùng ngưỡng xác suất 0.5. Ta sẽ quét ngưỡng từ 0.1 đến 0.9 để tìm ngưỡng cho F1-Score cao nhất.",
        
        21: "## Phần 9: Kiểm thử cuối cùng trên tập Test\n\nÁp dụng mô hình và ngưỡng quyết định tốt nhất lên dữ liệu kiểm thử thực tế.",
        
        22: "## Phần 10: Lưu trữ và Tái sử dụng mô hình\n\nCuối cùng, đóng gói các trọng số của mô hình thành file JSON để tiện lợi cho việc triển khai (deployment)."
    }

    new_cells = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            if i in mark_dict:
                new_cells.append(create_markdown_cell(mark_dict[i]))
        new_cells.append(cell)
        
    nb['cells'] = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Thêm thành công {len(mark_dict)} markdown cells vào {notebook_path}.")

if __name__ == '__main__':
    main()
