# Bài thuyết trình về quy trình từ dữ liệu thô cho đến một mô hình dự đoán tử vong ICU

--- 

Mục tiêu của phương pháp này là giúp đỡ hỗ trợ việc ra quyết định cho các bác sĩ đối với bệnh nhân vào viện ICU: 
- Quy trình sẽ được chia làm 3 bước chính: 
  - Bước 1: Biến dữ liệu thô thành các dữ liệu có giá trị 
  - Bước 2: Từ dữ liệu có giả trị + Thống kê + Tri thức LLM và RAG ==> Dữ liệu huấn luyện cuối cùng
  - Bước 3: Huấn luyện mô hình để ra quyết định so sánh đô hiệu quả + Giải thích kết quả và là nguồn cho bác sĩ tham khảo

- **Tổng quan:** Sử dụng dữ liệu thô két hợp việc trích xuất thông tin tài liệu từ RAG cùng hiểu rõ ngữ cảnh các biến định tính của LLM để tạo lựa chọn đặc trưng cho học máy
- **Bài toán được lựa chọn:** Dự doán tỷ lệ tử vong ở ICU trong 24h dầu

## Bước 1: Từ dữ liệu thô đến các dữ liệu có giá trị 

**Mục tiêu:** Từ các bảng nhỏ thực hiện sinh ra các bảng mang giá trị đúng với bài toán

- Lựa chọn 10 bảng làm bảng mục tiêu 
  - hosp: patients, admission,  
  - icu:
### 1.1. Sinh cohort



