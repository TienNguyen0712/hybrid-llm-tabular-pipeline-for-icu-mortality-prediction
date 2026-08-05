# Xây dựng một pipeline từ dữ liệu thô cho đến dự đoán tỷ lệ tử vong của bệnh nhân trong khoa ICU
--- 
## Tổng quan

Đề tài y tế đang là một trong những đề tài nổi bật trong việc ứng dụng các công nghệ kỹ thuật vào. Các bài toán luôn luôn được xuất hiện 
- Dự đoán thời gian nằm ICU
- Dự đoán bệnh sốc nhiễm khuẩn
- Dự đoán shock
- ...

Một trong số đó chính là bài toán dự đoán tỷ lệ tử vong ở bệnh nhân nằm trong ICU. Ở dự án này cung cấp một quy trình từ dữ liệu thô đến khi đưa ra được dự đoán bằng các mô hình học sâu / học máy 
- ICU: Là tên viết tắt của hồi sức cấp cứu. Nơi đây có rất nhiều bệnh nhân trong tình trạng nặng có người sông sót có người không. Do đó nhận thấy việc dự đoán tỷ lệ tử vong là rất cần thiết
  - Một phần cung cấp cái nhìn cho các bác sĩ sử lý
  - Một phần giảm tỷ lệ các ca bệnh tử vong

---

## Bộ dữ liệu 

MIMIC-IV v3.1 là dữ liệu phù hợp bởi khối lượng dữ liệu lớn, nhiều nhiễu cũng như bám sát với dữ liệu ý tế thực tế. Đề tài lựa chọn 2 module chính 
- hosp: Chứa thông tin lâm sàng của bệnh nhân, thủ tục nhập viện, các chỉ số lâm sàng khi nhập viện cur bệnh nhân 
- icu: Chứa các thông tin trong phòng nghiên cứu, lab, dịch truyền vào, ra của bệnh nhân khi nàm tại ICU

Đề tài không sử dụng toàn bộ các bảng trong database rộng lớn mà chọn lọc ra 8 bảng chính để chọn lọc đặc trưng cùng với 2 bảng phụ nhằm lấy các chỉ số trong phòng lab cũng như các chỉ số được đo trong phòng ICU

---

## Phương pháp đề xuất

Để phù hợp với mục tiêu của đề tài. Ta sẽ xây dựng một quy tắc nhằm gia tăng độ chính xác
- Các bệnh nhân phải trên 18 tuổi (Điều này gia tăng thêm mẫu và có cái nhìn lớn so với người trưởng thành)
- Lấy bắt đầu từ lần vào ICU đầu tiên (Do 1 bệnh nhân sẽ có rất nhiều lần vào ICU điều này giúp cho mô hình không bị nhiễu)
- Thời gian nằm ICU phải từ 24h trở đi (Điều này cũng làm gia tăng thêm mẫu)

### Thực hiện chọn lọc đặc trưng cho dữ liệu 

8 bảng được lựa chọn chính bao gồm 

- icustays: Gồm các thông tin về lượt nằm trong ICU (Đặc trưng cần lấy: `first_careunit`, `intime`, `subject_id`)
- patients: Gồm các thông tin về bệnh nhân (Đặc trưng cần lấy: `subject_id`, `gender`, `anchor_age`, `anchor_age_group`, `anchor_year`)
- admissions: Gồm các thông tin về thủ tục nhập viện (Đặc trưng cần lấy: `subject_id`, `hadm_id`, `admittime`, `admission_type`, `admission_location`, `race`, `insurance`, `edregtime`, `edouttime`)
- labevents: Gồm các thông tin về chỉ số lâm sàng (Đặc trưng cần lấy: `itemid`, `charttime`, `valuenum`, `flag`) - liên kết với bảng `d_label` để chọn ra dặc trưng rồi chuyển thành côt
- chartevents: Gồm các thông tin về chỉ số lầm sàng tại giường bệnh trong ICU: (Đặc trưng cần lấy: `itemid`, `charttime`, `valuenum`) - liên kết với bảng `d_item` để chọn ra dặc trưng rồi chuyển thành côt
- prescriptions: Gồm các thông tin về dược lý. Chia theo nhóm dược lý thay cho các tên thuốc thô
- inputevents / outputevents: Gồm các thông tin về dịch vào và dịch ra của bệnh nhân. Tính cân bằng dịch

### Xử lý đặc trưng thành các thông tin 
- Dùng các cột đã pivot trong bảng lab và chart tính toán min/max/std/ giá trị đầu tiên và cuối cùng trong 1 bệnh nhân sau những lần đo, delta: biến thiên chỉ số
- Xử lý missing
- Tính toán điểm phù hợp
- Bơm tri thức cho LLM kèm schema chính để trích xuất tạo đặc trưng mới
- Đánh giá các tri thức đặc trưng sau khi bơm

### Chọn lọc đặc trưng 

- Sau output của LLM cũng như các chỉ số lâm sàng chọn đăc trưng tạo Matrix cuối cùng -> Thực hiện huấn luyện

