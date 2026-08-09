# Xây dựng một mô hình dự đoán tỷ lệ sống sót của bệnh nhân trong khoa ICU

## Tổng quan 

Đề tài này kết hợp một mô hình ngôn ngữ lớn trong việc dự đoán tỷ lệ tử vong của bệnh nhân trong ICU: 
- Hướng tới mục tiêu tự động hóa việc tạo dặc trưng
- Tiết kiệm chi phí nhân công
- Dễ dàng quản lý

Bộ dữ liệu sử dụng MIMIC-IV 2 module chính: 
- hosp: Chứa thông tin bệnh nhân, thủ thuật, các lần nằm viện
- icu: Chứa thông tin lâm sàng trong khoa cấp cứu bệnh viện, dịch ra và dịch vào của bệnh nhân khi nằm trong ICU

---

## Mô tả kỹ thuật

Đề tài này thực hiện dựa trên 9 bảng bao gồm:
- patients: Thông tin bệnh nhân
- admission: Thông tin hành chính
- icustay: Thông tin lần nằm ICU của bệnh nhân
- labevents: Chỉ số lâm sàng trong phòng lab 
- chartevents: Chỉ số lâm sàng được đo trong giường nằm của bệnh nhân
- prescriptions: Chứa thông tin về thuốc 
- procedureevents: Thủ thuật trong ICU
- inputevents: Dịch vào của bệnh nhân
- outputevents: Dịch ra của bệnh nhân 

Ngoài ra còn kết hợp với các bảng khác để lấy mô tả của các đặc trưng

### 1. Xây dựng bảng ban đầu

#### 1.1 Xây dựng điều kiện 

Hướng tới mục tiêu của đề tài. Đề xuất các diều kiện sau
- Các bệnh nhân dược lựa chọn phải là người trường thành (Anchor_age >= 18)
- Dợt nằm ICU phải là những đợt đàu tiên của bệnh nhân
- Thời gian nằm viện phải trên 24g kể từ khi vào ICU

> Câu hỏi: Có nên lấy hoặc sử dụng cột `anchor_year` với mục đích tính `anchor_age` hay không? Do tuôi trong bảng có giới hạn và muốn tính tuổi thật thì phải dính tới `anchor_year` (`anchor_age` là tuổi mà bệnh nhân khai khi mới vào viện)

Dựa theo mục tiêu trên mà tác giả đã xử dụng 3 bảng: 

```
patients + icustay + admission (INNER JOIN) -> subject_id, hadm_id
```

**Đặc trưng được lựa chọn đáng chú ý**

- patients: `"subject_id", "gender", "anchor_age", "anchor_year", "dod"`

- admission: `"subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                       "admission_location", "edregtime", "edouttime", "deathtime",
                       "hospital_expire_flag", "insurance", "race"`
- icustay: giữ nguyên

_Lưu ý:_ Cột `dod`, `hospital_expire_flag`, `deathtime` sẽ được chọn làm nhãn và cột `dischtime` sẽ được cân nhắc loại bỏ

#### 1.2 Xây dựng bảng lâm sàng theo `chartevents`

**Đặc trưng được lựa chọn đáng chú ý**

- Nhịp tim - 220045
- Nhịp thở - 220210
- Nhiệt độ cơ thể độ C - 223762
- Nhiệt độ cơ thể độ F - 223761
- Huyết áp tâm thu/ tâm trương/ trung bình xâm lấn - 220050 / 220051 / 220052 (Trung bình)
- Huyết áp tâm thu/ tâm trương/ trung bình không xâm lấn - 220179 / 220180 / 220181
- SpO2 - 220277
- Glucose máu - 225664
- pH - 223830
- FiO2 - 223835
- Công thức máu -  220228 (Hemoglobin) / 220545 (Hematocrit)/ 220546 (WBC)227457 (Platelet)
- Chức năng thận - 220615 (Creatinine) / 225624 (BUN)
- Điện giải - 227442 (Potassium) / 220645 (Sodium) / 220602 (Chloride) / 227443 (HCO3) / 220635 (Magnesium)
- Khí máu - 223830 (pH Arterial) / 220224 (PaO2)/ 220235 (PaCO2) / 224828 (Base Excess) / 220227 (SaO2)
- Chức năng gan - 220644 (ALT) / 220587 (AST) / 225690 (Bilirubin Toàn phần) / 227456 (Albumin) / 225612 (Alkaline Phosphate)
- Thang điểm Glagsgow - 223900 (Verbal) / 223901 (Motor) / 220739 (Eye)
- Marker tim - 227429 (Troponin-T) / 227445 (CK-MB)227446 (BNP)
- Nội tiết - 228236 (Insulin) / 227463 (Cortisol)
- Cân nặng - 224639 (Cân nặng hàng ngày) / 226512 (Cân nặng nhập viện Kg)
- Chiều cao - 226707 / 226730

Các đặc trưng sau khi lựa chọn xong sẽ được di chuyển qua các bước tiền xử lý như: 
- Lọc nhiễu chọn ra cấc đơn vị nằm trong khoảng không bị nhiễu
- Biến đổi về kiểu dữ liệu phù hợp
- Chuyển về chung một đơn vị
- Tính min, max, mean các giá trị thống kê
- Chuyển thành pivot (Đổi các dòng giá trị thành cột)

> Câu hỏi: Có nên gộp chung với bảng điều kiện được xây dựng ở trên để lấy các đặc trưng tương đương hay không ?, Có nên thêm các mốc cảnh báo nhằm sinh đặc trung true/false nếu những vượt quá

#### 1.3 Xây dựng bảng lab theo `labevents`

**Đặc trưng được lựa chọn đáng chú ý**

- Nhiệt độ cơ thể - 50825
- SpO2 / Bão hòa O2 - 50817
- Glucose máu - 50931 (Huyết tương )/ 50809 (Máu toàn phần)
- pH - 50820
- Lactate - 50813
- Công thức máu -  51221 (Hematocrit) / 51222 (Hemoglobin) / 51301 (Bạch cầu - WBC) / 51265 (Tiểu cầu) / 51279 (Hồng cầu - RBC)
- Chức năng thận - 50912 (Creatinine) / 51006 (BUN) 
- Điện giải - 50971 (Kali / Potassium) / 50983 (Natri / Sodium) / 50902 (Clo / Chloride) / 50882 (Bicarbonate / HCO3) / 50893 (Calci toàn phần) / 50808 (Calci tự do) / 50960 (Magie) / 50970 (Phosphate)
- Khí máu - 50821 (pO2) / 50818 (pCO2) / 50802 (Base Excess) / 50820 (pH) / 50868 (Anion Gap)
- Chức năng gan - 50861 (ALT / SGPT) / 50878 (AST / SGOT) / 50863 (Alkaline Phosphatase) / 50885 (Bilirubin toàn phần) / 50862 (Albumin) / 50976 (Protein toàn phần)
- Marker tim - 51003 (Troponin T) / 50911 (CK-MB) / 50910 (CK tổng) / 50963 (NT-proBNP)
- Nội tiết - 50993 (TSH) / 50995 (Free T4) / 51001 (T3) / 50909 (Cortisol) / 50965 (PTH)

Đặc trưng sau khi lựa chọn cũng sẽ giống với `chartevents` cũng đi qua các bước tiền xử lý

> Câu hỏi: Có nên lấy các xét nghiệm từ labevents còn chartevents là lấy các sô sinh hiệu liên tục hay không

#### 1.4 Xây dựng các bảng liên quan:

**Đặc trưng được lựa chọn đáng chú ý**

- outputevents: rate liên quan tới nước tiểu
- inputevents: Các loại thuốc Thuốc vận mạch / Tăng co bóp cơ tim (Vasopressors / Inotropes) (Norepinephrine, Epinephrine, Dopamine, Phenylephrine..) - Tổng dịch vào
- procedureevents: Thở máy xâm lấn, Lọc mấu, v.v.v

Các bảng này sau khi lựa chọn dặc trưng sẽ được:
- Tính toán số lượng
- Lọc nhiễu
- Dùng làm nguyên liệu để sinh ra đặc trưng mới 

> Câu hỏi: Có nên thêm bảng `diagnoses_icd` nhằm trích xuất bệnh nền tiển sử trong quá khứ 

### 2. Nhiệm vụ của ngôn ngữ lớn

Sau khi đã có các thành phần bảng chính. Ngôn ngữ lớn sẽ thực hiện việc sử dụng các tài liệu như: 
- Tài liệu hướng dẫn lân sàng
- Tài liệu thang điểm tiên lượng
- Tài  liệu cấu trúc dữ liệu các bảng chính

Kết hợp cùng kiến trúc RAG. Nó sẽ thực hiện việc tóm tắt -> Chuyển đổi các tài liệu bằng chữ thành một Logic cấu trúc kết hợp với nhận diện lỗi sai tự động chuyển hóa, tính toán đặc trưng mới dựa trên những tài liệu được cung cấp 

```
PROMPT: Dựa DUY NHẤT vào tài liệu y khoa được cung cấp ở trên, hãy viết hàm Python (Pandas) để tạo các biến phái sinh từ dữ liệu MIMIC. Không tự ý thêm ngưỡng hoặc công thức không có trong tài liệu.
OUTPUT: LLM sẽ tạo đặc trưng mới dựa theo bảng đã được cung cấp trích xuất từ tài liệu đã cung cấo 
```

Sau khi mô hình ngôn ngữ lớn thực hiện nhiệm vụ của mình ta sẽ có một dữ liệu mới 

### 3. Chọn lọc đặc trưng

Sau khi đã thực hiện thủ công việc tính min, max, mean kết hợp cùng output mà ngôn ngữ lớn đã tạo. Ta sẽ có bước chuyển sang mô hình đánh giá tìm ra xem biến nào có ảnh hướng đến nhãn mục tiêu. Thực hiện chọn lọc loại bỏ các cột bị trùng. Cuối cùng ta đã có một bảng cuối sạch, đẹp vàn sẵn sàng 
đẻ cho việc xây dựng dự đoán sử dụng. Cuối cùng sau khi thực hiện huấn luyện xong ta sẽ dánh giá cũng như giải thích bằng SHAP nhằm chứng mình:
- Các đặc trưng do LLM tạo ra nằm ở top đầu về tầm quan trọng (Feature Importance).
- Sự kết hợp của đặc trưng do LLM tạo ra giúp cải thiện chỉ số AUROC và AUPRC (Precision-Recall Curve - chỉ số quan trọng cho bài toán lệch pha nhãn tử vong).

---

## Một số rủi ro cần cân nhắc và thảo luận

**Rủi ro rò rỉ dữ liệu:** 

Nếu lấy dữ liệu trong toàn bộ thời gian nằm ICU để tính Min/Max/Mean, mô hình sẽ vô tình "biết trước tương lai" (ví dụ: lấy các chỉ số suy hô hấp/ngừng tim ngay trước 
thời điểm tử vong ở ngày thứ 5 để dự đoán tử vong).Giải pháp: Bắt buộc phải cố định Cửa sổ quan sát. 
Chỉ trích xuất dữ liệu trong 24 giờ đầu tiên kể từ khi vào ICU ($t_{intime} \rightarrow t_{intime} + 24h$) để dự đoán tỷ lệ tử vong trong viện (In-Hospital Mortality) hoặc tử vong 28 ngày.
