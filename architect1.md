# Bài thuyết trình về quy trình từ dữ liệu thô cho đến một mô hình dự đoán tử vong ICU

--- 

Mục tiêu của phương pháp này là giúp đỡ hỗ trợ việc ra quyết định cho các bác sĩ đối với bệnh nhân vào viện ICU: 
- Quy trình sẽ được chia làm 3 bước chính: 
  - Bước 1: Biến dữ liệu thô thành các dữ liệu có giá trị 
  - Bước 2: Từ dữ liệu có giả trị + Thống kê + Tri thức LLM và RAG ==> Dữ liệu huấn luyện cuối cùng
  - Bước 3: Huấn luyện mô hình để ra quyết định so sánh đô hiệu quả + Giải thích kết quả và là nguồn cho bác sĩ tham khảo

- **Tổng quan:** Sử dụng dữ liệu thô két hợp việc trích xuất thông tin tài liệu từ RAG cùng hiểu rõ ngữ cảnh các biến định tính của LLM để tạo lựa chọn đặc trưng cho học máy
- **Bài toán được lựa chọn:** Dự doán tỷ lệ tử vong ở ICU trong 24h dầu

---

## Bước 1: Từ dữ liệu thô đến các dữ liệu có giá trị 

**Mục tiêu:** Từ các bảng nhỏ thực hiện sinh ra các bảng mang giá trị đúng với bài toán

- Lựa chọn 10 bảng làm bảng mục tiêu 
  - hosp: patients, admission,  
  - icu: icustay, 
### 1.1. Sinh cohort

Thực hiện việc join 3 bảng patients - admission - icustay: để đáp ứng điều kiện
- Bệnh nhân phải là người trường thành (anchor_age >= 18 or age >= 18)
> Do năm nhâp viện đã dược mã hóa nên tuổi thật của bệnh nhân sẽ được tính bằng age = (anchor_age + year(icu.intime)) - anchor_year là số tuổi thực tế người đó trong icu
- Lấy lần nhập ICU là lần nhập đầu tiên
- Thời gian nằm ICU từ 24h trở đi (LOS >= 24h)
Ngoài ra cũng thực hiện việc kiểm tra tính hợp lý như:
- Tính duy nhất của khóa
- Tính hợp lệ của mốc thời gian
- Kiểm tra tính hợp lệ của nhãn mục tiêu

```python
  # 1. Inner Join với patients
  cohort = icustays_df.merge(
  admissions_df[["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                        "admission_location", "edregtime", "edouttime", "deathtime",
                        "hospital_expire_flag", "insurance", "race"]],
          on=['subject_id', 'hadm_id'],
          how='inner'
      )
  # 2. Inner join với admission
  cohort = cohort.merge(
          patients_df[["subject_id", "gender", "anchor_age", "anchor_year"]],
          on='subject_id',
          how='inner'
      )
```

### 1.2. Thực hiện việc chia train/test/val

Mục tiêu là để tránh rò rỉ khi thực hiện huấn luyện mô hình sau khi chia ta sẽ JOIN bảng train_df với các bảng sau để sinh các hiệu số lâm sàng  
- Ta sẽ chia dữ liệu cohort theo subject_id
- Với tỷ lệ là 70% train - 15% val - 15% test
  
### 1.3. Xây dựng bảng chart vitals từ bảng `chartevents` - lab test từ bảng `labevents`

Bước này thực hiện việc sinh các đặc trưng theo id đươc so sánh ở bảng `d_items` trong module. Lưu ý dược lấy trong khoảng thời gian trong điều kiện intime <= charttime <= intime + 24h
- Sau khi đã chọn được các đặc trưng. Thực hiện các bước
  - Lọc mốc thời gian 24h & tính delta_hours.
  - Quy đổi đơn vị đo (F -> C, lbs -> kg, inches -> cm).
  - Lọc nhiễu sinh lý (Outliers).
  - Trích xuất chỉ số chuỗi thời gian: min, max, mean, first, last, delta, slope, trend chuẩn bị cho nhánh sau

Các dặc trưng được lựa chọn:

- chartevnents

```python
# Định nghĩa mapping itemid
item_mapping = {
    # Sinh hiệu cơ bản & Hô hấp
    220045: "heart_rate",        # Nhịp tim (bpm)
    220210: "resp_rate",         # Nhịp thở (lần/phút)
    223762: "temp_c",            # Nhiệt độ độ C (°C)
    223761: "temp_f",            # Nhiệt độ độ F (°F)
    220277: "spo2",              # Nồng độ Oxy máu SpO2 (%)
    223835: "fio2",              # Tỷ lệ Oxy hít vào FiO2 (%)

    # Huyết áp xâm lấn (Arterial Line)
    220050: "sbp_line",          # Huyết áp tâm thu xâm lấn (mmHg)
    220051: "dbp_line",          # Huyết áp tâm trương xâm lấn (mmHg)
    220052: "mbp_line",          # Huyết áp trung bình xâm lấn (mmHg)

    # Huyết áp không xâm lấn (NIBP) - Giữ lại để tránh thiếu dữ liệu
    220179: "sbp_nibp",          # Huyết áp tâm thu NIBP (mmHg)
    220180: "dbp_nibp",          # Huyết áp tâm trương NIBP (mmHg)
    220181: "mbp_nibp",          # Huyết áp trung bình NIBP (mmHg)

    # Đường huyết
    225664: "glucose",           # Glucose máu (mg/dL)

    # Khí máu động mạch (Arterial Blood Gas)
    223830: "ph",                # pH máu động mạch
    220224: "pao2",              # Áp suất một phần Oxy PaO2 (mmHg)
    220235: "paco2",             # Áp suất một phần CO2 PaCO2 (mmHg)
    224828: "base_excess",       # Kiềm dư Base Excess (mEq/L)
    220227: "sao2",              # Độ bão hòa Oxy động mạch SaO2 (%)

    # Công thức máu (CBC)
    220228: "hemoglobin",        # Hemoglobin (g/dL)
    220545: "hematocrit",        # Hematocrit (%)
    220546: "wbc",               # Bạch cầu WBC (K/uL)
    227457: "platelet",          # Tiểu cầu Platelets (K/uL)

    # Chức năng thận
    220615: "creatinine",        # Creatinine (mg/dL)
    225624: "bun",               # Blood Urea Nitrogen (mg/dL)

    # Chức năng gan
    220644: "alt",               # Alanine Aminotransferase ALT (U/L)
    220587: "ast",               # Aspartate Aminotransferase AST (U/L)
    225690: "bilirubin_total",   # Bilirubin toàn phần (mg/dL)
    227456: "albumin",           # Albumin (g/dL)
    225612: "alp",               # Alkaline Phosphatase (U/L)

    # Thang điểm hôn mê Glasgow (GCS)
    223900: "gcs_verbal",        # GCS - Đáp ứng lời nói (1 - 5)
    223901: "gcs_motor",         # GCS - Đáp ứng vận động (1 - 6)
    220739: "gcs_eye",           # GCS - Mở mắt (1 - 4)

    # Nội tiết
    228236: "insulin",           # Insulin (uIU/mL)
    227463: "cortisol",          # Cortisol (ug/dL)

    # Trọng lượng & Chiều cao
    224639: "weight_daily",      # Cân nặng hàng ngày (kg)
    226512: "weight_admit",      # Cân nặng khi nhập viện (kg)
    226707: "height_inch",       # Chiều cao (Inches)
    226730: "height_cm"          # Chiều cao (cm)
}

# Định nghĩa khoảng outlier
outlier_bounds = {
    # Sinh hiệu & Hô hấp
    "heart_rate": (20.0, 250.0),
    "resp_rate": (4.0, 60.0),
    "temp_c": (25.0, 43.0),
    "temp_f": (77.0, 109.4),
    "spo2": (50.0, 100.0),
    "fio2": (21.0, 100.0),        # Trong chartevents, FiO2 thường lưu dạng % (21-100)

    # Huyết áp
    "sbp_line": (40.0, 280.0),
    "dbp_line": (20.0, 150.0),
    "mbp_line": (20.0, 200.0),
    "sbp_nibp": (40.0, 280.0),
    "dbp_nibp": (20.0, 150.0),
    "mbp_nibp": (20.0, 200.0),

    # Đường huyết & Khí máu
    "glucose": (10.0, 2000.0),
    "ph": (6.5, 7.8),
    "pao2": (20.0, 700.0),
    "paco2": (10.0, 150.0),
    "base_excess": (-30.0, 30.0),
    "sao2": (50.0, 100.0),

    # Huyết học (CBC)
    "hemoglobin": (2.0, 25.0),
    "hematocrit": (5.0, 75.0),
    "wbc": (0.1, 200.0),
    "platelet": (5.0, 2000.0),

    # Chức năng thận
    "creatinine": (0.1, 30.0),
    "bun": (1.0, 250.0),

    # Chức năng gan
    "alt": (1.0, 10000.0),
    "ast": (1.0, 10000.0),
    "bilirubin_total": (0.1, 80.0),
    "albumin": (0.5, 7.0),
    "alp": (5.0, 5000.0),

    # GCS
    "gcs_verbal": (1.0, 5.0),
    "gcs_motor": (1.0, 6.0),
    "gcs_eye": (1.0, 4.0),

    # Nội tiết
    "insulin": (0.1, 500.0),
    "cortisol": (0.1, 200.0),

    # Thể trạng
    "weight_daily": (20.0, 300.0),
    "weight_admit": (20.0, 300.0),
    "height_inch": (20.0, 100.0),
    "height_cm": (50.0, 250.0)
}
```
- labevnents

```python
lab_item_mapping = {
    # Sinh hiệu & Khí máu cơ bản
    50825: "temp_c_lab",          # Nhiệt độ cơ thể (°C)
    50817: "spo2_lab",            # Độ bão hòa Oxy (O2 Saturation / SpO2 %)
    50820: "ph",                  # pH máu

    # Đường huyết
    50931: "glucose_plasma",      # Glucose Huyết tương (mg/dL)
    50809: "glucose_blood",       # Glucose Máu toàn phần (mg/dL)

    # Huyết học (Công thức máu - CBC)
    51221: "hematocrit",          # Hematocrit (%)
    51222: "hemoglobin",          # Hemoglobin (g/dL)
    51301: "wbc",                 # Bạch cầu WBC (K/uL)
    51265: "platelet",            # Tiểu cầu Platelets (K/uL)
    51279: "rbc",                 # Hồng cầu RBC (m/uL)

    # Chức năng thận
    50912: "creatinine",          # Creatinine (mg/dL)
    51006: "bun",                 # Blood Urea Nitrogen (mg/dL)

    # Điện giải đồ
    50971: "potassium",           # Kali / Potassium (mEq/L)
    50983: "sodium",              # Natri / Sodium (mEq/L)
    50902: "chloride",            # Clo / Chloride (mEq/L)
    50882: "bicarbonate",         # Bicarbonate / HCO3 (mEq/L)
    50893: "calcium_total",       # Calci toàn phần (mg/dL)
    50808: "calcium_free",        # Calci ion hóa / tự do (mmol/L)
    50960: "magnesium",           # Magie (mg/dL)
    50970: "phosphate",           # Phosphate (mg/dL)

    # Khí máu động mạch / tĩnh mạch (ABG/VBG)
    50821: "po2",                 # Áp suất một phần Oxy pO2 (mmHg)
    50818: "pco2",                # Áp suất một phần CO2 pCO2 (mmHg)
    50802: "base_excess",         # Kiềm dư Base Excess (mEq/L)
    50868: "anion_gap",           # Khoảng trống Anion Gap (mEq/L)

    # Chức năng gan
    50861: "alt",                 # Alanine Aminotransferase ALT (U/L)
    50878: "ast",                 # Aspartate Aminotransferase AST (U/L)
    50863: "alp",                 # Alkaline Phosphatase (U/L)
    50885: "bilirubin_total",     # Bilirubin toàn phần (mg/dL)
    50862: "albumin",             # Albumin (g/dL)
    50976: "protein_total",       # Protein toàn phần (g/dL)

    # Men tim & Marker sinh học (Cardiac Markers)
    51003: "troponin_t",          # Troponin T (ng/mL)
    50911: "ck_mb",               # CK-MB (ng/mL)
    50910: "ck_total",            # CK tổng (U/L)
    50963: "nt_probnp",           # NT-proBNP (pg/mL)

    # Nội tiết (Endocrinology)
    50993: "tsh",                 # Thyroid-Stimulating Hormone (uIU/mL)
    50995: "free_t4",             # Free T4 (ng/dL)
    51001: "t3",                  # Triiodothyronine T3 (ng/dL)
    50909: "cortisol",            # Cortisol (ug/dL)
    50965: "pth"                  # Parathyroid Hormone PTH (pg/mL)
}

# Định nghĩa khoảng outlier
lab_outlier_bounds = {
    # Sinh hiệu & Khí máu
    "temp_c_lab": (25.0, 43.0),
    "spo2_lab": (50.0, 100.0),
    "ph": (6.5, 7.8),

    # Đường huyết
    "glucose_plasma": (10.0, 2000.0),
    "glucose_blood": (10.0, 2000.0),

    # Huyết học (CBC)
    "hematocrit": (5.0, 75.0),
    "hemoglobin": (2.0, 25.0),
    "wbc": (0.1, 200.0),
    "platelet": (5.0, 2000.0),
    "rbc": (0.5, 10.0),

    # Chức năng thận
    "creatinine": (0.1, 30.0),
    "bun": (1.0, 250.0),

    # Điện giải đồ
    "potassium": (1.0, 10.0),
    "sodium": (90.0, 180.0),
    "chloride": (60.0, 160.0),
    "bicarbonate": (2.0, 60.0),
    "calcium_total": (2.0, 20.0),
    "calcium_free": (0.2, 3.0),
    "magnesium": (0.2, 10.0),
    "phosphate": (0.2, 20.0),

    # Khí máu
    "po2": (20.0, 700.0),
    "pco2": (10.0, 150.0),
    "base_excess": (-30.0, 30.0),
    "anion_gap": (0.0, 60.0),

    # Chức năng gan
    "alt": (1.0, 10000.0),
    "ast": (1.0, 10000.0),
    "alp": (5.0, 5000.0),
    "bilirubin_total": (0.1, 80.0),
    "albumin": (0.5, 7.0),
    "protein_total": (1.0, 15.0),

    # Men tim
    "troponin_t": (0.01, 100.0),
    "ck_mb": (0.1, 1000.0),
    "ck_total": (5.0, 50000.0),
    "nt_probnp": (10.0, 100000.0),

    # Nội tiết
    "tsh": (0.01, 100.0),
    "free_t4": (0.1, 10.0),
    "t3": (10.0, 800.0),
    "cortisol": (0.1, 200.0),
    "pth": (1.0, 3000.0)
}
```





