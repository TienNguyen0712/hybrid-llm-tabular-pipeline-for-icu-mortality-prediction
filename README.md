# A Hybrid Framework for ICU Mortality Prediction via Large Language Models 



> 🇻🇳 [Tiếng Việt](#-tiếng-việt) · 🇬🇧 [English](#-english)

---

## 🇬🇧 English

1. [Project Overview & Research Objectives](#1-project-overview--research-objectives)
   - [1.1 Problem Statement](#11-problem-statement)
   - [1.2 Research Objectives](#12-research-objectives)
   - [1.3 Scope & Problem Formulation](#13-scope--problem-formulation)
2. [Dataset Design & Extraction Methodology (MIMIC-IV)](#2-dataset-design--extraction-methodology-mimic-iv)
   - [2.1 Patient Cohort Selection Criteria](#21-patient-cohort-selection-criteria)
     - [Initial Table Join Schema](#initial-table-join-schema)
   - [2.2 Feature Extraction by Clinical Category](#22-feature-extraction-by-clinical-category)
     - [A. Bedside Monitoring Data (`chartevents`)](#a-bedside-monitoring-data-chartevents)
     - [B. Laboratory Test Data (`labevents`)](#b-laboratory-test-data-labevents)
     - [C. Medical Interventions & Fluid Balance (`inputevents`, `outputevents`, `procedureevents`)](#c-medical-interventions--fluid-balance-inputevents-outputevents-procedureevents)
   - [2.3 Preprocessing Pipeline & Feature Standardization](#23-preprocessing-pipeline--feature-standardization)
3. [Automated Feature Generation Architecture via LLM + RAG](#3-automated-feature-generation-architecture-via-llm--rag)
   - [3.1 Medical Knowledge Integration Workflow](#31-medical-knowledge-integration-workflow)
   - [3.2 Scope of Complex Feature Generation](#32-scope-of-complex-feature-generation)
   - [3.3 Automated Code Testing & Validation Mechanism](#33-automated-code-testing--validation-mechanism)
4. [Feature Selection, Model Training & Explainability](#4-feature-selection-model-training--explainability)

---

## 1. Project Overview & Research Objectives

### 1.1 Problem Statement
Early identification of mortality risk among patients admitted to the Intensive Care Unit (ICU) is crucial for patient triage, clinical decision support, and medical resource optimization. Traditional machine learning techniques rely heavily on manual feature engineering and selection, which is both time-consuming and demanding for domain experts.

This project proposes a framework that leverages **Large Language Models (LLMs)** combined with **Retrieval-Augmented Generation (RAG)** to automate the extraction, synthesis, and discovery of complex clinical features from heterogeneous electronic health records.

### 1.2 Research Objectives
- **Automate Feature Engineering:** Utilize LLM + RAG to convert clinical guidelines into clinically meaningful input features.
- **Optimize Expert Resource Allocation:** Minimize manual labor and clinicians' time commitment during the data preprocessing phase.
- **Enhance Performance & Interpretability:** Leverage complex composite indicators to improve predictive accuracy while enabling clear traceability of clinical logic.

### 1.3 Scope & Problem Formulation
- **Dataset Used:** MIMIC-IV (`hosp` and `icu` modules).
- **Fixed Observation Window:** Extract data recorded strictly within the **first 24 hours** of ICU admission ($t_{\text{intime}} \to t_{\text{intime}} + 24\text{h}$).
- **Target Label:** In-Hospital Mortality.

[Back to top](#top)

---

## 2. Dataset Design & Extraction Methodology (MIMIC-IV)

The extraction pipeline retrieves data from 9 core tables in MIMIC-IV:
- `patients`, `admissions`, `icustays`: Administrative data, demographics, and hospital/ICU admission management.
- `chartevents`, `labevents`: Continuous vital signs and clinical laboratory test results.
- `prescriptions`, `inputevents`, `outputevents`, `procedureevents`: Medical interventions (medications, fluid balance, bedside procedures).

---

### 2.1 Patient Cohort Selection Criteria

To ensure standardization and prevent data leakage or model degeneration, the selected cohort must satisfy all of the following conditions simultaneously:

1. **Adult Patients:** Age at admission $\ge 18$.
2. **First ICU Stay:** Include only the first ICU stay per hospital admission to prevent duplicate event bias.
3. **Length of Stay (LOS):** $\text{LOS} \ge 24\text{h}$.
4. **Time of Death:** If mortality occurs, the timestamp of death ($\text{deathtime}$) must occur **after the initial 24-hour ICU window** ($\text{deathtime} > t_{\text{intime}} + 24\text{h}$).

> **Clinical Rationale:** Excluding patients who expire within the first 24 hours removes noise caused by extreme physiological volatility right before cardiac arrest and ensures a complete 24-hour window for model learning.

#### Initial Table Join Schema

```
patients ──(INNER JOIN)── icustays ──(INNER JOIN)── admissions
└─ Khoá chính: subject_id, hadm_id, stay_id
```

- **Selected Demographic Variables:** `gender`, `anchor_age`, `anchor_year`, `insurance`, `race`.
- **Timestamp & Target Variables:** `admittime`, `dischtime`, `intime`, `outtime`, `deathtime`, `hospital_expire_flag` (Primary Target Label).

---

### 2.2 Feature Extraction by Clinical Category

#### A. Bedside Monitoring Data (`chartevents`)
Extraction of continuous vital signs and clinical assessment scores:
- **Core Vitals:** Heart Rate, Respiratory Rate, Temperature ($^\circ\text{C}$ / $^\circ\text{F}$), Invasive/Non-Invasive Blood Pressure (Systolic, Diastolic, Mean Arterial Pressure - MAP), $\text{SpO}_2$.
- **Bedside Gas Exchange & Urgent Metabolic Indicators:** $\text{FiO}_2$, Blood $\text{pH}$, Blood Glucose.
- **Neurological Assessment:** Glasgow Coma Scale (GCS - Verbal, Motor, Eye).
- **Anthropometric Data:** Height, Daily Weight / Admission Weight.

#### B. Laboratory Test Data (`labevents`)
Extraction of central biochemistry, hematology, and arterial blood gas values:
- **Complete Blood Count (CBC):** Red Blood Cells (RBC), White Blood Cells (WBC), Hemoglobin, Hematocrit, Platelets.
- **Renal Function & Electrolytes:** Creatinine, Blood Urea Nitrogen (BUN), $\text{Na}^+$, $\text{K}^+$, $\text{Cl}^-$, $\text{HCO}_3^-$, Magnesium, Calcium (Total/Free), Phosphate.
- **Liver Function:** ALT, AST, Total Bilirubin, Albumin, Alkaline Phosphatase, Total Protein.
- **Cardiac Biomarkers:** Troponin T, CK-MB, Total CK, NT-proBNP.
- **Arterial Blood Gas (ABG) & Metabolism:** $\text{pO}_2$, $\text{pCO}_2$, Base Excess, Lactate, Anion Gap.
- **Endocrine Profile:** TSH, Free T4, T3, Cortisol.

#### C. Medical Interventions & Fluid Balance (`inputevents`, `outputevents`, `procedureevents`)
- **Fluid Balance:** Total fluid intake volume (`inputevents`), Total urine output volume (`outputevents`).
- **Vasoactive & Inotropic Agents:** Norepinephrine, Epinephrine, Dopamine, Phenylephrine, Vasopressin.
- **Invasive Procedures:** Invasive Mechanical Ventilation, Continuous Renal Replacement Therapy (CRRT), Endotracheal Intubation.

---

### 2.3 Preprocessing Pipeline & Feature Standardization

Each time-series indicator within the first 24-hour window is transformed into summary statistics:
1. **Noise Filtering & Unit Standardization:** Remove physiologically impossible values (outliers) and convert units into standard metrics (e.g., convert $^\circ\text{F} \to ^\circ\text{C}$).
2. **Summary Statistics Aggregation:** Calculate $\text{Min}$, $\text{Max}$, $\text{Mean}$, $\text{First Value}$ (admission value), and $\text{Last Value}$ (value at the 24-hour mark).
3. **Dynamic Features (Delta/Slope):**
   $$\Delta X = \bar{X}_{(12\text{h} \to 24\text{h})} - \bar{X}_{(0\text{h} \to 12\text{h})}$$
4. **Data Pivoting:** Reshape data structure from long format to wide format corresponding to each `stay_id`.

[Back to top](#top)

---

## 3. Automated Feature Generation Architecture via LLM + RAG

```
┌────────────────────────┐      ┌───────────────────────────┐
│ Dynamic Data Schema    │      │ Clinical Guidelines &     │
│ (MIMIC-IV Columns)     │      │ Scoring Systems Docs      │
└───────────┬────────────┘      └─────────────┬─────────────┘
            │                                 │
            └───────────────┐ ┌───────────────┘
                            ▼ ▼
                 ┌───────────────────────┐
                 │    LLM + RAG Engine   │
                 └──────────┬────────────┘
                            │ Generates Python Pandas Code
                            ▼
                 ┌───────────────────────┐
                 │  Code Guardrails Test │
                 └──────────┬────────────┘
                            │ Pass?
                            ▼
                 ┌───────────────────────┐
                 │ Joined Feature Matrix │
                 └───────────────────────┘
```

### 3.1 Medical Knowledge Integration Workflow
The Large Language Model (LLM) acts as a "Clinical Feature Engineer." The RAG system ingests domain knowledge from:
- Clinical practice guidelines and treatment protocols.
- Standard formulas for scoring ICU severity and prognosis ($SOFA$, $OASIS$, $SAPS\ II$).
- MIMIC-IV metadata and data dictionaries.

### 3.2 Scope of Complex Feature Generation
The LLM automatically analyzes clinical domain logic and writes Pandas code to compute advanced feature groups:

1. **Clinical Severity & Prognostic Scores:** Automatically calculates organ failure scores such as $SOFA$ and $SAPS\ II$ based on the initial 24-hour window.
2. **Medication & Interventions Interaction Indicators:**
   - `is_on_vasopressor_AND_ventilated`: Binary indicator (0/1) identifying patients requiring both hemodynamic vasopressor support and mechanical ventilation.
   - `fluid_overload_flag`: Alert flag for fluid overload ($\text{Input} > 3 \times \text{Output}$).
3. **Advanced Biochemical Composite Features:**
   - $\text{PaO}_2 / \text{FiO}_2$ Ratio (P/F Ratio for assessing ARDS and acute respiratory failure).
   - $\text{BUN} / \text{Creatinine}$ Ratio (Differentiates pre-renal from intrinsic acute kidney injury).
   - Corrected Anion Gap: $\text{Anion Gap} = (\text{Na}^+ + \text{K}^+) - (\text{Cl}^- + \text{HCO}_3^-)$.

### 3.3 Automated Code Testing & Validation Mechanism (Code Guardrails)
All Python code generated by the LLM must pass through an automated testing pipeline before execution across the entire dataset.

```python
# System Prompt for Code Generation Quality Control
PROMPT_TEMPLATE = """
Based on the following clinical guideline document:
---
{CLINICAL_GUIDELINE_DOC}
---
Write a Python function using the Pandas library with the following signature:
`def generate_features(df: pd.DataFrame) -> pd.DataFrame:`

Mandatory technical requirements:
1. Use ONLY existing columns present in the table schema: {AVAILABLE_COLUMNS}.
2. Explicitly handle division-by-zero exceptions (ZeroDivisionError) by returning np.nan.
3. Preserve the exact row count of the input Dataframe (Do NOT use dropna in a way that removes records).
4. Return a new Dataframe containing ONLY the 'stay_id' column and the newly generated feature columns.
"""
```

[Back to top](#top)

---

## 4. Feature Selection, Model Training & Explainability
1. **Feature Selection:** Integrate hand-crafted baseline features with LLM-generated features. Apply variable filtering methods based on correlation analysis, L1-regularization (Lasso), or Tree-based feature importance to eliminate multicollinearity.
2. **Predictive Model Training:** Train Gradient Boosting algorithms (XGBoost, LightGBM, CatBoost) on the finalized feature matrix.
3. **Model Evaluation & Explainability**:
   - Evaluate model performance using **AUROC** and **AUPRC** (the core benchmark metrics for class-imbalanced medical datasets).
   - Compute **SHAP** values to analyze the relative predictive contribution of LLM-generated features compared to traditional vital signs.

[Back to top](#top)

---

## 🇻🇳 Tiếng Việt

1. [Tổng quan Đề tài & Mục tiêu Nghiên cứu](#1-tổng-quan-đề-tài--mục-tiêu-nghiên-cứu)
   - [1.1 Đặt vấn đề](#11-đặt-vấn-đề)
   - [1.2 Mục tiêu nghiên cứu](#12-mục-tiêu-nghiên-cứu)
   - [1.3 Phạm vi & Tiêu chí bài toán](#13-phạm-vi--tiêu-chí-bài-toán)
2. [Thiết kế Bộ dữ liệu & Phương pháp Trích xuất (MIMIC-IV)](#2-thiết-kế-bộ-dữ-liệu--phương-pháp-trích-xuất-mimic-iv)
   - [2.1 Tiêu chí lựa chọn mẫu bệnh nhân](#21-tiêu-chí-lựa-chọn-mẫu-bệnh-nhân)
     - [Sơ đồ kết nối bảng khởi tạo](#sơ-đồ-kết-nối-bảng-khởi-tạo)
   - [2.2 Trích xuất đặc trưng theo nhóm dữ liệu lâm sàng](#22-trích-xuất-đặc-trưng-theo-nhóm-dữ-liệu-lâm-sàng)
     - [A. Dữ liệu Theo dõi tại giường (`chartevents`)](#a-dữ-liệu-theo-dõi-tại-giường-chartevents)
     - [B. Dữ liệu Xét nghiệm Cận lâm sàng (`labevents`)](#b-dữ-liệu-xét-nghiệm-cận-lâm-sàng-labevents)
     - [C. Can thiệp Y tế & Xuất nhập dịch (`inputevents`, `outputevents`, `procedureevents`)](#c-can-thiệp-y-tế--xuất-nhập-dịch-inputevents-outputevents-procedureevents)
   - [2.3 Quy trình Tiền xử lý & Chuẩn hóa Đặc trưng](#23-quy-trình-tiền-xử-lý--chuẩn-hóa-đặc-trưng)
3. [Kiến trúc Sinh Đặc trưng Tự động bằng LLM + RAG](#33-kiến-trúc-sinh-đặc-trưng-tự-động-bằng-llm--rag)
   - [3.1 Quy trình tích hợp Tri thức Y khoa](#31-quy-trình-tích-hợp-tri-thức-y-khoa)
   - [3.2 Phạm vi sinh đặc trưng phức hợp](#32-phạm-vi-sinh-đặc-trưng-phức-hợp)
   - [3.3 Cơ chế Tự động Kiểm thử & Xác thực Code](#33-cơ-chế-tự-động-kiểm-thử--xác-thực-code)
4. [Lựa chọn Đặc trưng, Huấn luyện & Giải thích Mô hình](#4-lựa-chọn-đặc-trưng-huấn-luyện--giải-thích-mô-hình)

---

## 1. Tổng quan Đề tài & Mục tiêu Nghiên cứu

### 1.1 Đặt vấn đề
Dự đoán sớm rủi ro tử vong của bệnh nhân khi nhập khoa Hồi sức tích cực (ICU) đóng vai trò then chốt trong việc phân loại bệnh nhân, hỗ trợ ra quyết định lâm sàng và tối ưu hóa nguồn lực y tế. Các kỹ thuật Machine Learning truyền thống đòi hỏi quá trình chọn lọc và xây dựng đặc trưng thủ công, tiêu tốn nhiều thời gian và công sức của bác sĩ y khoa. 

Đề tài đề xuất giải pháp ứng dụng **Mô hình ngôn ngữ lớn (LLM)** kết hợp với **Hệ thống Truy xuất Tăng cường Tạo mới (RAG)** nhằm tự động hóa quy trình trích xuất, tổng hợp và khai phá các đặc trưng lâm sàng chuyên sâu từ dữ liệu y tế phức tạp.

### 1.2 Mục tiêu nghiên cứu
- **Tự động hóa Feature Engineering:** Dùng LLM + RAG chuyển hóa các hướng dẫn lâm sàng thành các đặc trưng đầu vào có ý nghĩa y khoa.
- **Tối ưu hóa nguồn lực:** Giảm thiểu chi phí nhân công và thời gian tham gia của các chuyên gia y tế trong bước tiền xử lý dữ liệu.
- **Nâng cao hiệu năng & Tính giải thích:** Khai thác các chỉ số phức hợp giúp cải thiện độ chính xác dự đoán và dễ dàng truy xuất nguồn gốc logic lâm sàng.

### 1.3 Phạm vi & Tiêu chí bài toán
- **Bộ dữ liệu sử dụng:** MIMIC-IV (phân khúc module `hosp` và `icu`).
- **Cố định cửa sổ quan sát (Observation Window):** Chỉ trích xuất dữ liệu ghi nhận trong **24 giờ đầu tiên** kể từ thời điểm vào ICU ($t_{\text{intime}} \to t_{\text{intime}} + 24\text{h}$).
- **Nhãn mục tiêu dự đoán:** Tử vong trong đợt nằm viện này (*In-Hospital Mortality*).

[Về trang đầu](#top)

---

## 2. Thiết kế Bộ dữ liệu & Phương pháp Trích xuất (MIMIC-IV)

Quá trình trích xuất khai thác dữ liệu từ 9 bảng chính của MIMIC-IV:
- `patients`, `admissions`, `icustays`: Dữ liệu hành chính, nhân khẩu học và quản lý lượt nằm viện/ICU.
- `chartevents`, `labevents`: Dữ liệu sinh hiệu continuous và kết quả xét nghiệm cận lâm sàng.
- `prescriptions`, `inputevents`, `outputevents`, `procedureevents`: Dữ liệu can thiệp y tế (dùng thuốc, cân bằng xuất nhập dịch, thủ thuật hồi sức).

---

### 2.1 Tiêu chí lựa chọn mẫu bệnh nhân

Để đảm bảo tính chuẩn hóa và tránh suy biến mô hình, mẫu bệnh nhân chọn lọc cần thỏa mãn đồng thời các điều kiện sau:

1. **Người trưởng thành:** Tuổi tại thời điểm nhập viện $\ge 18$.
2. **Lần vào ICU đầu tiên:** Chỉ lấy lượt nằm ICU đầu tiên trong đợt nhập viện để tránh trùng lặp biến cố.
3. **Thời gian nằm ICU (LOS):** $\text{LOS} \ge 24\text{h}$.
4. **Thời điểm tử vong:** Nếu tử vong, thời điểm tử vong ($\text{deathtime}$) phải diễn ra **sau 24 giờ đầu tiên** ở ICU ($\text{deathtime} > t_{\text{intime}} + 24\text{h}$).

> **Giải thích lâm sàng:** Việc loại trừ bệnh nhân tử vong trong vòng 24 giờ đầu giúp loại bỏ nhiễu do các chỉ số biến động cực đoan ngay trước khi ngưng tim, đồng thời đảm bảo đủ cửa sổ dữ liệu 24h để mô hình học.

#### Sơ đồ kết nối bảng khởi tạo

```
patients ──(INNER JOIN)── icustays ──(INNER JOIN)── admissions
└─ Khoá chính: subject_id, hadm_id, stay_id
```

- **Biến nhân khẩu học chọn lọc:** `gender`, `anchor_age`, `anchor_year`, `insurance`, `race`.
- **Biến mốc thời gian & nhãn:** `admittime`, `dischtime`, `intime`, `outtime`, `deathtime`, `hospital_expire_flag` (Nhãn mục tiêu chính).

---

### 2.2 Trích xuất đặc trưng theo nhóm dữ liệu lâm sàng

#### A. Dữ liệu Theo dõi tại giường (`chartevents`)
Trích xuất các biến sinh hiệu liên tục và thang điểm theo dõi lâm sàng:
- **Sinh hiệu cơ bản:** Nhịp tim, Nhịp thở, Nhiệt độ ($^\circ\text{C}$ / $^\circ\text{F}$), Huyết áp xâm lấn/không xâm lấn (Tâm thu, Tâm trương, Trung bình - MAP), $\text{SpO}_2$.
- **Khí máu & Chuyển hóa khẩn cấp tại giường:** $\text{FiO}_2$, $\text{pH}$ máu, Glucose máu.
- **Thang điểm tri giác:** Glasgow Coma Scale (GCS - Verbal, Motor, Eye).
- **Nhân trắc học:** Chiều cao, Cân nặng hàng ngày / Cân nặng nhập viện.

#### B. Dữ liệu Xét nghiệm Cận lâm sàng (`labevents`)
Trích xuất các chỉ số hóa sinh, huyết học và khí máu trung tâm:
- **Công thức máu (CBC):** Hồng cầu (RBC), Bạch cầu (WBC), Huyết sắc tố (Hemoglobin), Hematocrit, Tiểu cầu (Platelets).
- **Chức năng thận & Điện giải:** Creatinine, BUN, $\text{Na}^+$, $\text{K}^+$, $\text{Cl}^-$, $\text{HCO}_3^-$, Magnesium, Calcium (Toàn phần/Tự do), Phosphate.
- **Chức năng gan:** ALT, AST, Bilirubin toàn phần, Albumin, Alkaline Phosphatase, Protein toàn phần.
- **Marker tổn thương cơ tim:** Troponin T, CK-MB, CK tổng, NT-proBNP.
- **Khí máu động mạch (ABG) & Chuyển hóa:** $\text{pO}_2$, $\text{pCO}_2$, Base Excess, Lactate, Anion Gap.
- **Nội tiết:** TSH, Free T4, T3, Cortisol.

#### C. Can thiệp Y tế & Xuất nhập dịch (`inputevents`, `outputevents`, `procedureevents`)
- **Cân bằng dịch:** Tổng lượng dịch vào (`inputevents`), Tổng thể tích nước tiểu xuất ra (`outputevents`).
- **Thuốc vận mạch & Tăng co bóp cơ tim:** Norepinephrine, Epinephrine, Dopamine, Phenylephrine, Vasopressin.
- **Thủ thuật can thiệp:** Thở máy xâm lấn (Invasive Mechanical Ventilation), Lọc máu liên tục (CRRT), Đặt ống nội khí quản.

---

### 2.3 Quy trình Tiền xử lý & Chuẩn hóa Đặc trưng

Mỗi chỉ số theo thời gian trong cửa sổ 24h đầu tiên được biến đổi thành các đặc trưng thống kê:
1. **Lọc nhiễu & Chuẩn hóa đơn vị:** Loại bỏ các giá trị ngoài ngưỡng sinh lý không tưởng (Outliers), quy đổi đơn vị về hệ chuẩn (ví dụ: chuyển $^\circ\text{F} \to ^\circ\text{C}$).
2. **Biến số Thống kê Tổng hợp:** Tính $\text{Min}$, $\text{Max}$, $\text{Mean}$, $\text{First Value}$ (Giá trị khi vào khoa) và $\text{Last Value}$ (Giá trị tại mốc 24h).
3. **Biến số Động lực học (Dynamic Features - Delta/Slope):**
   $$\Delta X = \bar{X}_{(12\text{h} \to 24\text{h})} - \bar{X}_{(0\text{h} \to 12\text{h})}$$
4. **Pivot Data:** Chuyển đổi cấu trúc từ dạng dọc sang dạng ngang tương ứng với mỗi `stay_id`.

[Về trang đầu](#top)

---

## 3. Kiến trúc Sinh Đặc trưng Tự động bằng LLM + RAG

```
┌────────────────────────┐      ┌───────────────────────────┐
│ Dynamic Data Schema    │      │ Clinical Guidelines &     │
│ (MIMIC-IV Columns)     │      │ Scoring Systems Docs      │
└───────────┬────────────┘      └─────────────┬─────────────┘
            │                                 │
            └───────────────┐ ┌───────────────┘
                            ▼ ▼
                 ┌───────────────────────┐
                 │    LLM + RAG Engine   │
                 └──────────┬────────────┘
                            │ Generates Python Pandas Code
                            ▼
                 ┌───────────────────────┐
                 │  Code Guardrails Test │
                 └──────────┬────────────┘
                            │ Pass?
                            ▼
                 ┌───────────────────────┐
                 │ Joined Feature Matrix │
                 └───────────────────────┘
```

### 3.1 Quy trình tích hợp Tri thức Y khoa
Mô hình ngôn ngữ lớn (LLM) đóng vai trò là một "Kỹ sư Đặc trưng Y khoa". Hệ thống RAG nạp dữ liệu tri thức từ:
- Tài liệu hướng dẫn điều trị lâm sàng 
- Công thức tính toán các thang điểm tiên lượng ICU chuẩn ($SOFA$, $OASIS$, $SAPS\ II$).
- Metadata và Data Dictionary của MIMIC-IV.

### 3.2 Phạm vi sinh đặc trưng phức hợp
LLM tự động phân tích và viết mã code Pandas để tính toán các nhóm biến nâng cao:

1. **Thang điểm Tiên lượng Lâm sàng:** Tự động tính điểm suy cơ quan $SOFA$, $SAPS\ II$ dựa trên dữ liệu 24h đầu.
2. **Chỉ số Tương tác Thuốc & Thủ thuật:**
   - `is_on_vasopressor_AND_ventilated`: Biến nhị phân (0/1) đánh giá bệnh nhân vừa cần hỗ trợ tuần hoàn vừa cần thở máy.
   - `fluid_overload_flag`: Cảnh báo quá tải dịch ($\text{Input} > 3 \times \text{Output}$).
3. **Phức hợp Chỉ số Hóa sinh Chuyên sâu:**
   - Tỷ lệ $\text{PaO}_2 / \text{FiO}_2$ (Chỉ số P/F đánh giá ARDS/Suy hô hấp).
   - Tỷ lệ $\text{BUN} / \text{Creatinine}$ (Phân biệt suy thận cấp trước thận và tại thận).
   - Anion Gap hiệu chỉnh: $\text{Anion Gap} = (\text{Na}^+ + \text{K}^+) - (\text{Cl}^- + \text{HCO}_3^-)$.

### 3.3 Cơ chế Tự động Kiểm thử & Xác thực Code (Code Guardrails)
Mọi đoạn code do LLM khởi tạo bắt buộc phải đi qua lớp kiểm thử tự động trước khi thực thi trên toàn bộ tập dữ liệu.

```python
# System Prompt kiểm soát chất lượng sinh mã
PROMPT_TEMPLATE = """
Dựa vào tài liệu hướng dẫn lâm sàng dưới đây:
---
{CLINICAL_GUIDELINE_DOC}
---
Hãy viết một hàm Python bằng thư viện Pandas có định dạng:
`def generate_features(df: pd.DataFrame) -> pd.DataFrame:`

Yêu cầu kỹ thuật bắt buộc:
1. Chỉ sử dụng các cột hiện có trong bảng dữ liệu: {AVAILABLE_COLUMNS}.
2. Xử lý triệt để ngoại lệ chia cho 0 (ZeroDivisionError) bằng cách trả về np.nan.
3. Bảo toàn số lượng dòng của Dataframe gốc (Không dropna làm mất record).
4. Trả về Dataframe mới CHỈ chứa cột 'stay_id' và các cột đặc trưng mới vừa sinh ra.
"""
```

[Về trang đầu](#top)

---

## 4. Lựa chọn Đặc trưng, Huấn luyện & Giải thích Mô hình
1. **Feature Selection:** Tích hợp các đặc trưng thủ công và các đặc trưng do LLM khởi tạo. Sử dụng phương pháp lọc biến dựa trên độ tương quan, L1-regularization (Lasso) hoặc Tree-based feature importance để loại bỏ các biến đa cộng tuyến.
2. **Huấn luyện Mô hình Dự đoán:** Sử dụng các thuật toán Gradient Boosting (XGBoost, LightGBM, CatBoost) trên bảng dữ liệu cuối cùng.
3. **Đánh giá & Giải thích Mô hình**:
- Đánh giá mô hình bằng các chỉ số **AUROC** và **AUPRC** (Chỉ số cốt lõi đối với bài toán lệch pha nhãn - Imbalanced Data trong y tế).
- Đánh giá giá trị **SHAP** để phân tích mức độ đóng góp của các chỉ số do LLM tự động tạo ra so với các biến sinh hiệu truyền thống.

[Về trang đầu](#top)
