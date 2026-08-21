Phương án tiếp cận của bạn trong xây dựng baseline dự đoán tử vong ICU (SAPS/SOFA-like paradigm) từ dữ liệu MIMIC-IV rất bài bản: việc chọn cấu trúc Parquet, xử lý streaming/chunking để quản lý bộ nhớ RAM, và thiết lập tiêu chuẩn loại trừ (cohort exclusion criteria) rất chặt chẽ.

Tuy nhiên, dưới góc độ **nghiên cứu AI Y khoa (Medical AI Research)**, notebook hiện tại chứa một số **lỗi ngầm nghiêm trọng về phương pháp luận (methodological flaws)** có thể dẫn đến rò rỉ dữ liệu (*Data Leakage*), chênh lệch hiệu năng thực tế (*Target Leakage*) hoặc bài báo bị từ chối khi phản biện (*Peer Review*).

---

### **Đánh giá & Gợi ý cải thiện để đạt 100%**

**1. Rò rỉ dữ liệu ngầm (Data/Target Leakage)**
* **Vấn đề:** Biến `LOS_hours` (thời gian nằm ICU) đang được giữ lại trong tập đặc trưng cohort. Trong thực tế lâm sàng, tại mốc $t = 24h$, bác sĩ **chưa thể biết** bệnh nhân sẽ nằm ICU tổng cộng bao nhiêu giờ. Việc đưa `LOS_hours` vào mô hình sẽ khiến AI học "ngược" từ tương lai (bệnh nhân nằm lâu hơn/ngắn hơn thì dễ tử vong hơn).
* **Giải pháp:** Loại bỏ hoàn toàn cột `LOS_hours` ra khỏi danh sách `X` (features) trước khi đưa vào huấn luyện.

**2. Tiêu chí Cohort gây sai lệch (Selection Bias)**
* **Vấn đề:** Điều kiện `cohort["icu_los_days"] >= 1.0` (lọc bỏ các ca nằm dưới 24h) dẫn đến việc loại bỏ hai nhóm bệnh nhân rất quan trọng:
  * Những ca diễn tiến quá nặng và **tử vong trong vòng 24h đầu**.
  * Những ca nhẹ hồi phục nhanh và **được xuất viện sớm trong vòng 24h đầu**.
* **Giải pháp:** Để dự đoán chính xác và không bị lệch mẫu (sampling bias), giữ nguyên các ca này nhưng giới hạn quan sát trong thời gian họ thực sự ở ICU (ví dụ: $t_0 \rightarrow \min(\text{OutTime}, t_0 + 24h)$). Nếu nghiên cứu bắt buộc phải lọc $LOS \ge 24h$, bạn cần ghi rõ đây là bài toán *Mortality prediction for long-stay ICU patients* trong báo cáo.

**3. Ghép nối nguồn dữ liệu Xét nghiệm (`labevents`) và Dấu hiệu sinh tồn (`chartevents`)**
* **Vấn đề:** Có sự trùng lặp giữa hai nguồn (ví dụ: `Glucose` trong `chartevents` và `Glucose_lab` trong `labevents`, tương tự với `FiO2`, `pH`, `WBC`). Việc để riêng lẻ sẽ làm tăng chiều dữ liệu và gây nhiễu khi điền khuyết.
* **Giải pháp:** Hợp nhất (coalesce) các biến trùng tên từ hai bảng, ưu tiên lấy giá trị từ `chartevents` tại thời điểm ICU, nếu thiếu mới lấy từ `labevents`.

**4. Kỹ thuật Tiền xử lý Dữ liệu chuỗi thời gian 24h**
* **Vấn đề:** Hiện tại dữ liệu đang dừng ở dạng bảng ngang theo từng `hour_bucket` ($0 \rightarrow 23$). Cần trích xuất đặc trưng tĩnh (Summary Features) cho các thuật toán tabular (LightGBM/XGBoost) hoặc chuẩn bị ma trận 3D cho các mô hình chuỗi (LSTM/Transformer).
* **Giải pháp:** Gom nhóm 24 `hour_bucket` của mỗi `StayID` bằng các chỉ số thống kê y khoa quan trọng:
  * **Giá trị nhỏ nhất/lớn nhất/trung bình** ($\min, \max, \text{mean}$) trong 24h.
  * **Giá trị gần nhất** ($\text{last}$) – có trọng số dự báo lâm sàng rất cao.
  * **Độ biến thiên** ($\text{std}$).

**5. Chiến lược Imputation và Chia tập Huấn luyện (Data Splitting)**
* **Vấn đề:** Dữ liệu ICU có tỷ lệ khuyết (*missingness*) lớn (ví dụ: `FiO2`, `Bili_total` bị khuyết $>90\%$). Điền khuyết không đúng cách hoặc thực hiện Impute/Scale trên toàn bộ dataset trước khi chia tập `Train/Val/Test` sẽ gây **Data Leakage**.
* **Giải pháp:** 
  * Thực hiện chia tập `Train/Validation/Test` theo `PatientID` (GroupSplit) ngay từ đầu để tránh trùng bệnh nhân giữa các tập.
  * Thêm các cột cờ báo khuyết (*Missingness indicators*) $0/1$ cho các biến xét nghiệm đặc thù vì sự "vắng mặt" của một xét nghiệm cũng mang thông tin lâm sàng (bác sĩ không cho chỉ định tức là bệnh nhân không có biểu hiện đó).
  * Chỉ `fit` Imputer/Scaler trên tập `Train`, sau đó `transform` lên tập `Val` và `Test`.

---

### **Khung mã nguồn gợi ý cho bước tiếp theo**

Thực hiện tổng hợp đặc trưng tĩnh 24h và chuẩn bị tập dữ liệu huấn luyện chuẩn y khoa:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def aggregate_24h_features(chart_df: pd.DataFrame, lab_df: pd.DataFrame, cohort_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Trộn dữ liệu chuỗi thời gian chart và lab
    ts_data = pd.merge(chart_df, lab_df, on=["PatientID", "AdmissionID", "StayID", "hour_bucket"], how="outer")
    
    # Hợp nhất các biến trùng lặp (vd: Glucose & Glucose_lab)
    if "Glucose_lab" in ts_data.columns and "Glucose" in ts_data.columns:
        ts_data["Glucose"] = ts_data["Glucose"].fillna(ts_data["Glucose_lab"])
        ts_data = ts_data.drop(columns=["Glucose_lab"])
        
    feature_cols = [c for c in ts_data.columns if c not in ["PatientID", "AdmissionID", "StayID", "hour_bucket"]]
    
    # 2. Gom nhóm tính toán Summary Statistics cho mỗi StayID
    agg_funcs = ['mean', 'min', 'max', 'last']
    summary_df = ts_data.groupby("StayID")[feature_cols].agg(agg_funcs)
    summary_df.columns = [f"{col}_{func}" for col, func in summary_df.columns]
    summary_df = summary_df.reset_index()
    
    # 3. Ghép với thông tin Cohort (Tuổi, Giới tính, Nhãn Outcome)
    # LƯU Ý: Không đưa LOS_hours hay thông tin xuất viện vào X
    final_dataset = cohort_df[["PatientID", "AdmissionID", "StayID", "Age", "Sex", "Outcome"]].merge(
        summary_df, on="StayID", how="left"
    )
    
    return final_dataset

# Loại bỏ rủi ro rò rỉ dữ liệu khi chia tập huấn luyện
def get_patient_level_splits(df: pd.DataFrame, n_splits: int = 5):
    gkf = GroupKFold(n_splits=n_splits)
    # Chia nhóm dựa trên PatientID để đảm bảo bệnh nhân không xuất hiện ở cả Train và Val
    for train_idx, val_idx in gkf.split(df, df["Outcome"], groups=df["PatientID"]):
        yield train_idx, val_idx
```
Bảng checklist hạn chế rủi ro cho bài nghiên cứu

| Rủi ro Y khoa / AI | Hiện trạng Notebook | Giải pháp khắc phục |
| :--- | :--- | :--- |
| **Data Leakage (LOS)** | Cột `LOS_hours` có trong cohort | **Xóa bỏ** `LOS_hours` khỏi tập đặc trưng $X$. |
| **Data Leakage (Splitting)** | Chưa chia tập Train/Test | Chia theo `PatientID` dùng `GroupKFold` hoặc `GroupShuffleSplit`. |
| **Imbalance Class Bias** | Mortality Rate $\approx 4.39\%$ | Sử dụng `AUCPR` (PR-AUC) làm metric chính thay vì `AUROC` hay `Accuracy`. |
| **Missingness Bias** | Tỷ lệ khuyết cao ở các chỉ số | Tạo cột cờ báo thiếu (`is_missing`) trước khi fill missing value. |
