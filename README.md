# Dự đoán nhu cầu sản phẩm và nguy cơ hết hàng

## 1. Giới thiệu đề tài
Đề tài gồm 2 bài toán:
- Bài toán 1 (Regression): Dự đoán Demand dựa trên các yếu tố ảnh hưởng (giá, khuyến mãi, dịch bệnh, vùng miền, thời tiết,...).
- Bài toán 2 (Classification): Dự đoán nguy cơ hết hàng (Stockout).

Mục tiêu:
- Hỗ trợ doanh nghiệp ước lượng nhu cầu và phát hiện sớm rủi ro hết hàng để đưa ra quyết định nhập hàng phù hợp.
---
## 2. Dataset
Nguồn dữ liệu: https://www.kaggle.com/code/christophergspencer/retail-store-inventory-and-demand-forecasting/input

### Mô tả cột
- Date: Ngày ghi chép
- Store ID: Mã cửa hàng
- Product ID: Mã sản phẩm
- Category: Danh mục sản phẩm
- Region: Khu vực
- Inventory Level: Tồn kho hiện tại
- Units Sold: Số lượng bán ra
- Units Ordered: Số lượng đặt thêm
- Price: Giá bán
- Discount: Mức giảm giá
- Weather Condition: Thời tiết
- Promotion: Có khuyến mãi (0/1)
- Competitor Pricing: Giá đối thủ
- Seasonality: Mùa trong năm
- Epidemic: Có dịch bệnh (0/1)
- Demand: Nhu cầu ước tính

---

## 3. Pipeline
Quy trình xử lý:
1) Tiền xử lý dữ liệu: chuẩn hoá Date, kiểm tra missing/duplicate/validity  
2) Xử lý outlier theo Region (Price, Competitor Pricing, Units Ordered) bằng capping Q1–Q99  
3) EDA: phân phối, boxplot nghiệp vụ, demand theo tháng, ma trận tương quan  
4) Feature Engineering (không dùng lag):  
   - Temporal: Month, DayOfWeek, IsWeekend  
   - Price: Effective_Price, Has_Discount  
   - Event: Promo_Epidemic  
   - Label Stockout  
5) Train/Test split theo thời gian (80/20)  
6) Train -> Evaluate -> Inference
---
## 4. Mô hình sử dụng và lý do chọn
### 4.1 Ridge Regression (Demand)
- Ridge là hồi quy tuyến tính có regularization (L2), phù hợp khi dữ liệu có nhiều biến và có khả năng đa cộng tuyến.
- Dễ triển khai, dễ giải thích, phù hợp mức machine learning cơ bản.

### 4.2 Logistic Regression (Stockout)
- Mô hình baseline, đơn giản, dễ giải thích.

### 4.3 Random Forest (Stockout)
- Bắt được quan hệ phi tuyến giữa các biến và Stockout.
- Thường cho kết quả tốt hơn Logistic trong dữ liệu thực tế có tương tác phức tạp.
---
## 5. Kết quả
### 5.1 Metrics đánh giá
- Regression: MAE, RMSE  
- Classification: Precision, Recall, F1-score, Confusion Matrix  
(Lý do: bài toán Stockout có thể mất cân bằng lớp, nên ưu tiên Precision/Recall/F1 hơn Accuracy)

### 5.2 Kết quả đánh giá
**Ridge (Demand)**
- MAE: 16.837
- RMSE: 22.33
**Logistic Regression (Stockout)**
- Precision: 0.7754
- Recall: 1.0
- F1: 0.8735
- Confusion Matrix:
  [[13054, 482],
   [0, 1664]]

**Random Forest (Stockout)**
- Precision: 0.8959
- Recall: 0.7758
- F1: 0.8316
- Confusion Matrix:
  [[13386, 150],
   [373, 1291]]
Nhận xét:
- Logistic đạt Recall cao (ít bỏ sót Stockout) nhưng có nhiều False Positive hơn.
- Random Forest có Precision cao hơn (cảnh báo Stockout “chắc” hơn), giảm cảnh báo sai, phù hợp nếu ưu tiên giảm đặt hàng dư thừa.

---

## 6. Hướng dẫn chạy dự án

### 6.1 Cài môi trường
```bash
pip install -r requirements.txt
