# 🛒 Retail Sales Forecasting

A machine learning project that forecasts daily retail sales across 10 stores and 50 items using LightGBM, with an interactive Streamlit dashboard to explore the results.

---

## 📊 Dataset

| File | Description |
|---|---|
| `train.csv` | 5 years of daily sales (2013–2017) · 913K rows |
| `test.csv` | Q1 2018 prediction inputs · 45K rows |
| `submission.csv` | Model forecast output |
| `sample_submission.csv` | Expected submission format |

**Schema:** `date · store (1–10) · item (1–50) · sales`

---

## 🧠 Model

**Algorithm:** LightGBM (`regression_l1` / MAE loss)

**Feature Engineering:**
- Calendar features — month, day of week, day of month, day of year, is_weekend
- `sales_lag_365` — sales exactly 1 year ago (captures seasonality without leakage)
- `sales_roll_mean_90_lag_365` — 90-day rolling mean lagged 1 year (trend signal)
- Target log-transformed with `np.log1p` for training, inverse-transformed at prediction

**Train / Validation Split:**
- Train: 2013–2016
- Validation: 2017
- Test (predict): Jan–Mar 2018

**Result:** Validation RMSE **8.20** · Best iteration at **861 trees** (early stopping)

---

## 🖥️ Streamlit Dashboard

Interactive frontend to explore historical sales and model forecasts.

**Features:**
- Filter by store and item
- Sales history + forecast line chart
- Monthly average bar chart
- Day-of-week sales pattern
- Store × Item forecast heatmap
- Raw forecast data table

**Run locally:**

```bash
streamlit run app.py
```

Then open **http://localhost:8501**

---

## 🚀 Setup

```bash
# Clone the repo
git clone https://github.com/kr-aashay/Retail-Sales-Forcasting.git
cd Retail-Sales-Forcasting

# Install dependencies
pip install lightgbm streamlit plotly pandas numpy scikit-learn matplotlib

# Launch dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py              # Streamlit dashboard
├── forcast.ipynb       # Model training notebook
├── train.csv           # Historical sales data
├── test.csv            # Forecast input
├── submission.csv      # Forecast output
└── sample_submission.csv
```
