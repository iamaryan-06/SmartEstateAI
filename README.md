# SmartEstate AI — v2 (Construction-Cost-Aware)

Real estate price prediction app for **Bangalore** and **Indore** using Gradient Boosting ML models trained on **2026 market data** with **11 cost features** including construction costs, architect/engineer fees, approval charges, GST, and more.

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Model | Linear Regression | Gradient Boosting (400 trees) |
| Features | 4 (area, bhk, bath, location) | 11 (+ construction cost, architect fees, engineer fees, approvals, utilities, GST, resale) |
| R² Score (Bangalore) | 0.51 | 0.92 |
| R² Score (Indore) | 0.78 | 0.96 |
| Price Data | Old Kaggle data | 2026 market prices (99acres, NoBroker, Coldwell Banker) |
| Construction Costs | Not included | JK Cement, Houseyog, Construction Estimator India 2026 |
| Cost Breakdown | No | Full 7-component breakdown shown |
| Locations | 254 + 35 | 41 + 36 (curated, real zones) |

## Cost Components Modeled

1. **Land Cost + Builder Margin** — varies by location zone (ultra-premium to affordable)
2. **Construction Cost** (₹/sqft) — materials + labour, ranges from ₹1,200 to ₹4,500/sqft
3. **Architect / Design Fees** — 3-7% of construction cost
4. **Structural Engineer Fees** — 1.5-3% of construction cost
5. **Approval / Mapping / Statutory** — BBMP/IMC permits, RERA, fire NOC (₹0.8-5 Lakhs)
6. **Utility Connections** — water, electricity, sewage (₹0.4-2 Lakhs)
7. **GST & Taxes** — 5% standard, 1% affordable housing

## Quick Start

```bash
# Install dependencies
pip install flask scikit-learn pandas numpy

# (Optional) Retrain the model
python train_model.py

# Run the app
python app.py
# Open http://127.0.0.1:5000
```

## Project Structure

```
real-estate-ai/
├── app.py                          # Flask backend with /predict API
├── train_model.py                  # Model training script (v2)
├── models.pkl                      # Trained model bundle
├── locations.json                  # Location lists per city
├── requirements.txt
├── data/
│   ├── generate_bangalore_data.py  # Bangalore data generator (2026 prices)
│   ├── generate_indore_data_v2.py  # Indore data generator (2026 prices)
│   ├── Bengaluru_House_Data_v2.csv # 3,280 rows, 41 locations
│   ├── Indore_House_Data_v2.csv    # 2,340 rows, 36 locations
│   └── (old v1 data files)
├── templates/
│   └── index.html                  # Full frontend with cost breakdown UI
└── static/
    └── images/                     # Property images
```

## API Response Example

```json
POST /predict
{
  "city": "Bangalore",
  "location": "Koramangala",
  "area": 2000,
  "bhk": 3,
  "bathrooms": 3
}

Response:
{
  "price_lakhs": 324.0,
  "price_inr": 32400000,
  "price_psqft": 16200,
  "zone": "ultra_premium",
  "construction_cost_psqft": 3638,
  "breakdown": {
    "land_and_margin":  { "amount_lakhs": 155.2, "label": "Land Cost + Builder Margin" },
    "construction":     { "amount_lakhs": 116.8, "label": "Construction (Materials + Labour)" },
    "architect_fee":    { "amount_lakhs": 5.8,   "label": "Architect / Design Fees" },
    "engineer_fee":     { "amount_lakhs": 2.9,   "label": "Structural Engineer Fees" },
    "approval_fee":     { "amount_lakhs": 3.5,   "label": "Approvals / Mapping / Statutory" },
    "utility_cost":     { "amount_lakhs": 1.5,   "label": "Utility Connections" },
    "gst":              { "amount_lakhs": 5.8,   "label": "GST & Taxes" }
  }
}
```

## Data Sources (2026)

- **Property prices**: 99acres, NoBroker, Coldwell Banker, Square Yards (Mar 2026)
- **Construction costs**: JK Cement, Houseyog, Kenza TMT, Construction Estimator India
- **Architect fees**: Council of Architecture India, Houseyog
- **Approval costs**: BBMP/BDA (Bangalore), IMC (Indore) fee schedules
- **GST rates**: Government of India (5% standard, 1% affordable)

## Tech Stack

- **Frontend**: HTML5, Tailwind CSS, JavaScript (Fetch API)
- **Backend**: Python 3, Flask
- **ML**: scikit-learn (GradientBoostingRegressor), Pandas, NumPy
