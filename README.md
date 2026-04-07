<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Tailwind_CSS-3.x-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="Tailwind">
  <img src="https://img.shields.io/badge/Three.js-r128-000000?style=for-the-badge&logo=three.js&logoColor=white" alt="Three.js">
  <img src="https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge" alt="License">
</p>

<h1 align="center">
  <br>
  SmartEstate AI
  <br>
  <sub><sup>Hyperparameter-Tuned Real Estate Price Prediction</sup></sub>
</h1>

<p align="center">
  <strong>An advanced ML-powered property valuation engine for Bangalore & Indore, featuring confidence-bounded price ranges, local market accuracy scoring, acquisition probability estimation, and AI-generated property insights — all wrapped in a futuristic dark-mode UI with 3D tilt cards, parallax effects, and Three.js particle animations.</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> &bull;
  <a href="#-ml-architecture">ML Architecture</a> &bull;
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-api-reference">API Reference</a> &bull;
  <a href="#-project-structure">Project Structure</a> &bull;
  <a href="#-data-sources">Data Sources</a> &bull;
  <a href="#-tech-stack">Tech Stack</a> &bull;
  <a href="#-team">Team</a>
</p>

---

## What Sets This Apart

| Metric | Value |
|---|---|
| **R2 Score (Bangalore)** | 0.93 |
| **R2 Score (Indore)** | 0.96 |
| **Training Records** | 6,200+ |
| **Engineered Features** | 12 |
| **Locations Covered** | 104 |
| **Hyperparameter Tuning** | RandomizedSearchCV (30 iter, 5-fold CV) |
| **Price Range** | Dynamic +/-5-8% confidence bounds |
| **Local Market Accuracy** | 88-96% (zone-adaptive) |
| **Acquisition Probability** | 80-95% (market-condition-aware) |

---

## Features

### Intelligent Prediction Engine
- **Confidence-Bounded Price Ranges** — Not just a single number. Returns low/predicted/high with dynamically calculated +/-5-8% margins based on zone tier and historical price variance.
- **Local Market Accuracy** — A 88-96% accuracy metric computed from data density, zone classification, and price standard deviation for each micro-market.
- **Acquisition Probability** — "There is a 91% probability you can secure this property within this estimated price range." Calculated from market competition, inventory levels, and price stability signals.
- **AI Property Descriptions** — Auto-generated 2-3 sentence commercial descriptions based on zone tier, property type, size classification, price tier, and city-specific context.
- **7-Component Cost Breakdown** — Land/margin, construction, architect fees, engineer fees, approvals, utilities, GST — each scaled proportionally.

### Futuristic Frontend
- **3D Tilt Property Cards** — VanillaTilt.js-powered perspective transformations with glare effects on hover.
- **Three.js Particle Hero** — 1,500-particle system with wireframe house geometry, reactive to mouse movement.
- **Parallax Scrolling** — Multi-speed ambient glow orbs creating depth in the hero section.
- **Staggered Modal Animations** — Image clip-path reveals, content fade-in cascades, and smooth exit transitions.
- **Animated Accuracy Ring** — SVG progress ring with gradient stroke that animates to the accuracy percentage.
- **Glassmorphism & Glow Effects** — Backdrop-blur panels, animated border glows, conic-gradient card backgrounds.
- **Magnetic Buttons** — Hero CTAs subtly track cursor position for a magnetic pull effect.
- **Dark Mode** — System-aware with manual toggle, persisted to localStorage.

---

## ML Architecture

```
Input (6 user params)
  |
  v
+---------------------------------------------------+
|  Feature Engineering (12 features)                 |
|  area, bhk, bath, loc_enc, prop_type_enc,          |
|  construction_cost_psqft, architect_fee_lakhs,     |
|  engineer_fee_lakhs, approval_fee_lakhs,           |
|  utility_cost_lakhs, gst_lakhs, is_resale          |
+---------------------------------------------------+
  |
  v
+---------------------------------------------------+
|  sklearn Pipeline                                  |
|  [StandardScaler] -> [GradientBoostingRegressor]   |
|                                                    |
|  Tuned via RandomizedSearchCV:                     |
|    n_estimators:     300-800                       |
|    max_depth:        3-8                           |
|    learning_rate:    0.01-0.16                     |
|    subsample:        0.70-0.95                     |
|    min_samples_leaf: 3-15                          |
|    min_samples_split: 4-20                         |
|    max_features:     0.50-1.00                     |
|                                                    |
|  30 iterations x 5-fold CV = 150 fits              |
+---------------------------------------------------+
  |
  v
+---------------------------------------------------+
|  Post-Processing                                   |
|  - Price prediction (lakhs)                        |
|  - Price range: +/- 5-8% (zone-adjusted)           |
|  - Local market accuracy: 88-96%                   |
|  - Acquisition probability: 80-95%                 |
|  - 7-component cost breakdown                      |
|  - AI-generated property description               |
+---------------------------------------------------+
```

### Cost Components Modeled

| Component | Range | Source |
|---|---|---|
| **Land Cost + Builder Margin** | Varies by zone tier | 99acres, NoBroker |
| **Construction Cost** | Rs 1,200-4,500/sqft | JK Cement, Houseyog |
| **Architect / Design Fees** | 3-7% of construction | Council of Architecture India |
| **Structural Engineer Fees** | 1.5-3% of construction | Industry standard |
| **Approval / Statutory** | Rs 0.8-5 Lakhs | BBMP/IMC fee schedules |
| **Utility Connections** | Rs 0.4-2 Lakhs | BESCOM, BWSSB |
| **GST & Taxes** | 1-5% | Government of India |

---

## Quick Start

### Prerequisites

```bash
Python 3.10+
pip (package manager)
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/real-estate-ai.git
cd real-estate-ai

# 2. Install dependencies
pip install flask scikit-learn pandas numpy scipy

# 3. (Optional) Retrain the model with hyperparameter tuning
python train_model.py
# This runs RandomizedSearchCV (30 iterations, 5-fold CV) — takes ~2-3 minutes

# 4. Start the server
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## API Reference

### `GET /locations?city=Bangalore`

Returns available locations for a city.

```json
{
  "locations": ["Koramangala", "Whitefield", "Indiranagar", "..."]
}
```

### `POST /predict`

**Request:**
```json
{
  "city": "Bangalore",
  "location": "Koramangala",
  "area": 1800,
  "bhk": 3,
  "bathrooms": 3,
  "property_type": "flat"
}
```

**Response:**
```json
{
  "price_lakhs": 306.42,
  "price_low_lakhs": 283.44,
  "price_high_lakhs": 329.40,
  "margin_pct": 7.5,
  "price_inr": 30642000,
  "price_psqft": 17023,
  "city": "Bangalore",
  "location": "Koramangala",
  "zone": "premium",
  "local_market_accuracy": 94.2,
  "acquisition_probability": 85,
  "description": "This spacious 3-BHK apartment spanning 1,800 sq ft in Koramangala, Bangalore is situated in one of the most sought-after premium localities...",
  "breakdown": {
    "land_and_margin":  { "amount_lakhs": 155.20, "label": "Land Cost + Builder Margin" },
    "construction":     { "amount_lakhs": 98.50,  "label": "Construction (Materials + Labour)" },
    "architect_fee":    { "amount_lakhs": 6.80,   "label": "Architect / Design Fees" },
    "engineer_fee":     { "amount_lakhs": 3.40,   "label": "Structural Engineer Fees" },
    "approval_fee":     { "amount_lakhs": 3.50,   "label": "Approvals / Mapping / Statutory" },
    "utility_cost":     { "amount_lakhs": 1.50,   "label": "Utility Connections" },
    "gst":              { "amount_lakhs": 5.80,   "label": "GST & Taxes" }
  },
  "construction_cost_psqft": 3638
}
```

---

## Project Structure

```
real-estate-ai/
├── app.py                           # Flask API — prediction, accuracy, confidence
├── train_model.py                   # ML pipeline — StandardScaler + GBR + RandomizedSearchCV
├── models.pkl                       # Serialized Pipeline objects + location profiles
├── locations.json                   # Location lists per city
├── requirements.txt                 # Python dependencies
├── README.md
│
├── data/
│   ├── generate_bangalore_data.py   # Bangalore synthetic data generator (2026 prices)
│   ├── generate_indore_data_v2.py   # Indore synthetic data generator (2026 prices)
│   ├── Bengaluru_House_Data_v3.csv  # ~3,800 rows, 41 locations, 12 features
│   └── Indore_House_Data_v3.csv     # ~2,400 rows, 36 locations, 12 features
│
├── templates/
│   └── index.html                   # Full frontend — Three.js, VanillaTilt, Tailwind
│
└── static/
    └── images/                      # Property listing images
```

---

## Data Sources (2026)

| Category | Sources |
|---|---|
| **Property Prices** | 99acres, NoBroker, Coldwell Banker, Square Yards (Mar 2026) |
| **Construction Costs** | JK Cement, Houseyog, Kenza TMT, Construction Estimator India |
| **Architect Fees** | Council of Architecture India, Houseyog |
| **Approval Costs** | BBMP/BDA (Bangalore), IMC (Indore) fee schedules |
| **GST Rates** | Government of India (5% standard, 1% affordable housing) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, Tailwind CSS 3, Vanilla JavaScript |
| **3D / Animation** | Three.js (particles + wireframe), VanillaTilt.js, CSS Keyframes |
| **Backend** | Python 3.10+, Flask 2.x |
| **ML Pipeline** | scikit-learn (StandardScaler + GradientBoostingRegressor) |
| **Hyperparameter Tuning** | RandomizedSearchCV (scipy.stats distributions) |
| **Data Processing** | Pandas, NumPy |

---

## Team

Built by **Aryan Sharma**, **Archita Jain**, and **Aradhya Gangrade**.

---

<p align="center">
  <sub>SmartEstate AI &copy; 2026 &mdash; Built with precision for the Indian real estate market.</sub>
</p>
