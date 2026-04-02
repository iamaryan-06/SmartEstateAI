"""
generate_indore_data_v3.py
60 locations — ALL within Indore Municipal Corporation limits.
Removed: Pithampur, Mhow, Dharampuri, Solsinda, Hatod (separate towns).
Added: 30+ new Indore city localities from 99acres, houssed.com, Zricks 2026.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────
# 60 INDORE CITY LOCATIONS — Flat & Independent rates (₹/sqft)
# All within IMC (Indore Municipal Corporation) limits
# ──────────────────────────────────────────────────────────────────

LOCATIONS = [
    # (name, flat_min, flat_max, ind_min, ind_max, zone)

    # ── ELITE (posh central / super corridor) ─────────────────
    ("Nipania",              4800, 7000,   7000, 10500, "elite"),
    ("Super Corridor",       4500, 6800,   6500,  9800, "elite"),
    ("Old Palasia",          5000, 7000,   7500, 11000, "elite"),
    ("New Palasia",          4800, 6800,   7200, 10800, "elite"),
    ("Sapna Sangeeta Road",  4800, 6800,   7200, 10500, "elite"),
    ("Scheme 140",           4600, 6800,   6800, 10000, "elite"),
    ("South Tukoganj",       5000, 7200,   7500, 11000, "elite"),

    # ── PREMIUM (established residential hubs) ────────────────
    ("Vijay Nagar",          4200, 5800,   6500,  9500, "premium"),
    ("MG Road",              4500, 6000,   6800,  9800, "premium"),
    ("Mahalaxmi Nagar",      4000, 5500,   6000,  8800, "premium"),
    ("Bicholi Mardana",      3800, 5200,   5800,  8500, "premium"),
    ("MR 10",                4000, 5400,   6000,  8800, "premium"),
    ("Rajendra Nagar",       3800, 5200,   5800,  8500, "premium"),
    ("AB Road",              3600, 5000,   5500,  8200, "premium"),
    ("Scheme 94",            4000, 5500,   6000,  8800, "premium"),
    ("Bengali Square",       4000, 5600,   6200,  9000, "premium"),
    ("AB Bypass Road",       3800, 5200,   5800,  8500, "premium"),
    ("Scheme 136",           3800, 5400,   5800,  8500, "premium"),
    ("Navlakha",             3600, 5000,   5500,  8000, "premium"),
    ("Janki Nagar",          3800, 5200,   5800,  8200, "premium"),

    # ── CLASSIC (mid-range, well-connected) ───────────────────
    ("Bhawarkuan",           3200, 4400,   4800,  7000, "classic"),
    ("Bhawrasla",            3300, 4500,   5000,  7200, "classic"),
    ("Sudama Nagar",         3200, 4300,   4800,  6800, "classic"),
    ("Rau",                  3000, 4200,   4500,  6500, "classic"),
    ("Piplya Kumar",         3300, 4400,   5000,  7000, "classic"),
    ("Dewas Naka",           3100, 4200,   4600,  6600, "classic"),
    ("Pipliyahana",          3200, 4300,   4800,  6800, "classic"),
    ("Lasuriya Mori",        3000, 4100,   4500,  6200, "classic"),
    ("Scheme 54",            3300, 4400,   5000,  7000, "classic"),
    ("Kanadia Road",         2800, 3900,   4200,  6200, "classic"),
    ("Khandwa Road",         3000, 4000,   4400,  6400, "classic"),
    ("Talawali Chanda",      3000, 4100,   4500,  6400, "classic"),
    ("Annapurna Road",       3200, 4400,   4800,  7000, "classic"),
    ("Sneh Nagar",           3200, 4400,   4800,  6800, "classic"),
    ("Geeta Bhawan",         3400, 4600,   5000,  7200, "classic"),
    ("Tilak Nagar",          3000, 4200,   4500,  6500, "classic"),
    ("Scheme 71",            3200, 4400,   4800,  7000, "classic"),
    ("Scheme 103",           3000, 4200,   4500,  6500, "classic"),
    ("Ring Road",            3000, 4200,   4500,  6600, "classic"),
    ("Aerodrome Road",       3200, 4400,   4800,  7000, "classic"),
    ("MR 11",                3200, 4400,   4800,  7000, "classic"),
    ("Bijalpur",             3400, 4800,   5200,  7500, "classic"),
    ("Nihalpur Mundi",       3000, 4200,   4500,  6500, "classic"),
    ("Sukhliya",             3200, 4400,   4800,  6800, "classic"),
    ("Bicholi Hapsi",        3000, 4200,   4500,  6500, "classic"),
    ("Palasia Square",       3400, 4600,   5200,  7200, "classic"),
    ("Agrawal Nagar",        3200, 4400,   4800,  6800, "classic"),

    # ── AFFORDABLE (developing / peripheral Indore city) ──────
    ("Silicon City",         2500, 3500,   3800,  5500, "affordable"),
    ("Jetpura",              2600, 3400,   3800,  5200, "affordable"),
    ("Manglia",              2400, 3300,   3500,  5000, "affordable"),
    ("Scheme 78",            2600, 3500,   3800,  5400, "affordable"),
    ("Rau Pithampur Road",   2500, 3400,   3600,  5200, "affordable"),
    ("Musakhedi",            2400, 3200,   3400,  4800, "affordable"),
    ("Tejaji Nagar",         2200, 3200,   3200,  4800, "affordable"),
    ("Banganga",             2500, 3400,   3600,  5200, "affordable"),
    ("Chandan Nagar",        2600, 3500,   3800,  5400, "affordable"),
    ("Gandhi Nagar",         2800, 3600,   4000,  5600, "affordable"),
    ("Khajrana",             2600, 3600,   3800,  5400, "affordable"),
    ("Limbodi",              2400, 3200,   3400,  4800, "affordable"),
    ("Palhar Nagar",         2500, 3400,   3600,  5200, "affordable"),
    ("Rangwasa",             2200, 3000,   3200,  4600, "affordable"),
    ("Saket Nagar",          2800, 3800,   4000,  5800, "affordable"),
    ("Sanwer Road",          2200, 3200,   3200,  4800, "affordable"),
]

assert len(LOCATIONS) >= 60, f"Expected 60+ locations, got {len(LOCATIONS)}"

# ── Cost parameters (same structure as before) ────────────────

CONSTRUCTION_COST = {
    "elite":      ((2200, 3200), (2800, 4000)),
    "premium":    ((1800, 2600), (2400, 3400)),
    "classic":    ((1500, 2300), (1900, 2900)),
    "affordable": ((1200, 1900), (1500, 2400)),
}

ARCHITECT_FEE = {"flat": (0.01, 0.03), "independent": (0.03, 0.07)}
STRUCTURAL_ENGINEER_FEE = {"flat": (0.005, 0.015), "independent": (0.015, 0.03)}

APPROVAL_FEES = {
    "flat": {
        "elite": (40000, 120000), "premium": (30000, 100000),
        "classic": (25000, 80000), "affordable": (20000, 60000),
    },
    "independent": {
        "elite": (180000, 450000), "premium": (130000, 350000),
        "classic": (100000, 250000), "affordable": (70000, 180000),
    },
}

GST_RATE = {"elite": 0.05, "premium": 0.05, "classic": 0.05, "affordable": 0.01}

UTILITY_COST = {
    "flat": {
        "elite": (30000, 80000), "premium": (25000, 60000),
        "classic": (20000, 50000), "affordable": (15000, 40000),
    },
    "independent": {
        "elite": (100000, 200000), "premium": (80000, 160000),
        "classic": (60000, 120000), "affordable": (40000, 90000),
    },
}

BHK_BATH = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}

ZONE_AREA = {
    "elite":      {"flat": (800, 3500),  "independent": (1200, 5000)},
    "premium":    {"flat": (700, 2500),  "independent": (1000, 4000)},
    "classic":    {"flat": (600, 2000),  "independent": (800, 3000)},
    "affordable": {"flat": (500, 1600),  "independent": (700, 2500)},
}

rows = []

for loc_name, flat_min, flat_max, ind_min, ind_max, zone in LOCATIONS:
    for prop_type in ["flat", "independent"]:
        n = 35 if prop_type == "flat" else 30

        rate_min = flat_min if prop_type == "flat" else ind_min
        rate_max = flat_max if prop_type == "flat" else ind_max
        area_lo, area_hi = ZONE_AREA[zone][prop_type]
        areas = np.random.randint(area_lo, area_hi, size=n)

        if zone in ("elite", "premium"):
            if prop_type == "independent":
                bhk_ch = np.random.choice([2, 3, 4, 5], n, p=[0.05, 0.35, 0.40, 0.20])
            else:
                bhk_ch = np.random.choice([1, 2, 3, 4], n, p=[0.05, 0.35, 0.45, 0.15])
        elif zone == "classic":
            bhk_ch = np.random.choice([1, 2, 3, 4], n, p=[0.10, 0.40, 0.40, 0.10])
        else:
            bhk_ch = np.random.choice([1, 2, 3], n, p=[0.20, 0.50, 0.30])

        cc_range = CONSTRUCTION_COST[zone][0 if prop_type == "flat" else 1]
        arch_range = ARCHITECT_FEE[prop_type]
        eng_range = STRUCTURAL_ENGINEER_FEE[prop_type]
        appr_range = APPROVAL_FEES[prop_type][zone]
        util_range = UTILITY_COST[prop_type][zone]

        for i in range(n):
            bhk = int(bhk_ch[i])
            bath = BHK_BATH.get(bhk, 2)
            bath = min(bath + np.random.choice([0, 0, 0, 1]), 5)
            area = int(max(areas[i], bhk * 350))

            market_rate = np.random.uniform(rate_min, rate_max)
            construction_rate = np.random.uniform(*cc_range)
            construction_cost = area * construction_rate

            architect_fee = construction_cost * np.random.uniform(*arch_range)
            engineer_fee = construction_cost * np.random.uniform(*eng_range)
            approval_fee = np.random.uniform(*appr_range)
            utility_cost = np.random.uniform(*util_range)
            gst_amount = construction_cost * GST_RATE[zone]

            base_price = area * market_rate
            professional_overhead = architect_fee + engineer_fee + approval_fee + utility_cost

            if prop_type == "independent":
                total_price = base_price + (professional_overhead * 0.4) + (gst_amount * 0.5)
            else:
                total_price = base_price + (professional_overhead * 0.2) + (gst_amount * 0.4)

            bhk_mult = 1 + (bhk - 2) * 0.025
            total_price *= bhk_mult

            is_resale = np.random.random() < 0.30
            if is_resale:
                total_price *= np.random.uniform(0.78, 0.95)

            noise = np.random.uniform(-0.05, 0.05)
            total_price *= (1 + noise)
            price_lakhs = total_price / 1e5

            rows.append({
                "location": loc_name, "area": area, "bhk": bhk, "bath": int(bath),
                "property_type": prop_type, "zone": zone,
                "construction_cost_psqft": round(construction_rate),
                "architect_fee_lakhs": round(architect_fee / 1e5, 2),
                "engineer_fee_lakhs": round(engineer_fee / 1e5, 2),
                "approval_fee_lakhs": round(approval_fee / 1e5, 2),
                "utility_cost_lakhs": round(utility_cost / 1e5, 2),
                "gst_lakhs": round(gst_amount / 1e5, 2),
                "is_resale": int(is_resale),
                "price": round(price_lakhs, 2),
            })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Indore_House_Data_v3.csv")
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} rows, {df['location'].nunique()} locations")
print(f"\nBy property type:")
print(df.groupby('property_type')['price'].describe()[['count','mean','min','max']].round(1))
print(f"\nFlat prices by zone:")
print(df[df['property_type']=='flat'].groupby('zone')['price'].describe()[['count','mean','min','max']].round(1))
print(f"\nIndependent prices by zone:")
print(df[df['property_type']=='independent'].groupby('zone')['price'].describe()[['count','mean','min','max']].round(1))
print(f"\nAll 60 locations: {sorted(df['location'].unique().tolist())}")
print(f"\nSaved to {out_path}")
