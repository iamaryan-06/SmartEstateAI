"""
generate_bangalore_data_v3.py
Generates realistic Bangalore housing dataset with PROPERTY TYPE separation.
Independent houses: 40-70% higher per sqft than flats (full land ownership).
Sources: 99acres, NoBroker, Coldwell Banker (Mar 2026)
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

LOCATIONS = [
    #                              --- Flat ---      --- Independent ---
    # (name, flat_min, flat_max, ind_min, ind_max, zone)

    # Ultra Premium
    ("Koramangala",         12000, 20000,  18000, 30000, "ultra_premium"),
    ("Indiranagar",         12000, 22000,  18000, 32000, "ultra_premium"),
    ("MG Road",             10000, 18000,  16000, 28000, "ultra_premium"),
    ("Rajajinagar",         11000, 20000,  17000, 30000, "ultra_premium"),
    ("Jayanagar",           10000, 16000,  15000, 25000, "ultra_premium"),
    ("Basavanagudi",         9500, 15000,  14000, 24000, "ultra_premium"),
    ("Frazer Town",          8500, 13000,  13000, 20000, "ultra_premium"),
    ("Ulsoor",               9000, 14000,  14000, 22000, "ultra_premium"),

    # Premium
    ("Whitefield",           7000, 13000,  10500, 18000, "premium"),
    ("HSR Layout",           8000, 13000,  12000, 20000, "premium"),
    ("Bellandur",            7000, 11000,  10000, 17000, "premium"),
    ("Sarjapur Road",        6000, 10000,   9000, 15000, "premium"),
    ("Hebbal",               8000, 12000,  12000, 18000, "premium"),
    ("Thanisandra",          7000, 10000,  10000, 15000, "premium"),
    ("Marathahalli",         6000,  9500,   9000, 14000, "premium"),
    ("JP Nagar",             6500, 10000,  10000, 16000, "premium"),
    ("BTM Layout",           7000, 10000,  10000, 15000, "premium"),
    ("Bannerghatta Road",    6500, 10000,   9500, 15000, "premium"),
    ("Malleshwaram",         8000, 13000,  12000, 20000, "premium"),
    ("Hennur",               6000,  9000,   9000, 14000, "premium"),

    # Mid-Range
    ("Electronic City",      4200,  7000,   6500, 10500, "mid"),
    ("KR Puram",             4500,  7000,   7000, 10500, "mid"),
    ("Varthur",              5000,  8000,   7500, 12000, "mid"),
    ("Hoodi",                5500,  8500,   8000, 12500, "mid"),
    ("Kengeri",              4000,  6500,   6000, 10000, "mid"),
    ("Banashankari",         5500,  8500,   8000, 13000, "mid"),
    ("Yelahanka",            5000,  8000,   7500, 12000, "mid"),
    ("Mysore Road",          4500,  7000,   7000, 10500, "mid"),
    ("Tumkur Road",          4000,  6500,   6000, 10000, "mid"),
    ("Rajarajeshwari Nagar", 4000,  6500,   6000, 10000, "mid"),
    ("Vidyaranyapura",       5000,  7500,   7500, 11000, "mid"),
    ("Horamavu",             4500,  7000,   7000, 10500, "mid"),

    # Affordable
    ("Hoskote",              2800,  4500,   4200,  7000, "affordable"),
    ("Nelamangala",          2800,  4500,   4200,  7000, "affordable"),
    ("Devanahalli",          3200,  5500,   5000,  8500, "affordable"),
    ("Chandapura",           2800,  4500,   4200,  7000, "affordable"),
    ("Attibele",             2500,  4000,   3800,  6500, "affordable"),
    ("Jigani",               2800,  4500,   4200,  7000, "affordable"),
    ("Ramamurthy Nagar",     3500,  5500,   5500,  8500, "affordable"),
    ("TC Palaya",            3000,  4500,   4500,  7000, "affordable"),
    ("Bagalur",              3500,  6000,   5500,  9000, "affordable"),
]

CONSTRUCTION_COST = {
    "ultra_premium": ((2500, 4000), (3000, 5000)),
    "premium":       ((2000, 3200), (2500, 4000)),
    "mid":           ((1600, 2500), (2000, 3200)),
    "affordable":    ((1300, 2000), (1600, 2500)),
}

ARCHITECT_FEE = {"flat": (0.01, 0.03), "independent": (0.03, 0.07)}
STRUCTURAL_ENGINEER_FEE = {"flat": (0.005, 0.015), "independent": (0.015, 0.03)}

APPROVAL_FEES = {
    "flat": {
        "ultra_premium": (60000, 180000), "premium": (50000, 150000),
        "mid": (40000, 120000), "affordable": (30000, 80000),
    },
    "independent": {
        "ultra_premium": (250000, 550000), "premium": (180000, 450000),
        "mid": (120000, 350000), "affordable": (80000, 220000),
    },
}

GST_RATE = {"ultra_premium": 0.05, "premium": 0.05, "mid": 0.05, "affordable": 0.01}

UTILITY_COST = {
    "flat": {
        "ultra_premium": (40000, 100000), "premium": (30000, 80000),
        "mid": (25000, 60000), "affordable": (20000, 50000),
    },
    "independent": {
        "ultra_premium": (120000, 250000), "premium": (90000, 180000),
        "mid": (60000, 140000), "affordable": (45000, 100000),
    },
}

BHK_BATH = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}

ZONE_AREA = {
    "ultra_premium": {"flat": (700, 3500),  "independent": (1200, 5000)},
    "premium":       {"flat": (600, 2800),  "independent": (1000, 4000)},
    "mid":           {"flat": (500, 2200),  "independent": (800, 3200)},
    "affordable":    {"flat": (450, 1800),  "independent": (600, 2500)},
}

rows = []

for loc_name, flat_min, flat_max, ind_min, ind_max, zone in LOCATIONS:
    for prop_type in ["flat", "independent"]:
        n = 50 if prop_type == "flat" else 40

        rate_min = flat_min if prop_type == "flat" else ind_min
        rate_max = flat_max if prop_type == "flat" else ind_max
        area_lo, area_hi = ZONE_AREA[zone][prop_type]
        areas = np.random.randint(area_lo, area_hi, size=n)

        if zone in ("ultra_premium",):
            if prop_type == "independent":
                bhk_ch = np.random.choice([3, 4, 5], n, p=[0.35, 0.40, 0.25])
            else:
                bhk_ch = np.random.choice([2, 3, 4, 5], n, p=[0.10, 0.40, 0.35, 0.15])
        elif zone == "premium":
            if prop_type == "independent":
                bhk_ch = np.random.choice([2, 3, 4, 5], n, p=[0.05, 0.35, 0.40, 0.20])
            else:
                bhk_ch = np.random.choice([1, 2, 3, 4], n, p=[0.05, 0.30, 0.45, 0.20])
        elif zone == "mid":
            bhk_ch = np.random.choice([1, 2, 3, 4], n, p=[0.10, 0.40, 0.40, 0.10])
        else:
            bhk_ch = np.random.choice([1, 2, 3], n, p=[0.25, 0.50, 0.25])

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

            bhk_mult = 1 + (bhk - 2) * 0.02
            total_price *= bhk_mult

            is_resale = np.random.random() < 0.35
            if is_resale:
                total_price *= np.random.uniform(0.75, 0.95)

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

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Bengaluru_House_Data_v3.csv")
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} rows, {df['location'].nunique()} locations")
print(f"\nBy property type:")
print(df.groupby('property_type')['price'].describe()[['count','mean','min','max']].round(1))
print(f"\nFlat prices by zone:")
print(df[df['property_type']=='flat'].groupby('zone')['price'].describe()[['count','mean','min','max']].round(1))
print(f"\nIndependent house prices by zone:")
print(df[df['property_type']=='independent'].groupby('zone')['price'].describe()[['count','mean','min','max']].round(1))
print(f"\nSaved to {out_path}")
