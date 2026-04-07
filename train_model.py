"""
train_model.py  (v4 — Advanced ML Pipeline with Hyperparameter Tuning)
Features: area, bhk, bath, loc_enc, property_type_enc,
          construction_cost_psqft, architect_fee, engineer_fee,
          approval_fee, utility_cost, gst, is_resale
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import pickle, json, os

BASE = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [
    "area", "bhk", "bath", "loc_enc", "prop_type_enc",
    "construction_cost_psqft",
    "architect_fee_lakhs", "engineer_fee_lakhs",
    "approval_fee_lakhs", "utility_cost_lakhs",
    "gst_lakhs", "is_resale",
]

# Hyperparameter search space for RandomizedSearchCV
PARAM_DIST = {
    "gbr__n_estimators": randint(300, 800),
    "gbr__max_depth": randint(3, 8),
    "gbr__learning_rate": uniform(0.01, 0.15),
    "gbr__subsample": uniform(0.7, 0.25),
    "gbr__min_samples_leaf": randint(3, 15),
    "gbr__min_samples_split": randint(4, 20),
    "gbr__max_features": uniform(0.5, 0.5),
}


def train_city(city_name, csv_path):
    print("=" * 60)
    print(f"  Training {city_name} (v4 — Scaled GBR + RandomizedSearchCV)")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["location", "area", "bhk", "bath", "price", "property_type"])
    df["location"] = df["location"].str.strip()
    df["property_type"] = df["property_type"].str.strip().str.lower()

    lo, hi = df["price"].quantile(0.01), df["price"].quantile(0.99)
    df = df[(df["price"] >= lo) & (df["price"] <= hi)].copy()

    # Encode location
    le = LabelEncoder()
    df["loc_enc"] = le.fit_transform(df["location"])

    # Encode property type: flat=0, independent=1
    df["prop_type_enc"] = (df["property_type"] == "independent").astype(int)

    # Per-location + per-type cost profiles
    cost_cols = [
        "construction_cost_psqft", "architect_fee_lakhs",
        "engineer_fee_lakhs", "approval_fee_lakhs",
        "utility_cost_lakhs", "gst_lakhs"
    ]

    loc_profiles = {}
    for loc in df["location"].unique():
        loc_profiles[loc] = {}
        for ptype in ["flat", "independent"]:
            subset = df[(df["location"] == loc) & (df["property_type"] == ptype)]
            if len(subset) == 0:
                continue
            profile = {}
            for col in cost_cols:
                if col in df.columns:
                    profile[col + "_mean"] = float(subset[col].mean())
            profile["avg_area"] = float(subset["area"].mean())
            profile["avg_price_psqft"] = float(
                (subset["price"] * 1e5).mean() / subset["area"].mean()
            )
            profile["zone"] = subset["zone"].iloc[0] if "zone" in subset.columns else "unknown"
            # Store std for confidence interval calculation
            profile["price_std_pct"] = float(
                (subset["price"].std() / subset["price"].mean() * 100)
            ) if len(subset) > 1 else 6.0
            loc_profiles[loc][ptype] = profile

    X = df[FEATURE_COLS].values
    y = df["price"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline: StandardScaler -> GradientBoostingRegressor
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(random_state=42)),
    ])

    # Hyperparameter tuning with RandomizedSearchCV
    print(f"  Running RandomizedSearchCV (30 iterations, 5-fold CV)...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DIST,
        n_iter=30,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(Xtr, ytr)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_

    print(f"  Best params: {best_params}")
    print(f"  Best CV R2:  {search.best_score_:.4f}")

    preds = best_pipeline.predict(Xte)
    r2 = r2_score(yte, preds)
    mae = mean_absolute_error(yte, preds)
    cv = cross_val_score(best_pipeline, X, y, cv=5, scoring="r2")

    print(f"  Rows       : {len(df):,}  (flat={len(df[df.property_type=='flat']):,}, ind={len(df[df.property_type=='independent']):,})")
    print(f"  Locations  : {df['location'].nunique()}")
    print(f"  Features   : {len(FEATURE_COLS)}")
    print(f"  R2 (test)  : {r2:.4f}")
    print(f"  MAE (test) : Rs {mae:.1f} Lakhs")
    print(f"  CV R2      : {cv.mean():.4f} +/- {cv.std():.4f}")

    # Extract the GBR model from the pipeline for feature importances
    gbr_model = best_pipeline.named_steps["gbr"]
    importances = gbr_model.feature_importances_
    print("\n  Feature Importance:")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {name:<30} {imp:.4f}  {bar}")

    return best_pipeline, le, loc_profiles, df


# ── Train both cities ──────────────────────────
model_b, le_b, prof_b, df_b = train_city(
    "Bangalore", os.path.join(BASE, "data", "Bengaluru_House_Data_v3.csv"))
print()
model_i, le_i, prof_i, df_i = train_city(
    "Indore", os.path.join(BASE, "data", "Indore_House_Data_v3.csv"))


# ── Save ───────────────────────────────────────
bundle = {
    "bangalore": {"model": model_b, "encoder": le_b, "profiles": prof_b, "features": FEATURE_COLS},
    "indore":    {"model": model_i, "encoder": le_i, "profiles": prof_i, "features": FEATURE_COLS},
}
with open(os.path.join(BASE, "models.pkl"), "wb") as f:
    pickle.dump(bundle, f)

locations = {
    "Bangalore": sorted(df_b["location"].unique().tolist()),
    "Indore":    sorted(df_i["location"].unique().tolist()),
}
with open(os.path.join(BASE, "locations.json"), "w") as f:
    json.dump(locations, f)

print("\n" + "=" * 60)
print(f"  Saved models.pkl & locations.json")
print(f"  Bangalore: {len(locations['Bangalore'])} locations")
print(f"  Indore:    {len(locations['Indore'])} locations")
print("=" * 60)


# ── Sanity checks ──────────────────────────────
def predict_sample(pipeline, le, profiles, loc, area, bhk, bath, ptype):
    enc = int(le.transform([loc])[0])
    ptype_enc = 1 if ptype == "independent" else 0
    prof = profiles[loc].get(ptype, profiles[loc].get("flat", {}))
    ar = area / prof.get("avg_area", 1500)
    f = [[area, bhk, bath, enc, ptype_enc,
          prof.get("construction_cost_psqft_mean", 2500),
          prof.get("architect_fee_lakhs_mean", 2) * ar,
          prof.get("engineer_fee_lakhs_mean", 1) * ar,
          prof.get("approval_fee_lakhs_mean", 2),
          prof.get("utility_cost_lakhs_mean", 1),
          prof.get("gst_lakhs_mean", 1.5) * ar, 0]]
    return pipeline.predict(f)[0]

print("\n-- Bangalore: Flat vs Independent --")
for loc, area, bhk, bath in [
    ("Koramangala", 1500, 3, 2), ("Whitefield", 1200, 2, 2),
    ("Electronic City", 1000, 2, 2), ("Hoskote", 800, 2, 1),
]:
    pf = predict_sample(model_b, le_b, prof_b, loc, area, bhk, bath, "flat")
    pi = predict_sample(model_b, le_b, prof_b, loc, area, bhk, bath, "independent")
    print(f"  {loc:<22} {area}sqft {bhk}BHK  Flat=Rs {pf:.0f}L  Indep=Rs {pi:.0f}L  (+{(pi/pf-1)*100:.0f}%)")

print("\n-- Indore: Flat vs Independent --")
for loc, area, bhk, bath in [
    ("Nipania", 1500, 3, 2), ("Vijay Nagar", 1200, 2, 2),
    ("AB Road", 1500, 3, 2), ("Silicon City", 1000, 2, 2),
    ("Bijalpur", 1200, 2, 2), ("Bengali Square", 1400, 3, 2),
    ("Navlakha", 1000, 2, 2), ("Rangwasa", 800, 1, 1),
]:
    pf = predict_sample(model_i, le_i, prof_i, loc, area, bhk, bath, "flat")
    pi = predict_sample(model_i, le_i, prof_i, loc, area, bhk, bath, "independent")
    print(f"  {loc:<22} {area}sqft {bhk}BHK  Flat=Rs {pf:.0f}L  Indep=Rs {pi:.0f}L  (+{(pi/pf-1)*100:.0f}%)")
