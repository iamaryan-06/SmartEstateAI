"""
app.py — SmartEstate AI v4 (Scaled Pipeline + Price Range + AI Description)
"""

from flask import Flask, request, jsonify, render_template
import pickle, json, os, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__, template_folder="templates", static_folder="static")

with open(os.path.join(BASE, "models.pkl"), "rb") as f:
    MODELS = pickle.load(f)
with open(os.path.join(BASE, "locations.json"), "r") as f:
    LOCATIONS = json.load(f)

def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def estimate_cost_features(city_key, location, area, property_type):
    profiles = MODELS[city_key].get("profiles", {})
    loc_prof = profiles.get(location, {})
    prof = loc_prof.get(property_type, loc_prof.get("flat", {}))

    if not prof:
        # Fallback: average across all locations for this property type
        all_profs = []
        for lp in profiles.values():
            if property_type in lp:
                all_profs.append(lp[property_type])
        if not all_profs:
            for lp in profiles.values():
                for p in lp.values():
                    if isinstance(p, dict):
                        all_profs.append(p)
        prof = {}
        if all_profs:
            for key in all_profs[0]:
                vals = [p.get(key, 0) for p in all_profs if isinstance(p.get(key, 0), (int, float))]
                prof[key] = sum(vals) / len(vals) if vals else 0

    avg_area = prof.get("avg_area", 1500)
    area_ratio = area / avg_area if avg_area > 0 else 1.0

    return {
        "construction_cost_psqft": prof.get("construction_cost_psqft_mean", 2500),
        "architect_fee_lakhs": prof.get("architect_fee_lakhs_mean", 2.0) * area_ratio,
        "engineer_fee_lakhs": prof.get("engineer_fee_lakhs_mean", 1.0) * area_ratio,
        "approval_fee_lakhs": prof.get("approval_fee_lakhs_mean", 2.0),
        "utility_cost_lakhs": prof.get("utility_cost_lakhs_mean", 1.0),
        "gst_lakhs": prof.get("gst_lakhs_mean", 1.5) * area_ratio,
        "price_std_pct": prof.get("price_std_pct", 6.0),
    }


def compute_breakdown(area, cost_features, total_price_lakhs, property_type):
    total_inr = total_price_lakhs * 1e5
    construction_total = area * cost_features["construction_cost_psqft"]
    architect = cost_features["architect_fee_lakhs"] * 1e5
    engineer = cost_features["engineer_fee_lakhs"] * 1e5
    approval = cost_features["approval_fee_lakhs"] * 1e5
    utility = cost_features["utility_cost_lakhs"] * 1e5
    gst = cost_features["gst_lakhs"] * 1e5

    known = construction_total + architect + engineer + approval + utility + gst
    land_margin = max(total_inr - known, total_inr * 0.20)
    total_comp = land_margin + known
    scale = total_inr / total_comp if total_comp > 0 else 1

    land_label = "Land Cost + Builder Margin" if property_type == "flat" else "Land / Plot Cost"

    return {
        "land_and_margin": {"amount_lakhs": round(land_margin * scale / 1e5, 2), "label": land_label},
        "construction":    {"amount_lakhs": round(construction_total * scale / 1e5, 2),
                            "cost_psqft": round(cost_features["construction_cost_psqft"]),
                            "label": "Construction (Materials + Labour)"},
        "architect_fee":   {"amount_lakhs": round(architect * scale / 1e5, 2), "label": "Architect / Design Fees"},
        "engineer_fee":    {"amount_lakhs": round(engineer * scale / 1e5, 2), "label": "Structural Engineer Fees"},
        "approval_fee":    {"amount_lakhs": round(approval * scale / 1e5, 2), "label": "Approvals / Mapping / Statutory"},
        "utility_cost":    {"amount_lakhs": round(utility * scale / 1e5, 2), "label": "Utility Connections"},
        "gst":             {"amount_lakhs": round(gst * scale / 1e5, 2), "label": "GST & Taxes"},
    }


def compute_price_range(price_lakhs, zone, price_std_pct):
    """Compute realistic price range bounds based on zone and location variance."""
    # Base margin from location's historical price standard deviation
    base_pct = max(min(price_std_pct, 12.0), 4.0)

    # Adjust by zone: premium zones have tighter ranges, budget zones wider
    zone_adj = {
        "premium": -1.5,
        "tier1": -0.5,
        "tier2": 0.5,
        "budget": 1.5,
    }
    adj = zone_adj.get(zone, 0.0)
    margin_pct = base_pct + adj

    # Clamp to 5-8% as specified
    margin_pct = max(5.0, min(8.0, margin_pct))

    price_low = round(price_lakhs * (1 - margin_pct / 100), 2)
    price_high = round(price_lakhs * (1 + margin_pct / 100), 2)
    return price_low, price_high, round(margin_pct, 1)


def compute_local_market_accuracy(zone, price_std_pct, property_type):
    """
    Calculate a local market accuracy percentage (88-96%) based on:
    - Zone tier (premium zones have denser data, higher accuracy)
    - Price standard deviation (lower variance = more predictable = higher accuracy)
    - Property type (flats have more comparable data than independent houses)
    """
    zone_base = {
        "premium": 94.0,
        "tier1": 92.0,
        "tier2": 90.0,
        "budget": 88.5,
    }
    base = zone_base.get(zone, 90.0)

    # Lower price variance → higher accuracy (each 1% below 8% adds 0.3%)
    variance_bonus = max(0, (8.0 - min(price_std_pct, 12.0))) * 0.3

    # Flats have more market comparables than independent houses
    type_bonus = 0.5 if property_type == "flat" else 0.0

    accuracy = base + variance_bonus + type_bonus
    return round(max(88.0, min(96.0, accuracy)), 1)


def compute_acquisition_probability(zone, price_std_pct, margin_pct):
    """
    Calculate acquisition probability (80-95%): the likelihood a buyer can
    secure this property within the estimated price range.

    Logic:
    - Budget/tier2 zones → higher probability (less competition, more stock)
    - Premium zones → lower probability (bidding wars, limited inventory)
    - Wider margin → higher probability (more room for negotiation)
    - Lower price variance → more stable market → higher probability
    """
    zone_base = {
        "premium": 82.0,
        "tier1": 85.0,
        "tier2": 89.0,
        "budget": 92.0,
    }
    base = zone_base.get(zone, 86.0)

    # Wider confidence margin = easier to land within range
    margin_bonus = (margin_pct - 5.0) * 1.0  # 0 to 3

    # Lower price variance = more predictable market
    stability_bonus = max(0, (8.0 - min(price_std_pct, 12.0))) * 0.5

    prob = base + margin_bonus + stability_bonus
    return round(max(80.0, min(95.0, prob)), 0)


def generate_description(city, location, bhk, area, property_type, zone, price_lakhs):
    """Generate a compelling commercial description based on property attributes."""
    # Determine tier description
    tier_desc = {
        "premium": "one of the most sought-after premium localities",
        "tier1": "a well-established Tier-1 neighbourhood with excellent connectivity",
        "tier2": "an emerging Tier-2 micro-market with strong growth potential",
        "budget": "a value-driven locality ideal for first-time homebuyers",
    }
    zone_text = tier_desc.get(zone, "a promising residential area")

    # Property type context
    if property_type == "independent":
        type_text = f"independent house"
        lifestyle = "offering complete privacy, a personal garden space, and the freedom of standalone living"
    else:
        type_text = f"apartment"
        lifestyle = "featuring modern amenities, 24/7 security, and a vibrant community lifestyle"

    # Size classification
    if area >= 2500:
        size_class = "expansive"
    elif area >= 1500:
        size_class = "spacious"
    elif area >= 1000:
        size_class = "well-proportioned"
    else:
        size_class = "smartly designed"

    # Price tier context
    if price_lakhs >= 200:
        price_tier = "This ultra-luxury property represents a marquee investment"
    elif price_lakhs >= 100:
        price_tier = "This premium property is an excellent high-value investment"
    elif price_lakhs >= 50:
        price_tier = "This mid-premium property offers strong value appreciation potential"
    else:
        price_tier = "This competitively priced property is ideal for smart investors"

    # City-specific flavor
    city_flavor = {
        "Bangalore": "India's Silicon Valley, known for its thriving tech ecosystem and cosmopolitan culture",
        "Indore": "Central India's cleanest city and a rapidly growing commercial hub",
    }
    city_text = city_flavor.get(city, f"the vibrant city of {city}")

    desc = (
        f"This {size_class} {bhk}-BHK {type_text} spanning {area:,} sq ft in {location}, {city} "
        f"is situated in {zone_text}. {lifestyle.capitalize()}. "
        f"{price_tier} in {city_text}."
    )
    return desc


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/locations", methods=["GET", "OPTIONS"])
def locations():
    city = request.args.get("city", "")
    return _cors(jsonify({"locations": LOCATIONS.get(city, [])}))

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return _cors(app.make_default_options_response())
    try:
        data = request.get_json(force=True)
        city          = data.get("city", "").strip()
        location      = data.get("location", "").strip()
        area          = float(data.get("area", 0))
        bhk           = int(data.get("bhk", 1))
        bathrooms     = int(data.get("bathrooms", 1))
        property_type = data.get("property_type", "flat").strip().lower()

        if property_type not in ("flat", "independent"):
            property_type = "flat"

        if not city or not location or area <= 0:
            return _cors(jsonify({"error": "Invalid inputs"})), 400

        key = city.lower()
        if key not in MODELS:
            return _cors(jsonify({"error": f"No model for city: {city}"})), 400

        pipeline = MODELS[key]["model"]
        encoder  = MODELS[key]["encoder"]
        known    = list(encoder.classes_)
        loc_enc  = int(encoder.transform([location])[0]) if location in known else int(np.median(range(len(known))))
        prop_type_enc = 1 if property_type == "independent" else 0

        cost_features = estimate_cost_features(key, location, area, property_type)

        features = np.array([[
            area, bhk, bathrooms, loc_enc, prop_type_enc,
            cost_features["construction_cost_psqft"],
            cost_features["architect_fee_lakhs"],
            cost_features["engineer_fee_lakhs"],
            cost_features["approval_fee_lakhs"],
            cost_features["utility_cost_lakhs"],
            cost_features["gst_lakhs"],
            0,
        ]])

        # Pipeline includes StandardScaler + GBR
        price = max(float(pipeline.predict(features)[0]), 5.0)
        breakdown = compute_breakdown(area, cost_features, price, property_type)

        profiles = MODELS[key].get("profiles", {})
        loc_prof = profiles.get(location, {})
        zone = loc_prof.get(property_type, loc_prof.get("flat", {})).get("zone", "unknown")

        # Price range (Requirement 3)
        price_low, price_high, margin_pct = compute_price_range(
            price, zone, cost_features.get("price_std_pct", 6.0)
        )

        # AI Description (Requirement 5)
        description = generate_description(
            city, location, bhk, area, property_type, zone, price
        )

        # Local Market Accuracy (Requirement 3 — new)
        local_market_accuracy = compute_local_market_accuracy(
            zone, cost_features.get("price_std_pct", 6.0), property_type
        )

        # Acquisition Probability (Requirement 4 — new)
        acquisition_probability = compute_acquisition_probability(
            zone, cost_features.get("price_std_pct", 6.0), margin_pct
        )

        return _cors(jsonify({
            "price_lakhs": round(price, 2),
            "price_low_lakhs": price_low,
            "price_high_lakhs": price_high,
            "margin_pct": margin_pct,
            "price_inr":   int(price * 1e5),
            "price_psqft": round(price * 1e5 / area),
            "city": city, "location": location, "area": area,
            "bhk": bhk, "bathrooms": bathrooms,
            "property_type": property_type,
            "zone": zone,
            "breakdown": breakdown,
            "construction_cost_psqft": round(cost_features["construction_cost_psqft"]),
            "description": description,
            "local_market_accuracy": local_market_accuracy,
            "acquisition_probability": int(acquisition_probability),
        }))
    except Exception as e:
        import traceback; traceback.print_exc()
        return _cors(jsonify({"error": str(e)})), 500


if __name__ == "__main__":
    import os as _os
    port = int(_os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  SmartEstate AI v4 — Scaled Pipeline + Ranges")
    print(f"  http://127.0.0.1:{port}")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=port)
