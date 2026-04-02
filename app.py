"""
app.py — SmartEstate AI (Flat vs Independent + Construction Costs)
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

        model   = MODELS[key]["model"]
        encoder = MODELS[key]["encoder"]
        known   = list(encoder.classes_)
        loc_enc = int(encoder.transform([location])[0]) if location in known else int(np.median(range(len(known))))
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

        price = max(float(model.predict(features)[0]), 5.0)
        breakdown = compute_breakdown(area, cost_features, price, property_type)

        profiles = MODELS[key].get("profiles", {})
        loc_prof = profiles.get(location, {})
        zone = loc_prof.get(property_type, loc_prof.get("flat", {})).get("zone", "unknown")

        return _cors(jsonify({
            "price_lakhs": round(price, 2),
            "price_inr":   int(price * 1e5),
            "price_psqft": round(price * 1e5 / area),
            "city": city, "location": location, "area": area,
            "bhk": bhk, "bathrooms": bathrooms,
            "property_type": property_type,
            "zone": zone,
            "breakdown": breakdown,
            "construction_cost_psqft": round(cost_features["construction_cost_psqft"]),
        }))
    except Exception as e:
        import traceback; traceback.print_exc()
        return _cors(jsonify({"error": str(e)})), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  SmartEstate AI — Flat + Independent")
    print("  http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
