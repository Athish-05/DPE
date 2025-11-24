from fastapi import FastAPI
import numpy as np
import joblib
import xgboost as xgb

# --------------------------
# Load XGBoost Artifacts
# --------------------------
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("final_xgb_model.json")

route_encoder = joblib.load("route_label_encoder.pkl")
feature_list = joblib.load("feature_list.pkl")

app = FastAPI()

# --------------------------
# Demand / Revenue Function
# --------------------------
def compute_demand_and_revenue(data, price):
    trend = data["google_trends_score"] / 100
    demand_scale = (
        data["base_demand_strength"] *
        (1 + 0.2*data["event_intensity"] + 0.1*data["holiday_flag"]) *
        data["macro_demand_index"] *
        (0.8 + 0.4*trend)
    )

    demand = demand_scale + data["true_price_elasticity"] * (price - data["base_price"])
    demand = max(0, demand)

    revenue = price * demand
    return demand, revenue

# --------------------------
# Prediction Endpoint
# --------------------------
@app.post("/predict_xgb")
def predict_xgb(data: dict):

    # Encode route
    data["route_enc"] = int(route_encoder.transform([data["route"]])[0])
    del data["route"]

    # Prepare input vector
    X = np.array([[data[c] for c in feature_list]])

    # Predict price
    predicted_price = float(xgb_model.predict(X)[0])

    # Demand + Revenue
    d, r = compute_demand_and_revenue(data, predicted_price)

    return {
        "model": "XGBoost",
        "predicted_price": predicted_price,
        "predicted_demand": d,
        "predicted_revenue": r
    }
# --------------------------
# Revenue Curve (Graph Data)
# --------------------------
@app.post("/revenue_curve")
def revenue_curve(data: dict):

    # Encode route
    data["route_enc"] = int(route_encoder.transform([data["route"]])[0])
    del data["route"]

    # Generate a range of possible ticket prices
    min_price = max(1000, data["base_price"] * 0.5)
    max_price = data["base_price"] * 1.8

    prices = np.linspace(min_price, max_price, 50)  # 50 price points

    revenues = []
    demands = []

    # Compute demand + revenue for each price
    for price in prices:
        d, r = compute_demand_and_revenue(data, price)
        demands.append(d)
        revenues.append(r)

    # Find best price
    best_idx = int(np.argmax(revenues))
    best_price = float(prices[best_idx])
    best_revenue = float(revenues[best_idx])

    return {
        "prices": prices.tolist(),
        "demands": demands,
        "revenues": revenues,
        "best_price": best_price,
        "best_revenue": best_revenue
    }