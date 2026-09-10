from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from src.shap_flask_explanation import get_shap_explanation

app = Flask(__name__)

# Load deployment package
package = joblib.load(
    "models/deployment/water_quality_disease_model.pkl"
)

model = package["model"]
preprocessor = package["preprocessor"]
feature_names = package["feature_names"]
disease_classes = package["disease_classes"]

# Input features
input_features = [
    "water_quality_index",
    "ph",
    "turbidity_ntu",
    "dissolved_oxygen_mg_l",
    "bod_mg_l",
    "fecal_coliform_per_100ml",
    "total_coliform_per_100ml",
    "tds_mg_l",
    "nitrate_mg_l",
    "fluoride_mg_l",
    "arsenic_ug_l",
    "open_defecation_rate",
    "toilet_access",
    "sewage_treatment_pct",
    "water_source",
    "water_treatment",
    "handwashing_practice",
    "flooding",
    "avg_temperature_c",
    "avg_rainfall_mm",
    "avg_humidity_pct",
    "season",
    "month"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = {}

        # Numerical features
        numerical_features = [
            "water_quality_index",
            "ph",
            "turbidity_ntu",
            "dissolved_oxygen_mg_l",
            "bod_mg_l",
            "fecal_coliform_per_100ml",
            "total_coliform_per_100ml",
            "tds_mg_l",
            "nitrate_mg_l",
            "fluoride_ug_l",
            "arsenic_ug_l",
            "open_defecation_rate",
            "toilet_access",
            "sewage_treatment_pct",
            "flooding",
            "avg_temperature_c",
            "avg_rainfall_mm",
            "avg_humidity_pct",
            "month"
        ]

        # Read all form values
        for feature in input_features:
            data[feature] = request.form.get(feature)

        # Correct fluoride field name
        if "fluoride_ug_l" in data:
            data["fluoride_mg_l"] = data.pop("fluoride_ug_l")

        # Convert numerical values
        for feature in [
            "water_quality_index",
            "ph",
            "turbidity_ntu",
            "dissolved_oxygen_mg_l",
            "bod_mg_l",
            "fecal_coliform_per_100ml",
            "total_coliform_per_100ml",
            "tds_mg_l",
            "nitrate_mg_l",
            "fluoride_mg_l",
            "arsenic_ug_l",
            "open_defecation_rate",
            "toilet_access",
            "sewage_treatment_pct",
            "flooding",
            "avg_temperature_c",
            "avg_rainfall_mm",
            "avg_humidity_pct",
            "month"
        ]:
            data[feature] = float(data[feature])

        # Create DataFrame in correct feature order
        input_df = pd.DataFrame(
            [[data[feature] for feature in input_features]],
            columns=input_features
        )

        # Preprocess
        X_processed = preprocessor.transform(input_df)

        # Prediction
        prediction = model.predict(X_processed)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(X_processed)[0]

        # Convert prediction to disease name
        predicted_disease = prediction

        # Confidence
        confidence = float(np.max(probabilities)) * 100

        # Disease probability list
        probability_data = []

        for disease, probability in zip(
            disease_classes,
            probabilities
        ):
            probability_data.append({
                "disease": disease,
                "probability": round(float(probability) * 100, 2)
            })

        # Sort highest probability first
        probability_data.sort(
            key=lambda x: x["probability"],
            reverse=True
        )

        # Generate SHAP explanation
        shap_explanations = get_shap_explanation(
            input_df,
            predicted_disease
        )

        return render_template(
            "result.html",
            prediction=predicted_disease,
            confidence=round(confidence, 2),
            probabilities=probability_data,
            shap_explanations=shap_explanations
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=str(e)
        )

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:
        data = request.get_json()

        # Required input features
        required_features = [
            "water_quality_index",
            "ph",
            "turbidity_ntu",
            "dissolved_oxygen_mg_l",
            "bod_mg_l",
            "fecal_coliform_per_100ml",
            "total_coliform_per_100ml",
            "tds_mg_l",
            "nitrate_mg_l",
            "fluoride_mg_l",
            "arsenic_ug_l",
            "open_defecation_rate",
            "toilet_access",
            "sewage_treatment_pct",
            "water_source",
            "water_treatment",
            "handwashing_practice",
            "flooding",
            "avg_temperature_c",
            "avg_rainfall_mm",
            "avg_humidity_pct",
            "season",
            "month"
        ]

        # Check for missing features
        missing_features = [
            feature
            for feature in required_features
            if feature not in data
        ]

        if missing_features:
            return {
                "error": "Missing required features",
                "missing_features": missing_features
            }, 400

        # Convert numerical values
        numerical_features = [
            "water_quality_index",
            "ph",
            "turbidity_ntu",
            "dissolved_oxygen_mg_l",
            "bod_mg_l",
            "fecal_coliform_per_100ml",
            "total_coliform_per_100ml",
            "tds_mg_l",
            "nitrate_mg_l",
            "fluoride_mg_l",
            "arsenic_ug_l",
            "open_defecation_rate",
            "toilet_access",
            "sewage_treatment_pct",
            "flooding",
            "avg_temperature_c",
            "avg_rainfall_mm",
            "avg_humidity_pct",
            "month"
        ]

        for feature in numerical_features:
            data[feature] = float(data[feature])

        # Create DataFrame
        input_df = pd.DataFrame(
            [[data[feature] for feature in required_features]],
            columns=required_features
        )

        # Preprocess
        X_processed = preprocessor.transform(input_df)

        # Prediction
        prediction = model.predict(X_processed)[0]

        # Probabilities
        probabilities = model.predict_proba(X_processed)[0]

        # Confidence
        confidence = float(np.max(probabilities)) * 100

        # Probability distribution
        probability_distribution = {}

        for disease, probability in zip(
            model.classes_,
            probabilities
        ):
            probability_distribution[disease] = round(
                float(probability) * 100,
                2
            )

        # SHAP explanation
        shap_explanations = get_shap_explanation(
            input_df,
            prediction
        )

        return {
            "predicted_disease": prediction,
            "confidence": round(confidence, 2),
            "probabilities": probability_distribution,
            "shap_explanation": shap_explanations
        }

    except Exception as e:

        return {
            "error": str(e)
        }, 500
    
if __name__ == "__main__":
    app.run(debug=True)