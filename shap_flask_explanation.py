import joblib
import shap
import numpy as np


# Load deployment package
package = joblib.load(
    "models/deployment/water_quality_disease_model.pkl"
)

model = package["model"]
preprocessor = package["preprocessor"]


# Create optimized SHAP explainer
explainer = shap.TreeExplainer(
    model,
    feature_perturbation="tree_path_dependent"
)


# User-friendly feature names
FEATURE_NAME_MAP = {
    "num__water_quality_index": "Water Quality Index",
    "num__ph": "pH",
    "num__turbidity_ntu": "Turbidity (NTU)",
    "num__dissolved_oxygen_mg_l": "Dissolved Oxygen (mg/L)",
    "num__bod_mg_l": "BOD (mg/L)",
    "num__fecal_coliform_per_100ml": "Fecal Coliform (per 100 mL)",
    "num__total_coliform_per_100ml": "Total Coliform (per 100 mL)",
    "num__tds_mg_l": "TDS (mg/L)",
    "num__nitrate_mg_l": "Nitrate (mg/L)",
    "num__fluoride_mg_l": "Fluoride (mg/L)",
    "num__arsenic_ug_l": "Arsenic (µg/L)",
    "num__open_defecation_rate": "Open Defecation Rate (%)",
    "num__toilet_access": "Toilet Access",
    "num__sewage_treatment_pct": "Sewage Treatment (%)",
    "num__flooding": "Flooding",
    "num__avg_temperature_c": "Average Temperature (°C)",
    "num__avg_rainfall_mm": "Average Rainfall (mm)",
    "num__avg_humidity_pct": "Average Humidity (%)",
    "num__month": "Month",

    "cat__water_source_Borewell": "Water Source: Borewell",
    "cat__water_source_Canal": "Water Source: Canal",
    "cat__water_source_Pond": "Water Source: Pond",
    "cat__water_source_Rainwater": "Water Source: Rainwater",
    "cat__water_source_River": "Water Source: River",
    "cat__water_source_Tap": "Water Source: Tap",

    "cat__water_treatment_Boiling": "Water Treatment: Boiling",
    "cat__water_treatment_Chlorination": "Water Treatment: Chlorination",
    "cat__water_treatment_Filtration": "Water Treatment: Filtration",
    "cat__water_treatment_Untreated": "Water Treatment: Untreated",

    "cat__handwashing_practice_Always": "Handwashing: Always",
    "cat__handwashing_practice_Never": "Handwashing: Never",
    "cat__handwashing_practice_Sometimes": "Handwashing: Sometimes",

    "cat__season_Autumn": "Season: Autumn",
    "cat__season_Monsoon": "Season: Monsoon",
    "cat__season_Post-Monsoon": "Season: Post-Monsoon",
    "cat__season_Summer": "Season: Summer"
}


def get_shap_explanation(input_df, predicted_class):
    """
    Generate SHAP feature contributions for the predicted disease.
    """

    # Preprocess input
    X_processed = preprocessor.transform(input_df)

    # Calculate SHAP values
    shap_values = model.predict(
        X_processed,
        pred_contrib=True
    )

    # SHAP 0.51 multiclass format:
    # (samples, features, classes)
    shap_array = np.asarray(shap_values)

    print("SHAP ARRAY SHAPE:", shap_array.shape)
    
    # Number of features and classes
    n_features = len(preprocessor.get_feature_names_out())
    n_classes = len(model.classes_)
    
    # LightGBM returns:
    # features + 1 base value, for each class
    # Example: 38 × 8 = 304
    shap_array = shap_array.reshape(
        1,
        n_features + 1,
        n_classes
    )
    
    # Find predicted class index
    class_index = list(model.classes_).index(predicted_class)
    
    # Get SHAP values for predicted disease
    # Last feature position is the base/expected value
    class_shap_values = shap_array[0, :-1, class_index]

    # Get processed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Create feature contribution list
    explanations = []

    for feature, value in zip(
        feature_names,
        class_shap_values
    ):

        # Convert technical feature name to readable name
        readable_name = FEATURE_NAME_MAP.get(
            feature,
            feature
        )

        shap_value = float(value)

        explanations.append({
            "feature": readable_name,
            "shap_value": shap_value,
            "abs_shap": abs(shap_value)
        })

    # Sort by absolute SHAP importance
    explanations.sort(
        key=lambda x: x["abs_shap"],
        reverse=True
    )

    # Get top 10 features
    top_explanations = explanations[:10]

    # Maximum absolute SHAP value
    max_abs_shap = max(
        item["abs_shap"] for item in top_explanations
    )

    # Calculate relative bar width
    for item in top_explanations:

        if max_abs_shap > 0:
            item["bar_width"] = round(
                (item["abs_shap"] / max_abs_shap) * 100,
                2
            )
        else:
            item["bar_width"] = 0

        # CSS class based on SHAP direction
        if item["shap_value"] > 0:
            item["direction"] = "positive"
        else:
            item["direction"] = "negative"

    return top_explanations
