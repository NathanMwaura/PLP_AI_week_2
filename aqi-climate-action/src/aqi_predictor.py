"""
Air Quality Index (AQI) Prediction for SDG 13: Climate Action
Author: [Your Name]
Date: October 2025

This project uses supervised learning to predict air quality levels,
helping cities take preventive measures against pollution.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# 1. DATA GENERATION (Replace with real data)
# ============================================
print("=" * 60)
print("AIR QUALITY PREDICTION MODEL - SDG 13: CLIMATE ACTION")
print("=" * 60)


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic air quality data
    In production, replace with real data from:
    - World Air Quality Index: https://aqicn.org/api/
    - OpenAQ: https://openaq.org/
    - Kaggle AQI datasets
    """
    data = {
        "PM2.5": np.random.uniform(10, 200, n_samples),
        "PM10": np.random.uniform(20, 300, n_samples),
        "NO2": np.random.uniform(10, 100, n_samples),
        "SO2": np.random.uniform(5, 80, n_samples),
        "CO": np.random.uniform(0.1, 5, n_samples),
        "Temperature": np.random.uniform(10, 40, n_samples),
        "Humidity": np.random.uniform(30, 90, n_samples),
        "Wind_Speed": np.random.uniform(0, 20, n_samples),
        "Traffic_Volume": np.random.randint(100, 10000, n_samples),
        "Industrial_Activity": np.random.uniform(0, 100, n_samples),
    }

    df = pd.DataFrame(data)

    # Calculate AQI (simplified formula based on PM2.5)
    # Real AQI calculation is more complex: https://www.airnow.gov/aqi/aqi-calculator/
    df["AQI"] = (
        df["PM2.5"] * 0.5
        + df["PM10"] * 0.2
        + df["NO2"] * 0.15
        + df["SO2"] * 0.1
        + df["CO"] * 0.05
        + np.random.normal(0, 10, n_samples)
    )  # Add noise

    return df


# Generate data
print("\n📊 Loading and preparing data...")
df = generate_sample_data(1000)
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Statistics:")
print(df.describe())

print("\nCorrelation with AQI:")
correlations = df.corr()["AQI"].sort_values(ascending=False)
print(correlations)

# ============================================
# 3. DATA PREPROCESSING
# ============================================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Split features and target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Data preprocessing completed!")

# ============================================
# 4. MODEL TRAINING
# ============================================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "predictions": y_pred,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R² Score: {r2:.3f}")

# ============================================
# 5. MODEL EVALUATION
# ============================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "MAE": [results[m]["MAE"] for m in results],
        "RMSE": [results[m]["RMSE"] for m in results],
        "R2_Score": [results[m]["R2"] for m in results],
    }
)

print("\n", comparison_df.to_string(index=False))

best_model_name = comparison_df.loc[comparison_df["R2_Score"].idxmax(), "Model"]
print(f"\n🏆 Best Model: {best_model_name}")

# ============================================
# 6. FEATURE IMPORTANCE (for tree-based models)
# ============================================
if best_model_name in ["Random Forest", "Gradient Boosting"]:
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    best_model = results[best_model_name]["model"]
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": best_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\n", feature_importance.to_string(index=False))

# ============================================
# 7. PREDICTIONS & REAL-WORLD APPLICATION
# ============================================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

# Make predictions on test set
best_predictions = results[best_model_name]["predictions"]


# Show sample predictions with AQI categories
def categorize_aqi(aqi):
    """Categorize AQI values according to standard scale"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


sample_results = pd.DataFrame(
    {
        "Actual_AQI": y_test[:10].values,
        "Predicted_AQI": best_predictions[:10],
        "Actual_Category": [categorize_aqi(aqi) for aqi in y_test[:10].values],
        "Predicted_Category": [categorize_aqi(aqi) for aqi in best_predictions[:10]],
    }
)

print("\n", sample_results.to_string(index=False))

# ============================================
# 8. ETHICAL CONSIDERATIONS
# ============================================
print("\n" + "=" * 60)
print("ETHICAL CONSIDERATIONS & BIAS ANALYSIS")
print("=" * 60)

print("""
⚠️ Potential Biases:
1. Geographic Bias: Model trained on data from specific regions may not
   generalize to areas with different industrial/climate profiles
2. Temporal Bias: Seasonal variations may not be fully captured
3. Sensor Bias: Data quality depends on sensor calibration and placement

✅ Mitigation Strategies:
1. Collect diverse data across different geographic locations and seasons
2. Regular model retraining with updated data
3. Transparency in predictions with confidence intervals
4. Include uncertainty estimates in predictions
5. Validate with independent monitoring stations

🌍 Sustainability Impact:
- Enables early warning systems for pollution events
- Helps city planners make data-driven decisions
- Supports public health interventions
- Reduces healthcare costs from pollution-related illnesses
- Contributes to SDG 3 (Good Health) and SDG 11 (Sustainable Cities)
""")

# ============================================
# 9. ACTIONABLE INSIGHTS
# ============================================
print("\n" + "=" * 60)
print("ACTIONABLE INSIGHTS FOR POLICYMAKERS")
print("=" * 60)

print("""
📋 Recommendations based on model insights:

1. TRAFFIC MANAGEMENT
   - Implement congestion pricing during high-pollution forecast days
   - Promote public transport when AQI > 150 predicted

2. INDUSTRIAL REGULATION
   - Alert industries to reduce emissions on high-risk days
   - Schedule high-emission activities during favorable weather

3. PUBLIC HEALTH
   - Issue health advisories 24 hours before poor air quality
   - Alert vulnerable populations (children, elderly)

4. URBAN PLANNING
   - Use predictions to optimize green space placement
   - Design ventilation corridors in high-pollution zones

5. MONITORING EXPANSION
   - Deploy more sensors in areas with high prediction uncertainty
   - Focus on underrepresented geographic regions
""")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
print("\n💡 Next Steps:")
print("   1. Collect real-world data from air quality APIs")
print("   2. Deploy model as web application (Flask/Streamlit)")
print("   3. Integrate real-time weather data")
print("   4. Add confidence intervals to predictions")
print("   5. Create public dashboard for community awareness")
print("\n🌍 Together, AI and data science can help build cleaner,")
print("   healthier cities for everyone!")
