"""
Visualization Dashboard for AQI Prediction Model
Creates publication-ready charts for the project report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generate data (same as main script)
np.random.seed(42)

def generate_sample_data(n_samples=1000):
    data = {
        'PM2.5': np.random.uniform(10, 200, n_samples),
        'PM10': np.random.uniform(20, 300, n_samples),
        'NO2': np.random.uniform(10, 100, n_samples),
        'SO2': np.random.uniform(5, 80, n_samples),
        'CO': np.random.uniform(0.1, 5, n_samples),
        'Temperature': np.random.uniform(10, 40, n_samples),
        'Humidity': np.random.uniform(30, 90, n_samples),
        'Wind_Speed': np.random.uniform(0, 20, n_samples),
        'Traffic_Volume': np.random.randint(100, 10000, n_samples),
        'Industrial_Activity': np.random.uniform(0, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['AQI'] = (df['PM2.5'] * 0.5 + df['PM10'] * 0.2 + df['NO2'] * 0.15 + 
                 df['SO2'] * 0.1 + df['CO'] * 0.05 + np.random.normal(0, 10, n_samples))
    return df

df = generate_sample_data(1000)
X = df.drop('AQI', axis=1)
y = df['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# 1. Correlation Heatmap
ax1 = plt.subplot(3, 3, 1)
corr_matrix = df.corr()
sns.heatmap(corr_matrix[['AQI']], annot=True, fmt='.2f', cmap='RdYlGn_r', 
            center=0, ax=ax1, cbar_kws={'label': 'Correlation'})
ax1.set_title('Feature Correlation with AQI', fontsize=12, fontweight='bold')

# 2. Feature Importance
ax2 = plt.subplot(3, 3, 2)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)
ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
ax2.set_xlabel('Importance Score')
ax2.set_title('Feature Importance Analysis', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. AQI Distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(y, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax3.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.1f}')
ax3.set_xlabel('AQI Value')
ax3.set_ylabel('Frequency')
ax3.set_title('AQI Distribution in Dataset', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Actual vs Predicted
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(y_test, y_pred, alpha=0.5, s=30)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax4.set_xlabel('Actual AQI')
ax4.set_ylabel('Predicted AQI')
ax4.set_title('Actual vs Predicted AQI', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Residual Plot
ax5 = plt.subplot(3, 3, 5)
residuals = y_test - y_pred
ax5.scatter(y_pred, residuals, alpha=0.5, s=30, color='purple')
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted AQI')
ax5.set_ylabel('Residuals')
ax5.set_title('Residual Analysis', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Error Distribution
ax6 = plt.subplot(3, 3, 6)
ax6.hist(residuals, bins=30, color='teal', alpha=0.7, edgecolor='black')
ax6.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean Error: {residuals.mean():.2f}')
ax6.set_xlabel('Prediction Error')
ax6.set_ylabel('Frequency')
ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 7. AQI Categories Pie Chart
ax7 = plt.subplot(3, 3, 7)
def categorize_aqi(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy (Sensitive)"
    elif aqi <= 200: return "Unhealthy"
    else: return "Very Unhealthy"

categories = [categorize_aqi(aqi) for aqi in y]
category_counts = pd.Series(categories).value_counts()
colors_pie = ['green', 'yellow', 'orange', 'red', 'purple']
ax7.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
        colors=colors_pie[:len(category_counts)], startangle=90)
ax7.set_title('AQI Category Distribution', fontsize=12, fontweight='bold')

# 8. PM2.5 vs AQI
ax8 = plt.subplot(3, 3, 8)
scatter = ax8.scatter(df['PM2.5'], df['AQI'], c=df['AQI'], cmap='RdYlGn_r', 
                     alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax8.set_xlabel('PM2.5 Concentration (μg/m³)')
ax8.set_ylabel('AQI')
ax8.set_title('PM2.5 Impact on AQI', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax8, label='AQI Value')
ax8.grid(alpha=0.3)

# 9. Model Performance Metrics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))

metrics_text = f"""
MODEL PERFORMANCE METRICS

📊 Random Forest Regressor

Accuracy Metrics:
• R² Score: {r2:.4f}
• MAE: {mae:.2f}
• RMSE: {rmse:.2f}

Dataset Info:
• Training Samples: {len(X_train)}
• Testing Samples: {len(X_test)}
• Features: {len(X.columns)}

SDG Impact:
✓ Climate Action (SDG 13)
✓ Good Health (SDG 3)
✓ Sustainable Cities (SDG 11)
"""

ax9.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Air Quality Index (AQI) Prediction Model - Comprehensive Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('aqi_prediction_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Dashboard saved as 'aqi_prediction_dashboard.png'")
print("\n📊 All visualizations generated successfully!")
print("\n💡 Use these charts in your:")
print("   • Project presentation")
print("   • README.md file")
print("   • Technical report")
print("   • Pitch deck")
plt.show()