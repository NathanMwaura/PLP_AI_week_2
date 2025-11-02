"""
Utility Functions for AQI Prediction Project
Helper functions for common tasks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib


# ============================================
# AQI CATEGORIZATION FUNCTIONS
# ============================================

def categorize_aqi(aqi_value):
    """
    Categorize AQI value according to EPA standards
    
    Args:
        aqi_value (float): AQI numerical value
    
    Returns:
        str: AQI category
    """
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_aqi_color(aqi_value):
    """
    Get color code for AQI visualization
    
    Args:
        aqi_value (float): AQI numerical value
    
    Returns:
        str: Color hex code
    """
    if aqi_value <= 50:
        return "#00E400"  # Green
    elif aqi_value <= 100:
        return "#FFFF00"  # Yellow
    elif aqi_value <= 150:
        return "#FF7E00"  # Orange
    elif aqi_value <= 200:
        return "#FF0000"  # Red
    elif aqi_value <= 300:
        return "#8F3F97"  # Purple
    else:
        return "#7E0023"  # Maroon


def get_health_advisory(aqi_value):
    """
    Get health advisory message based on AQI
    
    Args:
        aqi_value (float): AQI numerical value
    
    Returns:
        str: Health advisory message
    """
    category = categorize_aqi(aqi_value)
    
    advisories = {
        "Good": "Air quality is satisfactory. Outdoor activities are safe for everyone.",
        "Moderate": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. Reduce prolonged or heavy outdoor exertion.",
        "Unhealthy": "Everyone may experience health effects. Sensitive groups should avoid outdoor activities.",
        "Very Unhealthy": "Health alert: everyone may experience serious health effects. Everyone should avoid outdoor activities.",
        "Hazardous": "Health warnings of emergency conditions. Everyone should remain indoors."
    }
    
    return advisories.get(category, "Unknown AQI level")


# ============================================
# DATA GENERATION FUNCTIONS
# ============================================

def generate_synthetic_aqi_data(n_samples=1000, random_state=42):
    """
    Generate synthetic AQI dataset for testing
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed
    
    Returns:
        pd.DataFrame: Synthetic AQI data
    """
    np.random.seed(random_state)
    
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
    
    # Calculate AQI (simplified formula)
    df['AQI'] = (
        df['PM2.5'] * 0.5 + 
        df['PM10'] * 0.2 + 
        df['NO2'] * 0.15 + 
        df['SO2'] * 0.1 + 
        df['CO'] * 0.05 +
        np.random.normal(0, 10, n_samples)
    )
    
    return df


# ============================================
# MODEL PERSISTENCE
# ============================================

def save_model(model, filepath):
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        filepath (str): Path to save model
    """
    try:
        joblib.dump(model, filepath)
        print(f"✅ Model saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")


def load_model(filepath):
    """
    Load trained model from disk
    
    Args:
        filepath (str): Path to saved model
    
    Returns:
        Trained model object
    """
    try:
        model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


# ============================================
# EVALUATION METRICS
# ============================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
    
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_absolute_error, 
        mean_squared_error, 
        r2_score,
        mean_absolute_percentage_error
    )
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }
    
    return metrics


def print_metrics(metrics, title="Model Performance"):
    """
    Pretty print evaluation metrics
    
    Args:
        metrics (dict): Dictionary of metrics
        title (str): Title for the metrics display
    """
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:10.4f}")
    print("="*60)


# ============================================
# VISUALIZATION HELPERS
# ============================================

def plot_prediction_comparison(y_true, y_pred, save_path=None):
    """
    Plot actual vs predicted values
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        save_path (str): Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual AQI', fontsize=12)
    plt.ylabel('Predicted AQI', fontsize=12)
    plt.title('Actual vs Predicted AQI', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    """
    Plot feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int): Number of top features to display
        save_path (str): Path to save plot (optional)
    """
    if not hasattr(model, 'feature_importances_'):
        print("❌ Model doesn't have feature importances")
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plot residual analysis
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        save_path (str): Path to save plot (optional)
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual scatter plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, color='purple')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted AQI', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, color='teal', alpha=0.7, edgecolor='black')
    axes[1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {residuals.mean():.2f}')
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


# ============================================
# REPORTING FUNCTIONS
# ============================================

def generate_prediction_report(y_true, y_pred, feature_names=None, 
                               model=None, save_path=None):
    """
    Generate comprehensive prediction report
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        feature_names (list): Feature names (optional)
        model: Trained model (optional)
        save_path (str): Path to save report (optional)
    
    Returns:
        dict: Report data
    """
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create report
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': metrics,
        'sample_predictions': {
            'actual': y_true[:10].tolist() if hasattr(y_true, 'tolist') else list(y_true[:10]),
            'predicted': y_pred[:10].tolist() if hasattr(y_pred, 'tolist') else list(y_pred[:10])
        }
    }
    
    # Add feature importance if available
    if model and feature_names and hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_names, model.feature_importances_))
        report['feature_importance'] = importance
    
    # Print report
    print_metrics(metrics, "Prediction Report")
    
    # Save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"✅ Report saved to {save_path}")
    
    return report


# ============================================
# DATA VALIDATION
# ============================================

def validate_input_data(df, required_columns):
    """
    Validate input dataframe has required columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return False, missing
    
    print("✅ All required columns present")
    return True, []


def check_value_ranges(df, column_ranges):
    """
    Check if values are within expected ranges
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_ranges (dict): Dict of {column: (min, max)}
    
    Returns:
        dict: Validation results
    """
    results = {}
    
    for column, (min_val, max_val) in column_ranges.items():
        if column in df.columns:
            out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
            results[column] = {
                'out_of_range_count': int(out_of_range),
                'percentage': float(out_of_range / len(df) * 100)
            }
    
    return results


# ============================================
# LOGGING HELPER
# ============================================

def log_message(message, level='INFO'):
    """
    Log message with timestamp
    
    Args:
        message (str): Message to log
        level (str): Log level (INFO, WARNING, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ============================================
# MAIN - EXAMPLES
# ============================================

if __name__ == "__main__":
    print("Utility Functions for AQI Prediction")
    print("\nAvailable functions:")
    print("  - categorize_aqi(aqi_value)")
    print("  - get_health_advisory(aqi_value)")
    print("  - generate_synthetic_aqi_data(n_samples)")
    print("  - save_model(model, filepath)")
    print("  - load_model(filepath)")
    print("  - calculate_metrics(y_true, y_pred)")
    print("  - plot_prediction_comparison(y_true, y_pred)")
    print("  - plot_feature_importance(model, feature_names)")
    print("  - generate_prediction_report(y_true, y_pred)")
    
    print("\n" + "="*60)
    print("Example: Testing AQI categorization")
    print("="*60)
    
    # Test AQI categorization
    test_values = [30, 75, 125, 175, 225, 350]
    for aqi in test_values:
        category = categorize_aqi(aqi)
        color = get_aqi_color(aqi)
        advisory = get_health_advisory(aqi)
        print(f"\nAQI: {aqi}")
        print(f"  Category: {category}")
        print(f"  Color: {color}")
        print(f"  Advisory: {advisory}")
    
    print("\n✅ Utils module loaded successfully!")