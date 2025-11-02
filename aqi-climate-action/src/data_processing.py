"""
Data Preprocessing Module for AQI Prediction
Handles data cleaning, transformation, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class AQIDataPreprocessor:
    """
    Handles all data preprocessing operations for AQI prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath, file_type='csv'):
        """
        Load data from various file formats
        
        Args:
            filepath (str): Path to data file
            file_type (str): File format ('csv', 'excel', 'json')
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if file_type == 'csv':
                df = pd.read_csv(filepath)
            elif file_type == 'excel':
                df = pd.read_excel(filepath)
            elif file_type == 'json':
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            print(f"✅ Data loaded successfully: {df.shape}")
            return df
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def check_data_quality(self, df):
        """
        Perform data quality checks
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            dict: Data quality report
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"Total Rows: {report['total_rows']}")
        print(f"Total Columns: {report['total_columns']}")
        print(f"Duplicate Rows: {report['duplicate_rows']}")
        print(f"\nMissing Values:")
        for col, missing in report['missing_values'].items():
            if missing > 0:
                pct = (missing / len(df)) * 100
                print(f"  {col}: {missing} ({pct:.2f}%)")
        
        return report
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values
                          ('mean', 'median', 'mode', 'drop', 'ffill')
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'mean':
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif strategy == 'median':
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif strategy == 'ffill':
            df_clean = df_clean.fillna(method='ffill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"✅ Missing values handled using '{strategy}' strategy")
        return df_clean
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers from specified columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to check for outliers (None = all numeric)
            method (str): Method for outlier detection ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(df_clean)
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / 
                                 df_clean[col].std())
                df_clean = df_clean[z_scores < threshold]
        
        removed_rows = initial_rows - len(df_clean)
        print(f"✅ Outliers removed: {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df_clean
    
    def create_features(self, df):
        """
        Create additional features from existing data
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with new features
        """
        df_features = df.copy()
        
        # Example feature engineering for AQI data
        
        # 1. Pollution Index (weighted combination of pollutants)
        if all(col in df.columns for col in ['PM2.5', 'PM10', 'NO2']):
            df_features['Pollution_Index'] = (
                df_features['PM2.5'] * 0.5 + 
                df_features['PM10'] * 0.3 + 
                df_features['NO2'] * 0.2
            )
        
        # 2. Temperature-Humidity interaction
        if 'Temperature' in df.columns and 'Humidity' in df.columns:
            df_features['Temp_Humidity_Interaction'] = (
                df_features['Temperature'] * df_features['Humidity'] / 100
            )
        
        # 3. Wind effectiveness (higher wind speed = better dispersion)
        if 'Wind_Speed' in df.columns:
            df_features['Wind_Effectiveness'] = 1 / (1 + df_features['Wind_Speed'])
        
        # 4. Traffic-Industrial combined impact
        if 'Traffic_Volume' in df.columns and 'Industrial_Activity' in df.columns:
            df_features['Human_Activity_Index'] = (
                (df_features['Traffic_Volume'] / 10000 + 
                 df_features['Industrial_Activity']) / 2
            )
        
        # 5. Pollutant ratios
        if 'PM2.5' in df.columns and 'PM10' in df.columns:
            df_features['PM2.5_to_PM10_Ratio'] = (
                df_features['PM2.5'] / (df_features['PM10'] + 1)  # +1 to avoid division by zero
            )
        
        new_features = [col for col in df_features.columns if col not in df.columns]
        print(f"✅ Created {len(new_features)} new features: {new_features}")
        
        return df_features
    
    def normalize_features(self, X_train, X_test=None, method='standard'):
        """
        Normalize features using specified method
        
        Args:
            X_train (array-like): Training features
            X_test (array-like): Test features (optional)
            method (str): Normalization method ('standard', 'minmax')
        
        Returns:
            tuple: Normalized train and test sets
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            print(f"✅ Features normalized using {method} scaling")
            return X_train_scaled, X_test_scaled, scaler
        
        print(f"✅ Features normalized using {method} scaling")
        return X_train_scaled, scaler
    
    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            test_size (float): Proportion of test set
            random_state (int): Random seed
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df, filepath):
        """
        Save processed data to file
        
        Args:
            df (pd.DataFrame): Processed dataframe
            filepath (str): Output file path
        """
        try:
            df.to_csv(filepath, index=False)
            print(f"✅ Processed data saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def get_preprocessing_summary(self, df_original, df_processed):
        """
        Generate summary of preprocessing steps
        
        Args:
            df_original (pd.DataFrame): Original dataframe
            df_processed (pd.DataFrame): Processed dataframe
        
        Returns:
            dict: Preprocessing summary
        """
        summary = {
            'original_shape': df_original.shape,
            'processed_shape': df_processed.shape,
            'rows_removed': df_original.shape[0] - df_processed.shape[0],
            'columns_added': df_processed.shape[1] - df_original.shape[1],
            'original_missing': df_original.isnull().sum().sum(),
            'processed_missing': df_processed.isnull().sum().sum()
        }
        
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Original Shape: {summary['original_shape']}")
        print(f"Processed Shape: {summary['processed_shape']}")
        print(f"Rows Removed: {summary['rows_removed']}")
        print(f"Columns Added: {summary['columns_added']}")
        print(f"Missing Values: {summary['original_missing']} → {summary['processed_missing']}")
        
        return summary


def preprocess_pipeline(filepath, target_column='AQI', save_path=None):
    """
    Complete preprocessing pipeline
    
    Args:
        filepath (str): Path to raw data
        target_column (str): Name of target variable
        save_path (str): Path to save processed data (optional)
    
    Returns:
        tuple: Preprocessed train/test data
    """
    print("\n" + "="*60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = AQIDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(filepath)
    if df is None:
        return None
    
    df_original = df.copy()
    
    # Quality check
    preprocessor.check_data_quality(df)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, strategy='mean')
    
    # Remove outliers
    df = preprocessor.remove_outliers(df, threshold=3.0)
    
    # Create features
    df = preprocessor.create_features(df)
    
    # Get summary
    preprocessor.get_preprocessing_summary(df_original, df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, target_column=target_column
    )
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = preprocessor.normalize_features(
        X_train, X_test
    )
    
    # Save processed data if path provided
    if save_path:
        preprocessor.save_processed_data(df, save_path)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING PIPELINE COMPLETE")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("This module provides functions for cleaning and preparing AQI data")
    print("\nTo use this module:")
    print("  from data_preprocessing import preprocess_pipeline")
    print("  X_train, X_test, y_train, y_test, scaler, features = preprocess_pipeline('data.csv')")