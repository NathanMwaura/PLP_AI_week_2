# Data Directory

## Structure

- `raw/`: Original, unmodified data files
- `processed/`: Cleaned and transformed data ready for modeling

## Data Sources

### For Production Use

1. **World Air Quality Index**: [https://aqicn.org/api/](https://aqicn.org/api/)
2. **OpenAQ**: [https://openaq.org/](https://openaq.org/)
3. **EPA Air Quality Data**: [https://www.epa.gov/outdoor-air-quality-data](https://www.epa.gov/outdoor-air-quality-data)

### Sample Datasets

- [Kaggle Air Quality Datasets](https://www.kaggle.com/datasets?search=air+quality)
- [UCI ML Repository - Air Quality](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

## Data Format

Expected columns:

- PM2.5 (μg/m³)
- PM10 (μg/m³)
- NO2 (μg/m³)
- SO2 (μg/m³)
- CO (mg/m³)
- Temperature (°C)
- Humidity (%)
- Wind_Speed (m/s)
- Traffic_Volume (vehicles/hour)
- Industrial_Activity (index 0-100)
- AQI (target variable, 0-500)

## Data Collection Guidelines

1. Ensure temporal coverage (at least 6 months)
2. Check for missing values
3. Verify sensor calibration
4. Document data sources
5. Include metadata (location, timezone, sensor type)
