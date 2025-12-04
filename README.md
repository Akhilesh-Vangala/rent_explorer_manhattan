# SmartRent Manhattan – Rental Price Prediction

DS-GA 1007: Programming for Data Science  
Fall 2025 Final Project

---

## Project Overview

SmartRent Manhattan is a comprehensive rental price prediction system that uses machine learning to estimate monthly rent for Manhattan apartments based on property characteristics, amenities, and neighborhood. The project demonstrates end-to-end data science workflow including data preprocessing, feature engineering, model training, interpretability analysis with SHAP, and deployment via an interactive Streamlit web application.

### Goals

- Build accurate regression models to predict Manhattan rental prices
- Identify key features that influence rent using SHAP explainability
- Create an interactive tool for exploring rental affordability by neighborhood
- Apply best practices in Python programming, data science, and software engineering

---

## Dataset

**Source**: StreetEasy-style NYC rental listings dataset

**Description**: Manhattan rental listings with property features, amenities, and neighborhood information.

**Size**: 3,539 rental listings

**Features** (18 columns):
- `rental_id`: Unique identifier
- `rent`: Monthly rent (target variable)
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `size_sqft`: Property size in square feet
- `min_to_subway`: Distance to nearest subway station (minutes)
- `floor`: Floor number
- `building_age_yrs`: Building age in years
- `no_fee`: Broker fee indicator (0/1)
- `has_roofdeck`: Rooftop access (0/1)
- `has_washer_dryer`: In-unit washer/dryer (0/1)
- `has_doorman`: Doorman service (0/1)
- `has_elevator`: Elevator access (0/1)
- `has_dishwasher`: Dishwasher (0/1)
- `has_patio`: Patio/balcony (0/1)
- `has_gym`: Building gym (0/1)
- `neighborhood`: Manhattan neighborhood name
- `borough`: Borough (all Manhattan)

**Data Quality**: No missing values across all columns.

---

## Pipeline Architecture

The project follows a structured data science pipeline:

```
Raw Data (manhattan.csv)
    ↓
Preprocessing (cleaning, type casting, outlier handling)
    ↓
Feature Engineering (one-hot encoding, log transforms, rent per sqft)
    ↓
Train/Test Split (80/20, random_state=42)
    ↓
Model Training (Linear Regression, Random Forest, XGBoost)
    ↓
Model Evaluation (RMSE, MAE, R²)
    ↓
SHAP Explainability (TreeExplainer, summary plots)
    ↓
Model Persistence (best_model.pkl)
    ↓
Streamlit Deployment (interactive web app)
```

---

## Preprocessing

The preprocessing pipeline transforms raw data into model-ready features:

### 1. Column Removal
- Drop `rental_id` (non-predictive identifier)
- Drop `borough` (constant value: all Manhattan)

### 2. Type Casting
- Convert `bedrooms` from float to integer
- Convert `floor` from float to integer

### 3. Outlier Handling
- Clip `size_sqft` at upper bound of 3,500 sqft
- Clip `rent` at upper bound of $20,000

### 4. Categorical Encoding
- One-hot encode `neighborhood` (32 unique neighborhoods)
- Binary amenity features remain as 0/1 integers

### 5. Feature Engineering
- `rent_per_sqft`: Rent divided by size_sqft
- `log_rent`: Log transform of rent using numpy.log1p
- `log_size_sqft`: Log transform of size_sqft using numpy.log1p

**Final Feature Count**: 49 features (14 base + 32 neighborhood dummies + 3 engineered)

---

## Model Training & Evaluation

Three regression models were trained and evaluated:

### Models

1. **Linear Regression**: Baseline model with no hyperparameter tuning
2. **Random Forest Regressor**: Ensemble model with 200 trees, random_state=42
3. **XGBoost Regressor**: Gradient boosting with 300 estimators, learning_rate=0.05, max_depth=6

### Evaluation Metrics

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 648.58 | 423.33 | 0.9554 |
| Random Forest | 10.20 | 3.06 | 0.9999 |
| **XGBoost** | **8.34** | **2.51** | **0.9999** |

**Selected Model**: XGBoost achieved the best performance with the lowest RMSE and MAE, and highest R² score of 0.9999.

### Model Interpretation

XGBoost predictions are within approximately $8 of actual rent on average, demonstrating exceptional accuracy. The near-perfect R² indicates the model captures nearly all variance in rental prices based on the available features.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) values were computed using TreeExplainer to interpret the XGBoost model's predictions.

### Analysis

- **Summary Plot (Beeswarm)**: Displays feature importance and impact direction for all test samples
- **Bar Plot**: Shows mean absolute SHAP values, ranking features by overall importance

### Key Insights

SHAP analysis reveals which features have the strongest impact on rental price predictions, enabling transparency and trust in the model's decision-making process. The visualizations are generated in the modeling notebook and can be integrated into reporting or stakeholder presentations.

---

## Streamlit Application

An interactive web application allows users to:

1. **Input Property Features**: Specify bedrooms, bathrooms, size, location, floor, building age, and amenities
2. **Predict Rent**: Generate instant rental price predictions using the trained XGBoost model
3. **Neighborhood Insights**: Compare predicted rent to neighborhood average
4. **Visualize Results**: View bar chart comparing predicted vs. average rent for the selected neighborhood

### Features

- User-friendly form interface for property inputs
- Real-time predictions using saved model
- Neighborhood-level rent statistics
- Matplotlib visualizations embedded in the app
- Formatted currency output

---

## Project Structure

```
smartrent_manhattan/
├── data/
│   ├── raw/
│   │   └── manhattan.csv          # Original dataset
│   └── processed/                  # Placeholder for processed data
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   └── 02_modeling.ipynb          # Model training and evaluation
├── src/
│   ├── __init__.py                # Package initialization
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── modeling.py                # Model training and evaluation functions
│   ├── visualization.py           # Visualization utilities (unused)
│   └── utils.py                   # Helper functions (unused)
├── app/
│   └── Rent_Estimation.py         # Streamlit web application
├── outputs/
│   ├── models/
│   │   └── best_model.pkl         # Saved XGBoost model
│   ├── shap/                      # Placeholder for SHAP outputs
│   └── metrics/                   # Placeholder for metrics
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone or download the project directory

2. Navigate to the project folder:
```bash
cd smartrent_manhattan
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy==1.24.3`
- `pandas==2.0.3`
- `matplotlib==3.7.2`
- `scikit-learn==1.3.0`
- `shap==0.42.1`
- `streamlit==1.25.0`
- `xgboost==1.7.6`
- `joblib==1.3.2`

---

## Usage

### Running the Notebooks

#### 1. Exploratory Data Analysis

Open and run the EDA notebook to inspect the dataset:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Execute all cells to view:
- Dataset structure and statistics
- Missing value analysis
- Distributions and correlations
- Visualizations

#### 2. Model Training

Open and run the modeling notebook:

```bash
jupyter notebook notebooks/02_modeling.ipynb
```

Execute all cells to:
- Load and preprocess data
- Split into train/test sets
- Train all three models
- Evaluate and compare performance
- Generate SHAP visualizations
- Save the best model to `outputs/models/best_model.pkl`

### Running the Streamlit App

Launch the interactive web application:

```bash
streamlit run app/Rent_Estimation.py
```

The app will open in your default web browser at `http://localhost:8501`.

**Usage**:
1. Enter property details in the input form
2. Select a neighborhood from the dropdown
3. Click "Predict Rent"
4. View predicted rent, neighborhood comparison, and visualization

---

## Key Functions

### Preprocessing (`src/preprocessing.py`)

- `load_raw_data(path)`: Load CSV dataset
- `drop_columns(df)`: Remove non-predictive columns
- `cast_types(df)`: Convert data types
- `handle_outliers(df)`: Clip extreme values
- `encode_categoricals(df)`: One-hot encode neighborhoods
- `feature_engineering(df)`: Create derived features
- `preprocess(df)`: Complete preprocessing pipeline

### Modeling (`src/modeling.py`)

- `train_test_split_df(df, target_column)`: Split data with 80/20 ratio
- `train_linear_regression(X_train, y_train)`: Train baseline model
- `train_random_forest(X_train, y_train)`: Train Random Forest
- `train_xgboost(X_train, y_train)`: Train XGBoost
- `evaluate_model(model, X_test, y_test)`: Compute RMSE, MAE, R²
- `model_comparison_table(results)`: Generate comparison DataFrame

---

## Results Summary

- **Best Model**: XGBoost Regressor
- **Performance**: RMSE of 8.34, MAE of 2.51, R² of 0.9999
- **Prediction Accuracy**: Within $8 of actual rent on average
- **Feature Count**: 49 features after preprocessing
- **Training Samples**: 2,831
- **Test Samples**: 708
- **Neighborhoods**: 32 Manhattan neighborhoods

---

## Future Enhancements

Potential extensions to the project:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Additional feature engineering (interaction terms, polynomial features)
- Time series analysis if temporal data becomes available
- Integration of external data sources (crime rates, school ratings, transit scores)
- Model deployment via cloud services (AWS, Azure, GCP)
- API endpoint for programmatic access to predictions

---

## Course Information

**Course**: DS-GA 1007 – Programming for Data Science  
**Institution**: New York University, Center for Data Science  
**Semester**: Fall 2025

This project demonstrates proficiency in:
- Python programming fundamentals
- Data manipulation with Pandas and NumPy
- Data visualization with Matplotlib
- Machine learning with scikit-learn and XGBoost
- Model interpretability with SHAP
- Web application development with Streamlit
- Software engineering best practices
- Version control and project organization

---

## License

This project is submitted as coursework for DS-GA 1007 at NYU.
