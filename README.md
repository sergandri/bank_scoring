# Hackathon Project

bank-scoring is a machine learning-based application for Hackathon 2024. This README 
provides instructions for running the application both locally and using Docker.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Running Locally](#running-locally)
3. [Running with Docker](#running-with-docker)
4. [Application Design](#application-design)
5. [License](#license)

---

## Prerequisites

### Local Environment:
- Python 3.12 or later
- `poetry` for dependency management
- `pip` (if you are not using Poetry)

### Docker Environment:
- Docker installed on your machine 

---

## Running Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sergandri/bank_scoring.git
2. Install Dependencies:

- If ur using Poetry:
-  ```bash
   poetry install
- If ur using pip:
- ```bash 
  pip install -r requirements.txt

3. Put train and test data into 'input_data' (default) directory

4. Run the Application with poetry env:
- ```bash 
   poetry run uvicorn main:app --host 0.0.0.0 --port 8000 

or 
- ```bash 
   uvicorn main:app --host 0.0.0.0 --port 8000

5. Access the Application:

- Open your browser and go to http://localhost:8000.

- API documentation will be available at http://localhost:8000/docs.

- Use API endpoints to launch training and prediction pipelines

- Use local directories to check results 

## Running with Docker

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sergandri/bank_scoring.git
2. Put train and test data into 'input_data' (default) directory
3. Build the Docker Image:
   ```bash
   docker build -t bank-scoring .

4. Run the Docker Container:
   ```bash 
   docker run -p 8000:8000 bank-scoring

5. Access the Application:

- Open your browser and go to http://localhost:8000.

- API documentation will be available at http://localhost:8000/docs.

- Use API endpoints to launch training and prediction pipelines

- Use docker exec /bin/bash to check results 

## Application Design
### 1. Application Structure
The application follows a modular architecture, organizing different functionalities into separate components for clarity, reusability, and scalability:


#### **Data and workflows configs** :
*src/tools/data_config.py*
#### **Preprocessing and Feature Engineering Modules** :
*src/ml/preprocessing.py*
- Handles data cleaning, type conversions, missing value handling, and outlier treatment.
- Integrates with external configurations for customizable preprocessing logic.

*src/ml/feature_engineering.py*
- Implements feature engineering to generate advanced features.

#### **Feature Analysis Module**:
*src/analyzers.py*
- Performs statistical analysis to calculate IV, Gini, PSI.
- Filters and selects features based on predefined thresholds.
- Calculates correlation and VIF for feature pruning.

#### **Binning Module**:
*src/binning.py*
- Utilizes optimal binning techniques to transform continuous features into discrete bins.
- Applies WoE transformation for linearity and monotonicity in modeling.
- Stores binning models for reproducibility and applies transformations to new datasets.

#### **Modeling Module**:
- Trains models such as Logistic Regression and GBDT
- Provides hyperparameter optimization using Optuna.
- Provides methods for saving, loading, and evaluating models.
- Generates predictions for labeled and unlabeled datasets.

#### **API Integration**:
- Built with FastAPI.
- Provides endpoints for triggering workflows.
- 
#### **Directories and artifacts**:
*/input_data* 
- df_BKI_30k.csv - train feastures
- df_target_30k.csv - train target
- df_test_notarget_10k.csv - test wo target

*/output_data*
- all_binnings.pkl - dict with binning for features
- top_35_feature_names.pkl - list with GBDT features
- lr_features.pkl - list with LR features
- final_model.pkl - LR model
- cb_model.cbm - GBDT CatBoost model
- lr_train_result.csv - LR train targets
- lr_test_result.csv - LR test targets
- catboost_train_result.csv - CatBoost train targets
- catboost_test_result.csv - CatbBoost test targets
- correlation_filtered_features.csv - table with filtered features
- factor_analysis.csv - table with IV GINI PSI analysis 
- vif_filtered_features.csv - table with filtered features
- real_cb_x_test_result.csv - dataset for fitting
- real_lr_x_test_result.csv - dataset for fitting
- real_cb_test_result.csv - TEST RESULT W GBDT CB MODEL
- real_lr_test_result.csv - TEST RESULT W BASELINE LR MODEL

---

### 2. Workflow

#### **1. Data Input**:
- The application accepts raw credit data in CSV format.
- Separate datasets for training, testing, and targets can be provided.

#### **2. Data Preprocessing**:
- Handles missing values, outliers, and type conversions.
- Performs advanced feature engineering to create meaningful metrics like overdue ratios, utilization rates, and risk scores.

#### **3. Feature Selection**:
- Analyzes feature importance using metrics like IV and Gini.
- Filters features based on correlation and multicollinearity thresholds.

#### **4. Model Training**:
- Models are trained using processed data, with hyperparameter optimization to enhance performance.
- Supports Logistic Regression and CatBoost classifiers with distinct pipelines for categorical and numerical data.

#### **5. Evaluation**:
- Evaluates models using metrics like Gini coefficient and ROC-AUC.
- Visualizes feature importance and ROC curves for interpretability.

#### **6. Prediction**:
- Generates predictions for both labeled and unlabeled datasets.
- Stores results in CSV files for downstream processing.

#### **7. API Interaction**:
- Exposes RESTful endpoints for initiating workflows and retrieving results.
- Allows users to interact with the application in UI with Swagger.

---

## 4. Deployment

### **Local Deployment**:
- Users can run the application locally using `poetry` or by directly installing dependencies via `requirements.txt`.

### **Docker Deployment**:
- The application is fully Dockerized, enabling hassle-free deployment without dependency conflicts.
- Docker ensures consistency across environments, from development to production.
- 
