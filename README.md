# JC-Devcomm-Task1

# Spaceship Titanic Prediction

## Overview
This project focuses on predicting whether passengers aboard the Spaceship Titanic were transported to an alternate dimension. The dataset includes various features about passengers, such as cabin details, spending habits, and personal information. The goal is to build a machine learning model that accurately predicts the "Transported" status of each passenger.

## Project Structure
- **Notebook**: `Task1.ipynb` contains the complete code for data preprocessing, model training, and evaluation.
- **Data Files**:
  - `train.csv`: Training dataset with passenger features and the target variable "Transported".
  - `test.csv`: Test dataset for making predictions.
  - `submission_best.csv`: Final predictions saved in the required format.

## Key Steps
1. **Data Preprocessing**:
   - Split the "Cabin" feature into "Deck", "CabinNum", and "Side".
   - Extract the "Group" from the "PassengerId".
   - Handle missing values using `SimpleImputer`.
   - Encode categorical variables using `OrdinalEncoder`.
   - Engineer new features like "TotalSpend" and "SpentAny".

2. **Model Training**:
   - Evaluated multiple models including:
     - Random Forest
     - XGBoost
     - LightGBM
     - CatBoost
   - Used `GridSearchCV` for hyperparameter tuning.
   - Selected CatBoost as the best-performing model based on ROC AUC scores.

3. **Evaluation**:
   - Achieved a validation ROC AUC score of **0.8949** with CatBoost.
   - Performed cross-validation to ensure model robustness.

4. **Prediction**:
   - Generated predictions for the test dataset and saved them in `submission_best.csv`.

## Results
- **Best Model**: CatBoostClassifier with tuned hyperparameters.
- **Validation ROC AUC**: 0.8949.
- **Cross-validation Mean ROC AUC**: 0.8414.

## Dependencies
- Python 3.12
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost

## How to Run
1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Open and run the `Task1.ipynb` notebook in Jupyter or any compatible environment.
4. The final predictions will be saved as `submission_best.csv`.
