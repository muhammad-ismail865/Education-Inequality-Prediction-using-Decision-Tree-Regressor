# Education Inequality Prediction using Decision Tree Regressor

A Machine Learning project that analyzes educational inequality factors and predicts **average test score percentage** using a **Decision Tree Regressor**.
The project demonstrates complete preprocessing of categorical and numerical data, feature engineering, model training, and evaluation using Scikit-learn.

# 📌 Project Overview

This project uses an education inequality dataset containing information about:

* School funding
* Student-teacher ratio
* Internet access
* Minority percentage
* Low-income student percentage
* Dropout rate
* School type
* Grade level
* State information

The goal is to predict:

> **`avg_test_score_percent`**

using a supervised machine learning regression model.

# 🚀 Features

* Data preprocessing using Pandas
* Label Encoding for ordinal categorical data
* One-Hot Encoding for high-cardinality categorical features
* Feature scaling using StandardScaler
* Train-Test split
* Decision Tree Regression model
* Performance evaluation using:

  * Mean Absolute Error (MAE)
  * R² Score

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

# 📂 Dataset

Dataset file used:

education_inequality_data.csv

Expected columns in dataset:

| Column Name             | Description                   |
| ----------------------- | ----------------------------- |
| id                      | Unique identifier             |
| state                   | State name                    |
| school_name             | School name                   |
| school_type             | Public/Private etc            |
| grade_level             | Grade category                |
| funding_per_student_usd | Funding per student           |
| student_teacher_ratio   | Student-teacher ratio         |
| percent_low_income      | Low-income student percentage |
| percent_minority        | Minority student percentage   |
| internet_access_percent | Internet access percentage    |
| dropout_rate_percent    | Dropout rate                  |
| avg_test_score_percent  | Target variable               |

---

# ⚙️ Data Preprocessing Steps

## 1. Remove Unnecessary Column

```python
df.drop(columns=["id"], inplace=True)

## 2. Label Encoding

Applied on:

* `state`
* `school_type`
* `grade_level`

## 3. One-Hot Encoding

Applied on:

* `school_name`

OneHotEncoder()

## 4. Feature Scaling

Scaled numerical features using:

python
StandardScaler()

Scaled columns:

* funding_per_student_usd
* student_teacher_ratio
* percent_low_income
* percent_minority
* internet_access_percent
* dropout_rate_percent

# 🤖 Machine Learning Model

## Decision Tree Regressor

python
DecisionTreeRegressor(max_depth=6)


The model predicts the educational test score percentage based on socioeconomic and school-related features.

# 📊 Model Evaluation

The following metrics are used:

## Mean Absolute Error (MAE)

Measures average prediction error.

python
mean_absolute_error()

## R² Score

Measures model performance and goodness of fit.

python
r2_score()

# 📈 Example Output

bash
Mean Absolute Error: 4.21
r2_score 0.87

# 💡 Future Improvements

* Hyperparameter tuning
* Random Forest Regressor
* XGBoost Regressor
* Feature importance visualization
* Model deployment using Flask or Streamlit
* Cross-validation

# 🧠 Learning Outcomes

Through this project, you can learn:

* Data preprocessing techniques
* Encoding categorical data
* Feature scaling
* Regression modeling
* Decision Tree algorithms
* Model evaluation metrics
