import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("C:/Users\HAFIZ TECH/Downloads/education_inequality_data.csv")

# Remove unnecessary ID column
df.drop(columns=["id"], inplace=True)

# Label Encoding for categorical ordinal-like features
le = LabelEncoder()
df["state"] = le.fit_transform(df["state"])
df["school_type"] = le.fit_transform(df["school_type"])
df["grade_level"] = le.fit_transform(df["grade_level"])

# One Hot Encoding for school_name (high cardinality categorical feature)
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded = encoder.fit_transform(df[["school_name"]])

# Get new column names after encoding
encoded_col = encoder.get_feature_names_out(["school_name"])

# Convert encoded array into DataFrame
encoded_df = pd.DataFrame(encoded, columns=encoded_col, index=df.index)

# Remove original categorical column after encoding
df.drop(columns=["school_name"], inplace=True)

# Combine original dataset with encoded features
dataset = pd.concat([df, encoded_df], axis=1)

# Drop selected numerical columns from dataset (will be re-added after scaling)
dataset.drop(columns=[
    'funding_per_student_usd',
    'student_teacher_ratio',
    'percent_low_income',
    'percent_minority',
    'internet_access_percent',
    'dropout_rate_percent'
], inplace=True)

# Standard scaling for numerical features
scaler = StandardScaler()
scaled = scaler.fit_transform(df[[
    'funding_per_student_usd',
    'student_teacher_ratio',
    'percent_low_income',
    'percent_minority',
    'internet_access_percent',
    'dropout_rate_percent'
]])

# Convert scaled data back into DataFrame
new_df = pd.DataFrame(scaled, columns=[
    'funding_per_student_usd',
    'student_teacher_ratio',
    'percent_low_income',
    'percent_minority',
    'internet_access_percent',
    'dropout_rate_percent'
])

# Combine encoded + scaled dataset
new_dataset = pd.concat([dataset, new_df], axis=1)

# Split features (X) and target variable (y)
X = new_dataset.drop("avg_test_score_percent", axis=1)
y = new_dataset["avg_test_score_percent"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regressor model
model = DecisionTreeRegressor(max_depth=6)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_predict = model.predict(X_test)

# Evaluate model using Mean Absolute Error
mae = mean_absolute_error(y_test, y_predict)
print("Mean Absolute Error:", mae)

# Evaluate model using R2 score
r2 = r2_score(y_test, y_predict)
print("r2_score", r2)