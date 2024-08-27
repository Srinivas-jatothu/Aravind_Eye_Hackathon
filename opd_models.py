import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.impute import SimpleImputer
import joblib

# Read the CSV file into a DataFrame
df = pd.read_csv('Final_Year - Sheet1 (2) (1).csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Extract additional features
df['Day'] = df['Date'].dt.day
df['Day_Of_Week'] = df['Date'].dt.weekday
df['Month_Number'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Prepare features and target variables
features = ['Day', 'Day_Of_Week', 'Month_Number', 'Year', 'Type_of_Day']
X = df[features]
y_new = df['Actual_New']
y_review = df['Actual_Review']

# Handle missing values for features
feature_imputer = SimpleImputer(strategy='mean')
X_imputed = feature_imputer.fit_transform(X)

# Handle missing values for targets
target_imputer_new = SimpleImputer(strategy='mean')
target_imputer_review = SimpleImputer(strategy='mean')
y_new_imputed = target_imputer_new.fit_transform(y_new.values.reshape(-1, 1)).ravel()
y_review_imputed = target_imputer_review.fit_transform(y_review.values.reshape(-1, 1)).ravel()

# Split data into training and test sets
X_train, X_test, y_new_train, y_new_test = train_test_split(X_imputed, y_new_imputed, test_size=0.2, random_state=42)
X_train, X_test, y_review_train, y_review_test = train_test_split(X_imputed, y_review_imputed, test_size=0.2, random_state=42)

# Define models
models = {
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'SVR': SVR(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'GaussianProcessRegressor': GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2)))
}

# Train and evaluate models
for name, model in models.items():
    # Train model for Actual_New
    model.fit(X_train, y_new_train)
    y_new_pred = model.predict(X_test)
    print(f"{name} - Actual_New Performance:")
    print("MSE:", mean_squared_error(y_new_test, y_new_pred))
    print("R^2:", r2_score(y_new_test, y_new_pred))
    print("Accuracy:", sum(abs(y_new_test - y_new_pred) / y_new_test < 0.1) / len(y_new_test))

    # Train model for Actual_Review
    model.fit(X_train, y_review_train)
    y_review_pred = model.predict(X_test)
    print(f"{name} - Actual_Review Performance:")
    print("MSE:", mean_squared_error(y_review_test, y_review_pred))
    print("R^2:", r2_score(y_review_test, y_review_pred))
    print("Accuracy:", sum(abs(y_review_test - y_review_pred) / y_review_test < 0.1) / len(y_review_test))

    # Save models
    joblib.dump(model, f'model_actual_new_{name}.pkl')
    joblib.dump(model, f'model_actual_review_{name}.pkl')

# Example user input
user_input = {
    'Day': 2,  # Example day
    'Day_Of_Week': 1,  # Example day of the week (Monday)
    'Month_Number': 1,
    'Year': 2024,
    'Type_of_Day': 2  # Example type of day (numeric)
}

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Impute missing values in user input
user_input_imputed = feature_imputer.transform(user_input_df)

# Predict using each model
for name, model in models.items():
    # Load models
    model_new = joblib.load(f'model_actual_new_{name}.pkl')
    model_review = joblib.load(f'model_actual_review_{name}.pkl')

    # Predict
    predicted_new = model_new.predict(user_input_imputed)
    predicted_review = model_review.predict(user_input_imputed)

    print(f"Predicted Actual_New with {name}:", predicted_new[0])
    print(f"Predicted Actual_Review with {name}:", predicted_review[0])




#take the higesht accuracy model for both new and review


