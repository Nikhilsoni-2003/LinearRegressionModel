import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Load datasets
@st.cache
def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, test_data

# Define preprocessing pipeline
def build_preprocessing_pipeline(X):
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    # Define transformers
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combine transformers into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ]
    )
    return preprocessor

# Load data
train_data, test_data = load_data()

# Separate features and target from training data
X_train = train_data.drop(['SalePrice'], axis=1)
y_train = train_data['SalePrice']

# Build preprocessing pipeline and fit on training data
preprocessor = build_preprocessing_pipeline(X_train)
X_train_processed = preprocessor.fit_transform(X_train)

# Train the model
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
model.fit(X_train, y_train)

# Predict on test data
test_features = test_data.copy()
test_predictions = model.predict(test_features)

# Streamlit App
st.title("House Price Prediction App")

# Display test predictions
if st.sidebar.button("Show Test Predictions"):
    test_data['PredictedPrice'] = test_predictions
    st.subheader("Predictions on Test File")
    st.dataframe(test_data[['Id', 'PredictedPrice']])

# User input for custom predictions
st.sidebar.header("Input Custom Data")
def user_input_features():
    inputs = {}
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            inputs[col] = st.sidebar.selectbox(f"{col}:", X_train[col].unique())
        else:
            inputs[col] = st.sidebar.number_input(f"{col}:", value=float(X_train[col].mean()))
    return pd.DataFrame([inputs])

# Get user input
input_data = user_input_features()

# Predict for user input
if not input_data.empty and st.sidebar.button("Predict"):
    predicted_price = model.predict(input_data)[0]
    st.subheader(f"Predicted House Price: ${predicted_price:,.2f}")
