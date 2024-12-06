import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load and preprocess the data
@st.cache_data
def load_and_preprocess_data():
    # Load the data
    data = pd.read_csv("LoanApprovalPrediction.csv")
    data.drop(['Loan_ID'], axis=1, inplace=True)  # Drop Loan_ID column

    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Handle missing values for numeric data
    numeric_imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # Handle missing values for categorical data
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    # Convert categorical columns to numerical values
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    X = data.drop(['Loan_Status'], axis=1)
    y = data['Loan_Status']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    return model, label_encoders, accuracy

model, label_encoders, accuracy = load_and_preprocess_data()

st.title('Loan Approval Prediction')
st.write(f"Model Accuracy: {accuracy:.2f}%")

# Define the input fields for the user
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=0)
credit_history = st.selectbox('Credit History', ['1', '0'])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Convert inputs to a DataFrame
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Check if input data contains unseen labels
def check_unseen_labels(input_df, label_encoders):
    for col in input_df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            if not set(input_df[col].unique()).issubset(set(le.classes_)):
                return True
    return False

# Notify the user if there are unseen labels
if check_unseen_labels(input_data, label_encoders):
    st.error("The input data contains values that are not recognized by the model. Please ensure all inputs are valid.")
else:
    # Apply the same label encoding to user inputs
    for col in input_data.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_data[col] = le.transform(input_data[col])
    
    # Predict using the model
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Denied')
