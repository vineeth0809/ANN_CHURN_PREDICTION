import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
#testing git fe1-second


model = tf.keras.models.load_model('churn_model.h5')


with open('lable_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)


st.title("Customer Churn Prediction")

geography = st.selectbox("Select Geography:", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender: ", label_encoder_gender.classes_)
age = st.slider("Age: ", 18, 100, 30)
balance = st.number_input("Balance: ")
credit_score = st.slider("Credit Score: ", 300, 850, 600)
estimated_salary = st.number_input("Estimated Salary: ")
tenure = st.slider("Tenure: ", 0, 10, 3)
num_of_products = st.slider("Number of Products: ", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card: ", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member: ", ["Yes", "No"])

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card == "Yes" else 0,
    'IsActiveMember': 1 if is_active_member == "Yes" else 0,
    'EstimatedSalary': estimated_salary
}

input_df = pd.DataFrame([input_data])
print(input_df)
geo_encoder = one_hot_encoder_geo.transform([[geography]])

geo_ecoder_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.reset_index(drop=True), geo_ecoder_df], axis=1)
print(input_df)
scaled_input = scaler.transform(input_df)

probabilities = model.predict(scaled_input)[0][0]
print(probabilities)
if probabilities > 0.5:
    st.write(f"The customer is likely to churn with a probability of {probabilities:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {probabilities:.2f}")

