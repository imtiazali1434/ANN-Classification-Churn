import pickle
import streamlit as st
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

## Load the model from h5 file save in the folder
model= load_model('model.h5')

## Load the encoder and scaler save in pkl file
with open('label_enco_gen.pkl','rb') as file :
    lab_enco_gen= pickle.load(file)
with open('onehot_enco_geo.pkl','rb') as file:
    lab_enco_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler= pickle.load(file) 

## Streamlit app 
st.title('Cutomer Churn Predictions ')


geography = st.selectbox('Geography', lab_enco_geo.categories_[0])
gender = st.selectbox('Gender', lab_enco_gen.classes_)
age= st.slider('Age',18,92)
balance= st.number_input('Balance')
credit_score= st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure= st.slider('Tenure',0,10)
num_of_products= st.slider('Number of Products',1,4)
has_cr_card= st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

## Prepare the input data

input_data= pd.DataFrame({

    'CreditScore':[credit_score],
    'Gender':[lab_enco_gen.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})
    
## one hot encode Geography 
geo_enco= lab_enco_geo.transform([[geography]]).toarray()
geo_enco_df= pd.DataFrame(geo_enco, columns= lab_enco_geo.get_feature_names_out(['Geography']))

## combine one hot encoded data with imput data 
input_data=pd.concat([input_data.reset_index(drop=True), geo_enco_df],axis=1)

## Scale the input data 
input_data_scaled= scaler.transform(input_data)

## Prediction Churn 
prediction= model.predict(input_data_scaled)
predict_prob= prediction[0][0]


st.write(f' Probabiltiy Value: {predict_prob:.2f}')
if predict_prob>0.05:
    st.write('The customer is likly to churn.')
else:
    st.write('The Customer in not likly to churn.')