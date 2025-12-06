import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

#Downloding the model from hugging face
model_path = hf_hub_download(repo_id="Shalyn/tourism-project",filename="best_tourism_pred_model.joblib")

#loading the model
model = joblib.load(model_path)

#Streamlit UI for tourism package predition
st.title("Tourism Package Purchase Prediction App")
st.write("Tourism Package Purchase Prediction App is an app to predict if the customer will purchase the package offered by the Visit with us company.")
st.write("Please enter the customer details to check if the customer is likely to purchase the package.")

#get user input
Age = st.number_input("Enter the Age of the customer.",value=25)
CityTier = st.number_input(" Enter the tier of The city category based on development, population, and living standards (1,2,3)",value=1)
DurationOfPitch = st.number_input("Enter the Duration of the sales pitch delivered to the customer.", value=10)
NumberOfPersonVisiting = st.number_input("Enter Total number of people accompanying the customer on the trip.", value =2)
NumberOfFollowups = st.number_input("Enter Total number of follow-ups by the salesperson after the sales pitch.", value =1)
PreferredPropertyStar = st.number_input("Enter Preferred hotel rating by the customer.",value=3)
NumberOfTrips = st.number_input("Enter Average number of trips the customer takes annually.", value=1)
Passport = st.number_input("Enter Whether the customer holds a valid passport (0: No, 1: Yes).",value=0)
PitchSatisfactionScore = st.number_input("Score indicating the customer's satisfaction with the sales pitch.(1-5)",value=3)
OwnCar=st.number_input("Enter Whether the customer owns a car (0: No, 1: Yes).",value=0)
NumberOfChildrenVisiting = st.number_input("Enter Number of children below age 5 accompanying the customer.", value=0)
MonthlyIncome = st.number_input("Enter Gross monthly income of the customer.",value=10000)
TypeofContact= st.selectbox("Enter The method by which the customer was contacted.",["Company Invited","Self Enquiry"])
Occupation = st.selectbox("Enter Customer's occupation", ["Free Lancer","Large Business","Salaried","Small Business"])
Gender= st.selectbox("Enter Gender of the customer (Male, Female).",["Male","Female"])
ProductPitched = st.selectbox("Enter The type of product pitched to the customer.",["Basic","Delux","King","Standard","Super Delux"])
MaritalStatus = st.selectbox("Enter Marital status of the customer.",["Single","Married","Divorced"])
Designation =st.selectbox("Enter Customer's designation in their current organization.",["AVP","Manager","Executive","Senior Manager","VP"])

input_data = pd.DataFrame([{
    'Age':Age,
    'CityTier':CityTier,
    'DurationOfPitch':DurationOfPitch,
    'NumberOfPersonVisiting':NumberOfPersonVisiting,
    'NumberOfFollowups':NumberOfFollowups,
    'PreferredPropertyStar':PreferredPropertyStar,
    'NumberOfTrips':NumberOfTrips,
    'Passport':Passport,
    'PitchSatisfactionScore':PitchSatisfactionScore,
    'OwnCar':OwnCar,
    'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
    'MonthlyIncome':MonthlyIncome,
    'TypeofContact':TypeofContact,
    'Occupation':Occupation,
    'Gender':Gender,
    'ProductPitched':ProductPitched,
    'MaritalStatus':MaritalStatus,
    'Designation':Designation
}])

# Set the classification threshold
classification_threshold = 0.45
# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the package" if prediction == 1 else "not purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
