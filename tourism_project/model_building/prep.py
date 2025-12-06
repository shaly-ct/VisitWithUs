#import necessary libraries
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi
from dotenv import load_dotenv

#Load env
load_dotenv()

#get variables for hf
api = HfApi(token="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")
dataset_path = "hf://datasets/Shalyn/tourism-project/tourism.csv"
input_dataset = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")

#defining target variable
target = 'ProdTaken'

#listing numerical variables in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
]

#listing categorical variables
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]
#Defining predictor matrix using selected features
X = input_dataset[numeric_features+categorical_features]

#Defining target variable
y = input_dataset[target]

#splitting data for train test

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,y, test_size = 0.2, random_state =42
)

Xtrain.to_csv("Xtrain.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","ytrain.csv","Xtest.csv","ytest.csv"]

repo_id = "Shalyn/tourism-project" # Define repo_id for consistency

for ifile in files:
  api.upload_file(
      path_or_fileobj=ifile,
      path_in_repo=ifile, # Use ifile directly as the path in repo
      repo_id=repo_id, # Use the defined repo_id variable
      repo_type="dataset",
      commit_message=f"Upload {ifile}" # Add a commit message
  )
