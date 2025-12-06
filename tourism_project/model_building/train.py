# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import hf_hub_download
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Shalyn/tourism-project"

# Download files locally
Xtrain_local_path = hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset")
Xtest_local_path = hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type="dataset")
ytrain_local_path = hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset")
ytest_local_path = hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type="dataset")


Xtrain = pd.read_csv(Xtrain_local_path)
Xtest = pd.read_csv(Xtest_local_path)
ytrain = pd.read_csv(ytrain_local_path)
ytest = pd.read_csv(ytest_local_path)


#defining target variable (not used directly for training in this block, but kept for context)
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

#setting up class weight
class_weight = ytrain.value_counts()[0]/ytrain.value_counts()[1]
#Defining preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

#Defining base xgboost model
xgb_model = xgb.XGBClassifier(scale_pos_weight = class_weight, random_state=42)

#setting up parameter grid
param_grid = param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],    # num of trees
    'xgbclassifier__max_depth': [2, 3],    # max depth of tree
    'xgbclassifier__colsample_bytree': [0.4, 0.6],    # % of attributes for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.6],    #  % of attributes for each level in tree
    'xgbclassifier__learning_rate': [0.01, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.6],    # L2 regularization factor
}

#Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
  grid_search = GridSearchCV(model_pipeline,param_grid, cv=5, n_jobs=-1)
  grid_search.fit(Xtrain,ytrain)

#logging params
result = grid_search.cv_results_

for i in range(len(result['params'])):
  param_set = result['params'][i]
  mean_score = result['mean_test_score'][i]
  std_score = result['std_test_score'][i]

  # logging each combinations
  with mlflow.start_run(nested=True):
    mlflow.log_params(param_set)
    mlflow.log_metric("mean_test_score", mean_score)
    mlflow.log_metric("std_test_score", std_score)

#logging best params
mlflow.log_params(grid_search.best_params_)

#storing best model
best_model = grid_search.best_estimator_

#set classification threshold
classification_threshold = 0.45
y_pred_train_prob = best_model.predict_proba(Xtrain)[:,1]
y_pred_train = (y_pred_train_prob >= classification_threshold).astype(int)

y_pred_test_prob =best_model.predict_proba(Xtest)[:,1]
y_pred_test = (y_pred_test_prob >= classification_threshold).astype(int)

train_report = classification_report(ytrain, y_pred_train,output_dict=True)
test_report = classification_report(ytest, y_pred_test, output_dict=True)

#printing classificaiton report
print(classification_report(ytrain, y_pred_train))
print(classification_report(ytest, y_pred_test))

mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

#Saving the best model
model_filename = 'best_tourism_pred_model.joblib'
joblib.dump(best_model, model_filename)

# Log the model artifact
with mlflow.start_run(nested=True):
    mlflow.log_artifact(model_filename, artifact_path="model")
    print(f"Model saved as artifact at: {model_filename}")

#Uploading to hugging face
repo_id = "Shalyn/tourism-project"
repo_type= "model"

api = HfApi(token="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")

#Checking if the repo already exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo
api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo="best_tourism_pred_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
