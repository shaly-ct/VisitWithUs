#importing necessary libraries
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

#Load env (ensure .env file is present or HF_TOKEN is in Colab Secrets)
load_dotenv()

#setting up repo id and repo type
repo_id = "Shalyn/tourism-project"
repo_type = "dataset"

#Initialise HF api with token from environment variable or Colab Secrets
api = HfApi(token="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")

#connecting to hf space
try:
  api.repo_info(repo_id=repo_id,repo_type=repo_type)
  print(f"Space '{repo_id}' already exist!!")
except (RepositoryNotFoundError, HfHubHTTPError) as e:
  print(f"Space '{repo_id}' is not found or inaccessible. Attempting to create the repo...")
  # Ensure proper authentication for create_repo as well
  create_repo(repo_id=repo_id,repo_type=repo_type,private=False, token="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")
  print(f"Space '{repo_id}' is created")

api.upload_folder(
    folder_path = 'tourism_project/data',
    repo_id = repo_id,
    repo_type = repo_type
)
