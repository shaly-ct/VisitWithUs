from huggingface_hub import HfApi
import os

api = HfApi(token="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")
api.upload_folder(
    folder_path="tourism_project/deployment",     # Corrected: the local folder containing your files, relative to repo root
    repo_id="Shalyn/tourism-project",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
