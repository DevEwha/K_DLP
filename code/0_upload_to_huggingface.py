from huggingface_hub import HfApi

folder_path = "/acpl-ssd32/k_models/zipnn_llama2_7b_chat_wanda_dlp_0.5_unstructured_alpha0.04"
repo_id = f"rinarina0429/{folder_path.split('/')[-1]}"

api = HfApi()

# 1. 먼저 레포지토리 생성 (이미 존재하면 무시)
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True  # 이미 존재해도 에러 발생 안 함
)

# 2. 폴더 전체를 업로드
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model"
)
