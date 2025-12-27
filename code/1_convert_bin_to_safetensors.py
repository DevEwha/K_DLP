from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "/acpl-ssd20/k_models/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15"
output_dir = "/acpl-ssd20/k_models/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15_safetensors"

# 1. shard된 bin 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,   # LLaMA면 거의 확실히 fp16
)

# 2. safetensors로 저장
model.save_pretrained(
    output_dir,
    safe_serialization=True
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.save_pretrained(output_dir)

print("✅ safetensors 변환 완료")
