# test_single_inference.py
# ZipNN 압축 모델로 단일 문장 추론
from transformers import AutoModelForCausalLM, AutoTokenizer
from zipnn import zipnn_safetensors
import torch

# ZipNN 플러그인 활성화
zipnn_safetensors()

print("Loading compressed model...")
model_path = "/acpl-ssd20/k_models/zipnn_llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("✓ Model loaded!\n")

# 프롬프트
prompt = "The capital of France is"

print(f"Prompt: {prompt}")
print("Generating...\n")

# 추론
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 결과 출력
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Result: {generated_text}")
