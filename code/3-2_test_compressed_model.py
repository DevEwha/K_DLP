# decompress_and_load.py
from zipnn import zipnn_safetensors
import zipnn
import os

print("Step 1: Decompressing model files...")

compressed_dir = "/acpl-ssd20/k_models/zipnn_llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15"
zpn = zipnn.ZipNN()

# .znn 파일들 압축 해제
znn_files = [
    "model-00001-of-00002.safetensors.znn",
    "model-00002-of-00002.safetensors.znn"
]

for znn_file in znn_files:
    znn_path = os.path.join(compressed_dir, znn_file)
    output_path = znn_path.replace(".znn", "")
    
    # 이미 압축 해제된 파일이 있으면 스킵
    if os.path.exists(output_path):
        print(f"  ✓ {os.path.basename(output_path)} already exists, skipping...")
        continue
    
    print(f"  Decompressing {znn_file}...")
    
    with open(znn_path, 'rb') as f:
        compressed_data = f.read()
    
    decompressed_data = zpn.decompress(compressed_data)
    
    with open(output_path, 'wb') as f:
        f.write(decompressed_data)
    
    print(f"  ✓ Saved to {os.path.basename(output_path)}")

print("\nStep 2: Loading model...")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = compressed_dir

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("✓ Model loaded!\n")

# 추론
prompt = "The capital of France is"
print(f"Prompt: {prompt}")
print("Generating...\n")

# return_token_type_ids=False 추가
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Result: {generated_text}")
