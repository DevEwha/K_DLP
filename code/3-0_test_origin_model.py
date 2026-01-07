# test_origin_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading pruned model (no compression)...")

model_path = "/acpl-ssd32/meta-llama/Llama-2-7b-chat-hf-safetensors"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("✓ Model loaded!\n")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"Device: {next(model.parameters()).device}\n")

# 추론
prompt = "The capital of France is"
print(f"Prompt: {prompt}")
print("Generating...\n")

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
