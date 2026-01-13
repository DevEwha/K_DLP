# test_origin_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity

print("Loading model...")

# model_path = "/acpl-ssd32/meta-llama/Llama-2-7b-chat-hf-safetensors"
model_path = "/acpl-ssd32/k_models/llama2_7b_chat_wanda_dlp_0.5_unstructured_alpha0.04"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# seqlen 설정 (원본 코드와 동일)
model.seqlen = 2048

# device 설정 (원본 코드와 동일)
device = torch.device("cuda:0")
if "30b" in model_path or "65b" in model_path:
    device = model.hf_device_map["lm_head"]
print("use device ", device)

print("✓ Model loaded!\n")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

model.eval()

# Sparsity 체크
print("*"*30)
sparsity_ratio = check_sparsity(model)
print(f"sparsity sanity check {sparsity_ratio:.4f}")
print("*"*30)

# PPL 측정
dataset = "wikitext2"  # 또는 "ptb", "c4"
ppl_test = eval_ppl(model, tokenizer, device, dataset=dataset)
print(f"ppl on {dataset} {ppl_test}")

# 여러 데이터셋에서 측정하려면:
# for dataset in ["wikitext2", "ptb", "c4"]:
#     ppl_test = eval_ppl(model, tokenizer, device, dataset=dataset)
#     print(f"ppl on {dataset} {ppl_test}")

# Zero-shot evaluation (선택사항)
eval_zero_shot_flag = False  # True로 설정하면 zero-shot 평가 실행
if eval_zero_shot_flag:
    accelerate = True
    task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "openbookqa", 'winogrande', 'arc_easy']
    num_shot = 0
    
    results = eval_zero_shot(model_path, task_list, num_shot, accelerate)
    print("Zero-shot results:", results)

# 추론 테스트 (선택사항)
print("\n" + "="*50)
print("Testing generation...")
print("="*50)

prompt = "The capital of France is"
print(f"Prompt: {prompt}")
print("Generating...\n")

inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)

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
