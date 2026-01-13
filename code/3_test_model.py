# test_model.py
# 
# 실행 예시:
# python 3_test_model.py --model Llama-2-7b-chat-hf-safetensors
# python 3_test_model.py --model Llama-2-7b-chat-hf-safetensors --base_dir meta-llama
# python 3_test_model.py --model llama2_7b_chat_wanda_dlp_0.5_unstructured_alpha0.04 --base_dir k_models
# python 3_test_model.py --model /acpl-ssd32/meta-llama/Llama-2-7b-chat-hf-safetensors  # 전체 경로도 가능
# python 3_test_model.py --model Llama-2-7b-chat-hf-safetensors --dataset ptb  # 다른 데이터셋
# python 3_test_model.py --model Llama-2-7b-chat-hf-safetensors --eval_zero_shot  # zero-shot 평가 포함

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity


def main():
    parser = argparse.ArgumentParser(description='Test LLM model with PPL evaluation')
    parser.add_argument('--model', type=str, required=True, 
                        help='Model name (e.g., Llama-2-7b-chat-hf-safetensors) or full path')
    parser.add_argument('--base_dir', type=str, default='meta-llama', 
                        choices=['meta-llama', 'k_models'],
                        help='Base directory: meta-llama or k_models (default: meta-llama)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'],
                        help='Dataset for PPL evaluation (default: wikitext2)')
    parser.add_argument('--eval_zero_shot', action='store_true',
                        help='Run zero-shot evaluation')
    parser.add_argument('--prompt', type=str, default='What is the capital of France?',
                        help='Prompt for generation test')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation (default: 0.7)')
    parser.add_argument('--do_sample', action='store_true', default=True,
                        help='Use sampling for generation')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Maximum new tokens to generate (default: 50)')
    
    args = parser.parse_args()
    
    # 모델 경로 설정
    if args.model.startswith('/'):
        # 전체 경로가 주어진 경우
        model_path = args.model
    else:
        # 모델 이름만 주어진 경우, base_dir과 결합
        model_path = f"/acpl-ssd32/{args.base_dir}/{args.model}"
    
    print(f"Model path: {model_path}")
    print("Loading model...")
    
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # seqlen 설정
    model.seqlen = 2048
    
    # device 설정
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
    ppl_test = eval_ppl(model, tokenizer, device, dataset=args.dataset)
    print(f"ppl on {args.dataset} {ppl_test}")
    
    # Zero-shot evaluation
    if args.eval_zero_shot:
        print("\n" + "="*50)
        print("Running zero-shot evaluation...")
        print("="*50)
        accelerate = True
        task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "openbookqa", 'winogrande', 'arc_easy']
        num_shot = 0
        
        results = eval_zero_shot(model_path, task_list, num_shot, accelerate)
        print("Zero-shot results:", results)
    
    # 추론 테스트
    print("\n" + "="*50)
    print("Testing generation...")
    print("="*50)
    
    print(f"Prompt: {args.prompt}")
    print("Generating...\n")
    
    inputs = tokenizer(args.prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Result: {generated_text}")


if __name__ == '__main__':
    main()
