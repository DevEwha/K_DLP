# code/test_sparse_model.py
"""
희소 포맷으로 저장된 모델을 불러와 텍스트 생성 테스트
"""
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
import argparse
import time


class SparseModelLoader:
    """희소 포맷 모델을 dense로 변환하여 추론 가능하게 만드는 클래스"""
    
    def __init__(self, sparse_model_path, device='cuda'):
        self.sparse_model_path = Path(sparse_model_path)
        self.device = device
        
    def load_and_convert(self):
        """희소 모델을 로드하고 dense 포맷으로 변환"""
        print(f"Loading sparse model from: {self.sparse_model_path}")
        
        # 1. Config 로드
        config = AutoConfig.from_pretrained(str(self.sparse_model_path))
        
        # 2. 빈 모델 생성
        print("Creating model architecture...")
        model = LlamaForCausalLM(config)
        
        # 3. 희소 state dict 로드
        sparse_model_file = self.sparse_model_path / 'sparse_csr_model.pt'
        if not sparse_model_file.exists():
            # coo나 csc일 수도 있음
            for fmt in ['coo', 'csc']:
                alt_file = self.sparse_model_path / f'sparse_{fmt}_model.pt'
                if alt_file.exists():
                    sparse_model_file = alt_file
                    break
        
        print(f"Loading sparse weights from: {sparse_model_file}")
        sparse_state_dict = torch.load(sparse_model_file, map_location='cpu')
        
        # 4. 희소 -> Dense 변환
        print("Converting sparse weights to dense...")
        dense_state_dict = {}
        converted_count = 0
        
        for name, param in sparse_state_dict.items():
            if hasattr(param, 'to_dense'):
                # 희소 텐서를 dense로 변환
                dense_state_dict[name] = param.to_dense()
                converted_count += 1
                if converted_count <= 3:
                    print(f"  Converted: {name}")
            else:
                # 이미 dense인 경우 (bias 등)
                dense_state_dict[name] = param
        
        print(f"Converted {converted_count} sparse tensors to dense")
        
        # 5. 모델에 가중치 로드
        model.load_state_dict(dense_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded to {self.device}")
        return model


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """텍스트 생성"""
    # 입력 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\nPrompt: {prompt}")
    print("-" * 60)
    
    # 생성 시작 시간
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 생성 시간
    generation_time = time.time() - start_time
    
    # 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 제거하고 생성된 부분만 추출
    response = generated_text[len(prompt):].strip()
    
    print(f"Response: {response}")
    print("-" * 60)
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens generated: {len(outputs[0]) - len(inputs.input_ids[0])}")
    print(f"Tokens/sec: {(len(outputs[0]) - len(inputs.input_ids[0])) / generation_time:.2f}")
    
    return response


def interactive_mode(model, tokenizer):
    """대화형 모드"""
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not prompt:
                continue
            
            generate_text(model, tokenizer, prompt)
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def test_examples(model, tokenizer):
    """미리 정의된 예시로 테스트"""
    examples = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
        "Tell me a short joke:",
    ]
    
    print("\n" + "="*60)
    print("Testing with example prompts")
    print("="*60)
    
    for prompt in examples:
        generate_text(model, tokenizer, prompt, max_new_tokens=50)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test sparse model inference')
    parser.add_argument('--sparse_model_path', type=str,
                       default='outputs/llama2_7b_wanda_dlp_0.7_sparse_csr',
                       help='Path to sparse model directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Custom prompt for single generation')
    
    args = parser.parse_args()
    
    # 절대 경로로 변환
    script_dir = Path(__file__).parent.parent
    sparse_model_path = script_dir / args.sparse_model_path
    
    # CUDA 사용 가능 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("Sparse Model Inference Test")
    print("="*60)
    
    # 1. 모델 로드
    loader = SparseModelLoader(sparse_model_path, device=args.device)
    model = loader.load_and_convert()
    
    # 2. 토크나이저 로드
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(sparse_model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n" + "="*60)
    print("Model ready for inference!")
    print("="*60)
    
    # 3. 추론 모드 선택
    if args.interactive:
        # 대화형 모드
        interactive_mode(model, tokenizer)
    elif args.prompt:
        # 단일 프롬프트
        generate_text(model, tokenizer, args.prompt, 
                     max_new_tokens=args.max_tokens,
                     temperature=args.temperature)
    else:
        # 예시 테스트
        test_examples(model, tokenizer)
