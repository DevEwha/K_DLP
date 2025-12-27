# code/convert_to_sparse.py
"""
프루닝된 모델을 희소 포맷(CSR)으로 변환하여 저장하는 스크립트
"""
import torch
import os
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def convert_model_to_sparse(model_path, output_path, format='csr'):
    """
    프루닝된 모델을 희소 포맷으로 변환
    
    Args:
        model_path: 프루닝된 모델 경로 (예: outputs/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15/)
        output_path: 희소 모델 저장 경로
        format: 'csr', 'coo', 'csc' 중 선택
    """
    print(f"Loading pruned model from: {model_path}")
    
    # 1. 프루닝된 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 2. 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 3. 희소 포맷 변환 통계
    total_params = 0
    zero_params = 0
    converted_layers = 0
    
    # 4. 각 레이어의 가중치를 희소 포맷으로 변환
    sparse_state_dict = {}
    
    print(f"\nConverting weights to sparse format ({format})...")
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
        
        # 2D 텐서(Linear layer weights)만 희소 포맷으로 변환
        if param.ndim == 2 and param.requires_grad:
            if format == 'csr':
                sparse_param = param.data.to_sparse_csr()
            elif format == 'coo':
                sparse_param = param.data.to_sparse_coo()
            elif format == 'csc':
                sparse_param = param.data.to_sparse_csc()
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            sparse_state_dict[name] = sparse_param
            converted_layers += 1
            
            # 압축률 계산
            original_size = param.numel() * param.element_size()
            if format == 'csr':
                sparse_size = (sparse_param.values().numel() * sparse_param.values().element_size() + 
                              sparse_param.crow_indices().numel() * sparse_param.crow_indices().element_size() +
                              sparse_param.col_indices().numel() * sparse_param.col_indices().element_size())
            else:
                sparse_size = original_size  # 간단한 추정
            
            compression_ratio = original_size / sparse_size if sparse_size > 0 else 1.0
            
            if converted_layers <= 5:  # 처음 5개만 출력
                print(f"  {name}: {param.shape} -> compression {compression_ratio:.2f}x")
        else:
            # 1D 텐서(bias 등)는 그대로 저장
            sparse_state_dict[name] = param.data
    
    # 5. 희소 state dict 저장
    sparse_model_path = os.path.join(output_path, f'sparse_{format}_model.pt')
    print(f"\nSaving sparse model to: {sparse_model_path}")
    torch.save(sparse_state_dict, sparse_model_path)
    
    # 6. config 및 토크나이저 파일 복사
    print("Copying config and tokenizer files...")
    for file in ['config.json', 'generation_config.json', 'special_tokens_map.json', 
                 'tokenizer_config.json', 'tokenizer.model']:
        src = os.path.join(model_path, file)
        dst = os.path.join(output_path, file)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
    
    # 7. 메타데이터 저장
    sparsity = (zero_params / total_params) * 100
    metadata = {
        'original_model_path': model_path,
        'sparse_format': format,
        'total_parameters': total_params,
        'zero_parameters': zero_params,
        'sparsity_ratio': f"{sparsity:.2f}%",
        'converted_layers': converted_layers,
        'compression_info': 'Weights with shape [out, in] converted to sparse format'
    }
    
    with open(os.path.join(output_path, 'sparse_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 8. 결과 요약
    print("\n" + "="*60)
    print("Conversion Summary:")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Zero parameters: {zero_params:,}")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Converted layers: {converted_layers}")
    print(f"Sparse format: {format.upper()}")
    print(f"Output directory: {output_path}")
    print("="*60)
    
    return sparse_state_dict, metadata


def load_sparse_model(sparse_model_path, format='csr'):
    """
    희소 포맷으로 저장된 모델 로드 (검증용)
    
    Args:
        sparse_model_path: 희소 모델 .pt 파일 경로
        format: 사용된 희소 포맷
    """
    print(f"Loading sparse model from: {sparse_model_path}")
    sparse_state_dict = torch.load(sparse_model_path)
    
    # Dense로 복원하여 검증
    dense_state_dict = {}
    for name, param in sparse_state_dict.items():
        if hasattr(param, 'to_dense'):
            dense_state_dict[name] = param.to_dense()
        else:
            dense_state_dict[name] = param
    
    print(f"Successfully loaded {len(sparse_state_dict)} parameters")
    return sparse_state_dict, dense_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert pruned model to sparse format')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15',
                       help='Path to pruned model')
    parser.add_argument('--output_path', type=str,
                       default='outputs/llama2_7b_wanda_dlp_0.7_sparse_csr',
                       help='Path to save sparse model')
    parser.add_argument('--format', type=str, default='csr',
                       choices=['csr', 'coo', 'csc'],
                       help='Sparse format to use')
    parser.add_argument('--verify', action='store_true',
                       help='Verify conversion by loading and comparing')
    
    args = parser.parse_args()
    
    # 절대 경로로 변환
    script_dir = Path(__file__).parent.parent  # lib/ -> K_DLP/
    model_path = script_dir / args.model_path
    output_path = script_dir / args.output_path
    
    # 변환 실행
    sparse_state_dict, metadata = convert_model_to_sparse(
        str(model_path),
        str(output_path),
        format=args.format
    )
    
    # 검증 (옵션)
    if args.verify:
        print("\n" + "="*60)
        print("Verification:")
        print("="*60)
        sparse_model_file = output_path / f'sparse_{args.format}_model.pt'
        loaded_sparse, loaded_dense = load_sparse_model(str(sparse_model_file), args.format)
        print("✓ Sparse model can be loaded and converted back to dense")
