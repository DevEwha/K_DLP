# code/debug_sparse_save.py
"""
Segmentation fault 원인 찾기
"""
import torch
import sys
import traceback
from pathlib import Path

def test_sparse_save():
    """단계별로 테스트하며 어디서 문제가 생기는지 확인"""
    
    print("="*60)
    print("Debugging Sparse Save Issue")
    print("="*60)
    
    # 1. 간단한 텐서로 테스트
    print("\n[Test 1] Small dense tensor -> sparse -> save")
    try:
        small = torch.randn(100, 100)
        small[small.abs() < 1.0] = 0  # 70% 정도 0
        print(f"  Sparsity: {(small == 0).sum().item() / small.numel() * 100:.1f}%")
        
        sparse_small = small.to_sparse_csr()
        print(f"  ✓ Converted to CSR")
        
        torch.save(sparse_small, 'test_small.pt')
        print(f"  ✓ Saved successfully")
        
        loaded = torch.load('test_small.pt')
        print(f"  ✓ Loaded successfully")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
    
    # 2. 중간 크기
    print("\n[Test 2] Medium tensor (1000x1000)")
    try:
        medium = torch.randn(1000, 1000)
        medium[medium.abs() < 1.0] = 0
        sparse_medium = medium.to_sparse_csr()
        print(f"  ✓ Converted to CSR")
        
        torch.save(sparse_medium, 'test_medium.pt')
        print(f"  ✓ Saved successfully")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
    
    # 3. 큰 텐서 (실제 모델 크기)
    print("\n[Test 3] Large tensor (4096x4096, like LLaMA)")
    try:
        large = torch.randn(4096, 4096, dtype=torch.float16)
        large[large.abs() < 0.5] = 0
        print(f"  Sparsity: {(large == 0).sum().item() / large.numel() * 100:.1f}%")
        
        sparse_large = large.to_sparse_csr()
        print(f"  ✓ Converted to CSR")
        print(f"  Values: {sparse_large.values().numel():,}")
        print(f"  Indices: {sparse_large.col_indices().numel():,}")
        
        # 저장 시도
        print(f"  Attempting save...")
        torch.save(sparse_large, 'test_large.pt', pickle_protocol=4)
        print(f"  ✓ Saved successfully")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
    
    # 4. Dictionary 저장 (실제 케이스)
    print("\n[Test 4] Dictionary of sparse tensors")
    try:
        state_dict = {}
        for i in range(5):
            t = torch.randn(4096, 4096, dtype=torch.float16)
            t[t.abs() < 0.5] = 0
            state_dict[f'layer_{i}'] = t.to_sparse_csr()
        
        print(f"  Created dict with {len(state_dict)} tensors")
        
        print(f"  Attempting save...")
        torch.save(state_dict, 'test_dict.pt', pickle_protocol=4)
        print(f"  ✓ Saved successfully")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
    
    # 5. 더 많은 텐서
    print("\n[Test 5] Large dictionary (50 tensors)")
    try:
        state_dict = {}
        for i in range(50):
            t = torch.randn(4096, 4096, dtype=torch.float16)
            t[t.abs() < 0.5] = 0
            state_dict[f'layer_{i}'] = t.to_sparse_csr()
            
            if (i+1) % 10 == 0:
                print(f"    Created {i+1} tensors...")
        
        print(f"  Attempting save...")
        torch.save(state_dict, 'test_large_dict.pt', pickle_protocol=4)
        print(f"  ✓ Saved successfully")
        
    except Exception as e:
        print(f"  ✗ Failed at {i+1} tensors: {e}")
        traceback.print_exc()
    
    # 6. 실제 모델 로드 테스트
    print("\n[Test 6] Real model shard")
    try:
        script_dir = Path(__file__).parent.parent
        model_file = script_dir / "outputs/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15/pytorch_model-00001-of-00002.bin"
        
        if model_file.exists():
            print(f"  Loading real shard...")
            shard = torch.load(str(model_file), map_location='cpu')
            print(f"  ✓ Loaded {len(shard)} tensors")
            
            # 첫 몇 개만 변환
            sparse_dict = {}
            for i, (name, param) in enumerate(list(shard.items())[:3]):
                if param.ndim == 2:
                    print(f"    Converting {name} {param.shape}...")
                    sparse_dict[name] = param.half().to_sparse_csr()
            
            print(f"  Attempting to save {len(sparse_dict)} real sparse tensors...")
            torch.save(sparse_dict, 'test_real.pt', pickle_protocol=4)
            print(f"  ✓ Saved successfully")
            
        else:
            print(f"  ⚠ Model file not found")
            
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Debug Complete")
    print("="*60)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    test_sparse_save()
