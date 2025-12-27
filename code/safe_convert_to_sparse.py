# code/convert_sparse_chunked.py
"""
청크 단위로 저장하는 안전한 희소 변환
"""
import torch
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def save_in_chunks(state_dict, output_path, chunk_size=50):
    """state_dict를 여러 파일로 나눠 저장"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    keys = list(state_dict.keys())
    num_chunks = (len(keys) + chunk_size - 1) // chunk_size
    
    index = {}
    
    for i in range(num_chunks):
        chunk_keys = keys[i * chunk_size : (i+1) * chunk_size]
        chunk_dict = {k: state_dict[k] for k in chunk_keys}
        
        chunk_file = f"sparse_chunk_{i:04d}.pt"
        chunk_path = output_path / chunk_file
        
        print(f"  Saving chunk {i+1}/{num_chunks} ({len(chunk_keys)} tensors)...")
        torch.save(chunk_dict, str(chunk_path), pickle_protocol=4)
        
        # 인덱스 업데이트
        for key in chunk_keys:
            index[key] = chunk_file
    
    # 인덱스 저장
    with open(output_path / 'sparse_index.json', 'w') as f:
        json.dump(index, f, indent=2)
    
    return num_chunks

def convert_to_sparse_chunked():
    """청크 단위 저장 방식"""
    
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / "outputs/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15"
    output_path = script_dir / "outputs/llama2_7b_wanda_dlp_0.7_sparse_csr"
    
    print("="*60)
    print("Chunked Sparse Conversion")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    if not input_path.exists():
        print(f"\n❌ Error: Input path not found")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Config 파일 복사
    print("\n[1/4] Copying config files...")
    config_files = [
        'config.json', 'generation_config.json', 
        'special_tokens_map.json', 'tokenizer_config.json', 
        'tokenizer.model', 'pytorch_model.bin.index.json'
    ]
    
    for file in config_files:
        src = input_path / file
        if src.exists():
            shutil.copy2(src, output_path / file)
            print(f"  ✓ {file}")
    
    # 2. 모델 로드 및 변환
    print("\n[2/4] Processing shards...")
    
    sparse_dict = {}
    total_params = 0
    zero_params = 0
    converted = 0
    
    for shard_num in [1, 2]:
        model_file = input_path / f"pytorch_model-0000{shard_num}-of-00002.bin"
        
        if not model_file.exists():
            continue
        
        print(f"\n  Shard {shard_num}:")
        shard = torch.load(str(model_file), map_location='cpu')
        
        for name, param in tqdm(shard.items(), desc=f"  Converting"):
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            
            if param.ndim == 2:
                try:
                    if param.dtype != torch.float16:
                        param = param.half()
                    sparse_dict[name] = param.to_sparse_csr()
                    converted += 1
                except:
                    sparse_dict[name] = param
            else:
                sparse_dict[name] = param
        
        del shard
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not sparse_dict:
        print("\n❌ No tensors converted")
        return
    
    # 3. 청크로 저장
    print("\n[3/4] Saving in chunks...")
    num_chunks = save_in_chunks(sparse_dict, output_path, chunk_size=50)
    print(f"  ✓ Saved {num_chunks} chunks")
    
    # 4. 메타데이터
    print("\n[4/4] Metadata...")
    sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0
    
    metadata = {
        'total_parameters': int(total_params),
        'zero_parameters': int(zero_params),
        'sparsity': f"{sparsity:.2f}%",
        'converted_layers': converted,
        'format': 'CSR_CHUNKED',
        'num_chunks': num_chunks
    }
    
    with open(output_path / 'sparse_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 결과
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"Parameters: {total_params:,}")
    print(f"Zeros: {zero_params:,} ({sparsity:.2f}%)")
    print(f"Converted: {converted} layers")
    print(f"Chunks: {num_chunks}")
    print(f"Output: {output_path}")
    print("="*60)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    convert_to_sparse_chunked()
