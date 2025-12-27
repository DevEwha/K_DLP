# compress_pruned_model.py (ZipNN을 사용하여 프루닝된 Llama 2 7B 모델 압축)
import zipnn
import os
from pathlib import Path
import time


# 프루닝된 모델 경로 (절대 경로로 변경)
model_dir = "/acpl-ssd20/k_models/llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15_safetensors"
output_dir = "/acpl-ssd20/k_models/zipnn_llama2_7b_wanda_dlp_0.7_unstructured_alpha0.15"


# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)


# ZipNN 객체 생성 (auto 모드로 최적 압축 방법 자동 선택)
zpn = zipnn.ZipNN(method='auto')


print("=" * 60)
print("ZipNN Compression for Pruned Llama 2 7B Model")
print("=" * 60)


# .safetensors 파일들 압축 (확장자 변경)
safetensor_files = list(Path(model_dir).glob("*.safetensors"))
print(f"\nFound {len(safetensor_files)} safetensors files")


total_original_size = 0
total_compressed_size = 0


for i, file_path in enumerate(safetensor_files, 1):
    print(f"\n[{i}/{len(safetensor_files)}] Compressing {file_path.name}...")
    start_time = time.time()
    
    # 파일 읽기
    with open(file_path, 'rb') as f:
        original_data = f.read()
    
    # 압축
    compressed_data = zpn.compress(original_data)
    
    # 압축 파일 저장
    compressed_filename = file_path.name + ".znn"
    compressed_path = os.path.join(output_dir, compressed_filename)
    with open(compressed_path, 'wb') as f:
        f.write(compressed_data)
    
    # 통계 계산
    original_size = len(original_data) / (1024**3)  # GB
    compressed_size = len(compressed_data) / (1024**3)  # GB
    ratio = (1 - compressed_size/original_size) * 100
    elapsed = time.time() - start_time
    
    total_original_size += original_size
    total_compressed_size += compressed_size
    
    print(f"  ✓ Original size: {original_size:.2f} GB")
    print(f"  ✓ Compressed size: {compressed_size:.2f} GB")
    print(f"  ✓ Reduction: {ratio:.2f}%")
    print(f"  ✓ Time: {elapsed:.1f}s")


# 설정 파일들 복사
print("\n" + "=" * 60)
print("Copying configuration files...")
config_files = ['config.json', 'generation_config.json', 'model.safetensors.index.json',
                'special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.model']


for config_file in config_files:
    src = os.path.join(model_dir, config_file)
    dst = os.path.join(output_dir, config_file)
    if os.path.exists(src):
        with open(src, 'rb') as f_src:
            with open(dst, 'wb') as f_dst:
                f_dst.write(f_src.read())
        print(f"  ✓ Copied {config_file}")


# 최종 통계
print("\n" + "=" * 60)
print("COMPRESSION SUMMARY")
print("=" * 60)
print(f"Total original size: {total_original_size:.2f} GB")
print(f"Total compressed size: {total_compressed_size:.2f} GB")
print(f"Overall reduction: {(1 - total_compressed_size/total_original_size) * 100:.2f}%")
print(f"Space saved: {total_original_size - total_compressed_size:.2f} GB")
print(f"\nCompressed model saved to: {output_dir}")
print("=" * 60)
