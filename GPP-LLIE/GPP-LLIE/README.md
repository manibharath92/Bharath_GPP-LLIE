# 1. Build CUDA extensions (DCN)
```bash
cd defor_cuda_ext
BASICSR_EXT=True /kaggle/working/py38/bin/python3.8 setup.py develop
```
# 3. Move compiled extensions to main path
```bash
mkdir -p ../ops/dcn/
cp -r basicsr/ops/dcn/* ../ops/dcn/
cd ..
```