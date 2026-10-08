echo "set torch expandable segments to False..."
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
  python3 expandable_ipc_repro.py

echo ================

echo "set torch expandable segments to True..."
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  python3 expandable_ipc_repro.py
