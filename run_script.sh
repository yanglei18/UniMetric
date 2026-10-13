# conda activate dvlt
# CUDA_VISIBLE_DEVICES=4,6,7,8,9 python get_metric_scale.py --config configs/argoverse/dvlt_long.yaml 
conda activate QWen
# CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python get_metric_scale.py --config configs/argoverse/stream3r_long.yaml
# CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python get_metric_scale.py --config configs/argoverse/scal3r_long.yaml 
# CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python get_metric_scale.py --config configs/argoverse/must3r_long.yaml 
# CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python get_metric_scale.py --config configs/argoverse/vggt_omega_long.yaml 
# CUDA_VISIBLE_DEVICES=6,7,8,9 python get_metric_scale.py --config configs/argoverse/lingbot_map_long.yaml 
# CUDA_VISIBLE_DEVICES=6,7,8,9 python get_metric_scale.py --config configs/argoverse/vggt_long.yaml
CUDA_VISIBLE_DEVICES=4 python get_metric_scale.py --config configs/argoverse/da3_long.yaml
CUDA_VISIBLE_DEVICES=4 python get_metric_scale.py --config configs/argoverse/pi3_long.yaml
