conda activate dvlt
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/dvlt_long.yaml 
conda activate QWen
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/4rc_long.yaml 
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/stream3r_long.yaml
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/scal3r_long.yaml 
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/vggt_omega_long.yaml 
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/lingbot_map_long.yaml 
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/vggt_long.yaml
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/dair_v2x_i/da3_long.yaml