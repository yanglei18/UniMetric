# conda activate dvlt
# CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/v2x_radar_i/dvlt_long.yaml 
# conda activate QWen
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/v2x_radar_i/4rc_long.yaml 
CUDA_VISIBLE_DEVICES=4,7,8,9 python get_metric_scale.py --config configs/v2x_radar_i/stream3r_long.yaml
CUDA_VISIBLE_DEVICES=4,7,8,9 python get_metric_scale.py --config configs/v2x_radar_i/scal3r_long.yaml 
CUDA_VISIBLE_DEVICES=4,7,8,9 python get_metric_scale.py --config configs/v2x_radar_i/must3r_long.yaml 
CUDA_VISIBLE_DEVICES=4,7,8,9 python get_metric_scale.py --config configs/v2x_radar_i/vggt_omega_long.yaml 
CUDA_VISIBLE_DEVICES=4,7,8,9 python get_metric_scale.py --config configs/v2x_radar_i/lingbot_map_long.yaml 
CUDA_VISIBLE_DEVICES=4,7,8,9 python get_metric_scale.py --config configs/v2x_radar_i/vggt_long.yaml
CUDA_VISIBLE_DEVICES=4,7,8,9 python get_metric_scale.py --config configs/v2x_radar_i/da3_long.yaml
CUDA_VISIBLE_DEVICES=4,6,8,9 python get_metric_scale.py --config configs/v2x_radar_i/pi3_long.yaml









