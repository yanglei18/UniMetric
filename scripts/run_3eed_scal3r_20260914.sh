#!/usr/bin/env bash
# One authorized queue step only; later steps are managed by the thread heartbeat.
set -Eeuo pipefail
cd /home/yanglei/ws_yuner/Metric-Depth
queue_dir=outputs/run_logs/3eed_serial_dvlt_stream3r_scal3r_20260914
mkdir -p "$queue_dir"
mkdir "$queue_dir/scal3r.started" || exit 73
exec >> "$queue_dir/scal3r.log" 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$queue_dir/scal3r.exit_code"; date -Is > "$queue_dir/scal3r.finished_at"' EXIT
date -Is > "$queue_dir/scal3r.started_at"
printf '%s\n' "$$" > "$queue_dir/scal3r.runner_pid"
source /home/yanglei/anaconda3/etc/profile.d/conda.sh
conda activate QWen
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python - <<'PY'
import os, shutil, subprocess
from pathlib import Path
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0]) >= 100*1024**2, 'Less than 100 GiB available RAM'
for path in ['outputs', '/data1/yanglei/Metric-Depth/tmp']:
    assert shutil.disk_usage(path).free >= 150*1024**3, 'Insufficient disk space'
    assert os.statvfs(path).f_favail > 100000, 'Insufficient free inodes'
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: args = (proc/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError): continue
    assert not any(a.endswith(b'/get_metric_scale.py') or a == b'get_metric_scale.py' for a in args), 'Another model is running'
apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
assert not apps, 'GPU compute processes still present'
rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
for row in rows.splitlines():
    index, used, util = map(int, row.split(','))
    if index in [0,1,2]: assert used < 1024 and util < 5, 'Selected GPU is not idle'
print('Preflight passed: QWen environment, GPUs 0,1,2, max 3 workers', flush=True)
PY
python -u get_metric_scale.py --config configs/3eed_dataset/scal3r_long.yaml --parallel --max-gpus 3 &
model_pid=$!
printf '%s\n' "$model_pid" > "$queue_dir/scal3r.pid"
wait "$model_pid"
