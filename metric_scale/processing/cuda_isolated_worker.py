"""Spawn targets that pin one GPU before torch is imported."""
import os


def _pin_gpu(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))


def _torch_visible_count():
    try:
        import torch
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _prepare_worker_device(gpu_id):
    # Spawn children re-import get_metric_scale.py and initialize torch before
    # this target runs. Changing CUDA_VISIBLE_DEVICES after that does not
    # shrink torch.cuda.device_count(), but it DOES affect backend subprocesses.
    # If torch already sees multiple GPUs, keep the parent visibility list and
    # use set_device(gpu_id). Only pin when torch sees a single device.
    visible = _torch_visible_count()
    if visible <= 1:
        _pin_gpu(gpu_id)
        return 0
    return int(gpu_id)


class _RewriteGpuIdQueue:
    def __init__(self, queue, gpu_id):
        self._queue = queue
        self._gpu_id = gpu_id

    def put(self, item):
        if isinstance(item, dict) and "gpu_id" in item:
            item = dict(item)
            item["gpu_id"] = self._gpu_id
        self._queue.put(item)


def run_sam_tasks_on_gpu(gpu_id, task_payloads, result_queue):
    local_id = _prepare_worker_device(gpu_id)
    from metric_scale.processing.processing import _run_sam_tasks_on_gpu

    _run_sam_tasks_on_gpu(local_id, task_payloads, _RewriteGpuIdQueue(result_queue, gpu_id))


def run_depth_tasks_on_gpu(gpu_id, task_payloads, result_queue):
    local_id = _prepare_worker_device(gpu_id)
    from metric_scale.processing.processing import _run_depth_tasks_on_gpu

    _run_depth_tasks_on_gpu(local_id, task_payloads, _RewriteGpuIdQueue(result_queue, gpu_id))


def run_sequence_pipeline_on_gpu(gpu_id, task_payloads, result_queue):
    local_id = _prepare_worker_device(gpu_id)
    from metric_scale.processing.processing import _run_sequence_pipeline_on_gpu

    _run_sequence_pipeline_on_gpu(
        local_id,
        task_payloads,
        _RewriteGpuIdQueue(result_queue, gpu_id),
    )
