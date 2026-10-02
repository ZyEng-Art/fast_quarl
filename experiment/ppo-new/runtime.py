"""Device handling for CUDA, CPU and the torch_xmlir CUDA facade on XPU."""
import os
import sys

import torch


def using_xmlir():
    return any(n == 'torch_xmlir' or n.startswith('torch_xmlir.') for n in sys.modules)


def tensor_graph_backend():
    backend = os.environ.get('QUARL_GRAPH_BACKEND', 'auto')
    if backend not in ('auto', 'torch', 'dgl'):
        raise ValueError('QUARL_GRAPH_BACKEND must be auto, torch or dgl')
    return backend == 'torch' or (backend == 'auto' and using_xmlir())


def graph_to_device(graph, device):
    # DGL's CUDA graph runtime cannot run on XMLIR's CUDA facade. Keep topology
    # and graph features on CPU; QGNN transfers tensors to its parameter device.
    if tensor_graph_backend():
        return graph.to('cpu')
    return graph.to(device)


def configure_runtime():
    if using_xmlir():
        # XMLIR in this image can abort during shutdown after autograd worker
        # threads have used the device. Running backward on the caller thread
        # avoids the teardown failure without disabling the runtime exit hook.
        torch.autograd.set_multithreading_enabled(False)


def select_device(gpus, rank=0):
    configure_runtime()
    if not gpus:
        return torch.device('cpu')
    device = torch.device(f'cuda:{gpus[rank]}')
    torch.cuda.set_device(device)
    return device


def distributed_backend(gpus):
    return os.environ.get('QUARL_DIST_BACKEND', 'nccl' if gpus else 'gloo')
