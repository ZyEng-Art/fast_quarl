"""Check CPU/DGL parity, XPU parity, gradients and an optimizer update."""
import copy
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgl
import torch
from model.qgnn import QGNN
from runtime import configure_runtime


def make_graph():
    # Parallel edges, a self-loop, an isolated node and multiple circuits.
    g = dgl.graph(([0, 0, 1, 2, 2], [1, 1, 2, 2, 3]), num_nodes=5)
    g.ndata['gate_type'] = torch.tensor([0, 1, 2, 3, 1])
    for key, values in {
        'src_idx': [0, 1, 0, 0, 1],
        'dst_idx': [1, 0, 1, 0, 0],
        'reversed': [0, 0, 1, 0, 1],
    }.items():
        g.edata[key] = torch.tensor(values)
    empty = dgl.graph(([], []), num_nodes=2)
    empty.ndata['gate_type'] = torch.tensor([1, 0])
    for key in ('src_idx', 'dst_idx', 'reversed'):
        empty.edata[key] = torch.empty(0, dtype=torch.long)
    return dgl.batch([g, empty])


def compare(device):
    torch.manual_seed(7)
    reference = QGNN(6, 4, 8, 16, 16)
    target = copy.deepcopy(reference).to(device)
    graph = make_graph()
    os.environ['QUARL_GRAPH_BACKEND'] = 'dgl'
    expected = reference(graph)
    expected.square().mean().backward()
    os.environ['QUARL_GRAPH_BACKEND'] = 'torch'
    actual = target(graph)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
    actual.square().mean().backward()
    for (name, p), (_, q) in zip(reference.named_parameters(), target.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(q.grad.cpu(), p.grad, rtol=5e-4, atol=5e-5)
    optimizer = torch.optim.Adam(target.parameters(), lr=3e-4)
    before = [p.detach().clone() for p in target.parameters()]
    optimizer.step()
    assert any(not torch.equal(a, p) for a, p in zip(before, target.parameters()))
    assert all(torch.isfinite(p).all() for p in target.parameters())
    return {'device': str(device), 'forward_max_error': float((actual.detach().cpu()-expected.detach()).abs().max()),
            'gradient_parity': True, 'optimizer_updated': True}


def main():
    configure_runtime()
    results = [compare(torch.device('cpu'))]
    if '--cpu-only' not in sys.argv:
        if not torch.cuda.is_available():
            raise RuntimeError('XPU/CUDA facade unavailable')
        torch.cuda.set_device(0)
        results.append(compare(torch.device('cuda:0')))
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    print(json.dumps({'checks': results, 'passed': True}), flush=True)


if __name__ == '__main__':
    main()
