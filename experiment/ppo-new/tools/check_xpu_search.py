"""Run bounded policy-guided search from a PPO smoke checkpoint."""
import gc
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'experiment/ppo-new'))
import qtz
import torch
from omegaconf import OmegaConf
from model.actor_critic import ActorCritic
from runtime import select_device
from tester import Tester
from utils import CostType
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def main():
    run = Path(sys.argv[1]).resolve()
    cfg = OmegaConf.load(run / '.hydra/config.yaml').c
    device = select_device([0])
    qtz.init_quartz_context(cfg.gate_set, str(root/'experiment/ecc_set/nam_325_ecc.json'), False, True)
    keys = ('gnn_type', 'num_gate_types', 'gate_type_embed_dim', 'gnn_num_layers',
            'gnn_hidden_dim', 'gnn_output_dim', 'gin_num_mlp_layers', 'gin_learn_eps',
            'gin_neighbor_pooling_type', 'gin_graph_pooling_type',
            'actor_hidden_size', 'critic_hidden_size')
    model = ActorCritic(**{k: cfg[k] for k in keys},
                        action_dim=qtz.quartz_context.num_xfers, device=device).to(device)
    checkpoint = torch.load(run/'ckpts/iter_1.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    original_qasm = (root/'experiment/circs/nam_circs/barenco_tof_3.qasm').read_text()
    graph = qtz.qasm_to_graph(original_qasm)
    output = run/'search_check'
    output.mkdir(exist_ok=True)
    tester = Tester(cost_type=CostType.from_str('gate_count'), ac_net=model,
                    device=device, output_dir=str(output), sync_tuning_dir=False,
                    hit_rate=cfg.hit_rate, batch_size=4, max_loss_tolerance=0.1,
                    max_search_sec=10, vmem_perct_limit=cfg.vmem_perct_limit)
    with torch.no_grad():
        best = tester.random_search('barenco_tof_3', graph)
    if best is None:
        best = graph
    equivalent = Operator(QuantumCircuit.from_qasm_str(original_qasm)).equiv(
        Operator(QuantumCircuit.from_qasm_str(best.to_qasm_str())))
    assert equivalent
    result = {'input_gates': graph.gate_count, 'best_gates': best.gate_count,
              'unitary_equivalent': equivalent, 'checkpoint_loaded': True}
    (output/'best.qasm').write_text(best.to_qasm_str())
    (output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
