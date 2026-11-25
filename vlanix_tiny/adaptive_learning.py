"""P0.11N: same foundation and batch order, guarded vs adaptive guarded."""
import json
import torch
from vlanix_tiny.protected_learning import run

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(dict(schema='ngc-adaptive-learning-v1',
        runs=[run(seed,('guarded','adaptive')) for seed in (42,43)],
        torch_version=torch.__version__,threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        claim='matched batch budget, bounded retry compute, exact local argmax guard'),sort_keys=True))
