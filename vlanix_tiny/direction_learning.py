"""P0.11O: matched gradient mixture versus one-sided projection."""
import json
import torch
from vlanix_tiny.protected_learning import run

if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(dict(schema='ngc-direction-learning-v1',
        runs=[run(seed,('mixed','projected')) for seed in (42,43)],
        torch_version=torch.__version__,threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        claim='gradient projection before AdamW, exact local argmax postcheck'),sort_keys=True))
