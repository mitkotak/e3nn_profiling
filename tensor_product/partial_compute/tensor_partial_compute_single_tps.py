# e3nn PyTorch

import torch
from torch.utils.benchmark import Timer

from e3nn import o3
from e3nn.util.jit import compile

device = "cuda" if torch.cuda.is_available() else "cpu"


# Constants
irreps_string = "1e"
batch = 10

# Training params
n = 1000
warmup = -1
irreps_in1 = o3.Irreps(irreps_string)
irreps_in2 = o3.Irreps(irreps_string)
irreps_out = o3.Irreps("1e")
irreps_out1 = o3.Irreps("2e")
tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out,
                                    _specialized_code=False,
                                    _optimize_einsums=True)
tp1 = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out1,
                                     _specialized_code=False,
                                     _optimize_einsums=True)
for tp_idx in [tp, tp1]:
    tp_idx = tp_idx.to(device=device)
    print(f"Tensor product: {tp_idx}")
    print("Instructions:")
    for ins in tp_idx.instructions:
        print(f" {ins}")

tp = compile(tp)
t1 = compile(tp1)

inputs = iter(
        [
            (irreps_in1.randn(batch, -1).to(device=device),
             irreps_in2.randn(batch, -1).to(device=device))
            for _ in range(n + warmup)
        ]
    )

# warmup

for _ in range(20):
    tp.zero_grad()
    out = tp(*next(inputs))
    out.tanh().sum().backward()
    tp1.zero_grad()
    out1 = tp1(*next(inputs))
    out1.tanh().sum().backward()
    

with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile/e3nn_pytorch/tensor_partial_compute_single_tps.trace.json'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    tp.zero_grad()
    out = tp(*next(inputs))
    out.tanh().sum().backward()
    tp1.zero_grad()
    out1 = tp1(*next(inputs))
    out1.tanh().sum().backward()

