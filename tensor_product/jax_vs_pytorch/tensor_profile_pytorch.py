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
irreps_out = o3.Irreps(irreps_string)
tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out,
                                    _specialized_code=True,
                                    _optimize_einsums=True)
tp = tp.to(device=device)
tp = compile(tp)


inputs = iter(
        [
            (irreps_in1.randn(batch, -1).to(device=device),
             irreps_in2.randn(batch, -1).to(device=device))
            for _ in range(n + warmup)
        ]
    )

t = Timer(
    stmt=("tp.zero_grad()\n" "out = tp(*next(inputs))\n" "out.tanh().sum().backward()\n"),
        globals={"tp": tp, "inputs": inputs},
    )

torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("profiling")
perloop = t.timeit(1)
torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()

