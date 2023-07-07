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
                                    _specialized_code=True,
                                    _optimize_einsums=True)
tp1 = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out1,
                                     _specialized_code=True,
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

t = Timer(
    stmt=("tp.zero_grad()\n" "out = tp(*next(inputs))\n"
          "out.tanh().sum().backward()\n"
          "tp1.zero_grad()\n" "out1 = tp1(*next(inputs))\n"
          "out1.tanh().sum().backward()\n"),
    globals={"tp": tp, "tp1": tp1, "inputs": inputs},
    )

t.timeit(20)
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("profiling")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
    perloop = t.timeit(1)
prof.export_chrome_trace("trace.json")
torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()

