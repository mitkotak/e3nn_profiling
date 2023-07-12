import site
site.addsitedir('/home/mkotak/atomic_architects/venv/jax_env/lib/python3.10/site-packages')
import time
import haiku as hk
import jax
import jax.numpy as jnp
import jaxlib

import e3nn_jax
from e3nn_jax._src.utils.jit import jit_code

# Constants
irreps_string = "1e"
batch = 10

# Training params
n = 1000
warmup = -1

irreps_in1_jax = e3nn_jax.Irreps(irreps_string)
irreps_in2_jax = e3nn_jax.Irreps(irreps_string)
irreps_out_jax = e3nn_jax.Irreps(irreps_string)

def k():
    k.key, x = jax.random.split(k.key)
    return x

k.key = jax.random.PRNGKey(0)

@hk.without_apply_rng
@hk.transform
def tp_jax(x1, x2):
    return e3nn_jax.haiku.FullyConnectedTensorProduct(irreps_out_jax)(x1, x2)

inputs_jax = (e3nn_jax.normal(irreps_in1_jax, k(), (batch,)),
              e3nn_jax.normal(irreps_in2_jax, k(), (batch,)))

w = tp_jax.init(k(), *inputs_jax)
w, inputs_jax = jax.tree_util.tree_map(jax.device_put, (w, inputs_jax))

f  = tp_jax.apply
f2 = f
f = jax.value_and_grad(
    lambda w, x1, x2: sum(jnp.sum(jnp.tanh(out)) for out in
                          jax.tree_util.tree_leaves(f2(w, x1, x2)))
)

f = jax.jit(f)

for _ in range(20):
    z = f(w, *inputs_jax)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)


with jax.profiler.trace("./profile/e3nn-jax"):
    z = f(w, *inputs_jax)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), z)

# with open("xla.txt", "wt") as file:
#     file.write(jit_code(f, w, *inputs_jax))
