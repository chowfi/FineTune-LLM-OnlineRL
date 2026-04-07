"""
Run : torchrun --nproc_per_node=2 FSDP.py --mixed-precision
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, get_model_state_dict, StateDictOptions

parser = argparse.ArgumentParser()
parser.add_argument("--mixed-precision", action="store_true")
args = parser.parse_args()

# Read local_rank, rank, world_size from env 
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Initialize communication between devices
dist.init_process_group(backend="gloo", init_method="env://")
mesh = init_device_mesh("cpu", mesh_shape=(world_size,))

if args.mixed_precision:
    fsdp_kwargs = {
    "mp_policy": MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
}

d_model = 64
model = nn.Transformer(
    d_model=d_model,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=d_model * 4,
    batch_first=True,
)

for layer in model.encoder.layers:
    fully_shard(layer, mesh=mesh, **fsdp_kwargs)
for layer in model.decoder.layers:
    fully_shard(layer, mesh=mesh, **fsdp_kwargs)
fully_shard(model, mesh=mesh, **fsdp_kwargs)

# Sharded parameters stay float32 under MixedPrecisionPolicy (compute uses bf16 after unshard).
for param in model.parameters():
    if rank == 0:
        print(model)
        print("local shard shape:", param.to_local().shape)
        assert param.dtype == torch.float32
    full = param.full_tensor()
    if rank == 0:
        print("global shape:", full.shape)
    break

if args.mixed_precision:
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()

# model_state_dict = get_model_state_dict(
#     model=model,
#     options=StateDictOptions(
#         full_state_dict=True,
#         cpu_offload=True,
#     )
# )
# torch.save(model_state_dict, "model_state_dict.pt")

full_sd = torch.load(
    "model_state_dict.pt",
    mmap=True,
    weights_only=True,
    map_location='cpu',
)

set_model_state_dict(
    model=model,
    model_state_dict=full_sd,
    options=StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
    ),
)

dist.destroy_process_group()
