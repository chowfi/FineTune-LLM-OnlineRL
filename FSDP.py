import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import fully_shard, FSDPModule
import os

# TODO: Read local_rank, rank, world_size from env ("torchrun --nproc_per_node=2 FSDP.py")
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# TODO: Initialize communication between devices
dist.init_process_group(backend="gloo", init_method="env://")
mesh = init_device_mesh("cpu", mesh_shape=(world_size,))

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
    fully_shard(layer, mesh=mesh)
for layer in model.decoder.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

if rank ==0:
    print(model)

dist.destroy_process_group()
