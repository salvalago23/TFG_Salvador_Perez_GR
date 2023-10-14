
import torch
 
print("Is CUDA supported by this system?", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
 
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print("ID of current CUDA device:", torch.cuda.current_device())
print("Name of current CUDA device:", torch.cuda.get_device_name(cuda_id))