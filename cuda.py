import torch

print(torch.cuda.memory_summary(device=None, abbreviated=False))

print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())  # Should return the current device ID (e.g., 0)
print(torch.cuda.device_count())  # Should return the number of GPUs available
print(torch.cuda.get_device_name(0))  # Should return the name of the GPU

