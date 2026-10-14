# quick environment check: verifies that pytorch can see the gpu and runs a
# small timed benchmark of simple elementwise operations on it.
# requires a cuda gpu (it fails on a cpu-only machine).

import torch
import time

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda")
# a batch of 64 random rgb images allocated directly on the gpu
x = torch.randn(64, 3, 224, 224, device=device)
# warm-up
for _ in range(20):
    y = x * 2.0

# wait for the warm-up kernels to finish before timing
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    y = x * 2.0

# cuda calls are asynchronous, so synchronize before stopping the timer
torch.cuda.synchronize()
end = time.time()
print("1000 GPU operations:", end - start, "seconds")
