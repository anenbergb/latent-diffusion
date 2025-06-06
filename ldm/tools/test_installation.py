import torch
import triton
import triton.language as tl
from xformers.ops import memory_efficient_attention

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Create test tensors
n = 1024
x = torch.rand(n, device='cuda')
y = torch.rand(n, device='cuda')
output = torch.empty_like(x)

# Launch kernel
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)

# Verify result
if torch.allclose(output, x + y):
    print("✅ Triton kernel executed successfully!")
else:
    print("❌ Output mismatch.")



# Dummy tensors
q = torch.randn(1, 128, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 128, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 128, 64, device="cuda", dtype=torch.float16)

# Run attention
output = memory_efficient_attention(q, k, v)

print("✅ xformers memory-efficient attention ran successfully:", output.shape)
