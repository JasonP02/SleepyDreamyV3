import torch
import torch.nn.functional as F

# Example tensor
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print("Original tensor:")
print(x)
print(f"Sum along dim=1: {x.sum(dim=1)}")
print()

# Method 1: Using torch.softmax()
print("Method 1: Using torch.softmax()")
softmax_x = torch.softmax(x, dim=1)
print(softmax_x)
print(f"Sum along dim=1: {softmax_x.sum(dim=1)}")
print()

# Method 2: Using torch.nn.functional.softmax()
print("Method 2: Using torch.nn.functional.softmax()")
softmax_x_f = F.softmax(x, dim=1)
print(softmax_x_f)
print(f"Sum along dim=1: {softmax_x_f.sum(dim=1)}")
print()

# Method 3: Manual normalization (for comparison)
print("Method 3: Manual normalization")
manual_x = x / x.sum(dim=1, keepdim=True)
print(manual_x)
print(f"Sum along dim=1: {manual_x.sum(dim=1)}")
print()

# Example with different dimensions
print("Example with 3D tensor:")
x_3d = torch.randn(2, 3, 4)
print(f"Original shape: {x_3d.shape}")
print(f"Sum along dim=2: {x_3d.sum(dim=2)}")

softmax_3d = torch.softmax(x_3d, dim=2)
print(f"After softmax along dim=2:")
print(f"Sum along dim=2: {softmax_3d.sum(dim=2)}")
print()

# Note: Softmax vs manual normalization
print("Note on Softmax vs Manual Normalization:")
print("- Softmax applies exponential function first: exp(x_i) / sum(exp(x_j))")
print("- Manual normalization directly divides by sum: x_i / sum(x_j)")
print("- Softmax is more common for probability distributions in ML")
print("- Use softmax for logits/activations, manual for raw values that should sum to 1")