import torch

# %% [1. Introduction to Autograd]
# Autograd is PyTorch's automatic differentiation engine. It builds a computational graph
# to track operations on tensors with requires_grad=True, allowing gradient computation
# for optimization (e.g., in neural networks).

# Simple example: y = x^2, compute dy/dx
x = torch.tensor(3.0, requires_grad=True)  # Track gradients for x
y = x ** 2  # Forward pass: y = x^2
y.backward()  # Compute gradients (dy/dx = 2x)
print("Gradient of y = x^2 at x=3:", x.grad)  # Should be 2 * 3 = 6

# %% [2. Dummy Dataset for Linear Regression]
# Creating a synthetic dataset for linear regression: y = 2x + 1 + noise
# Features: x (input), Labels: y (output)
torch.manual_seed(42)  # For reproducibility
x_data = torch.linspace(0, 10, 100).reshape(-1, 1)  # 100 points from 0 to 10
true_w = 2.0  # True weight
true_b = 1.0  # True bias
noise = torch.randn(100, 1) * 0.5  # Gaussian noise
y_data = true_w * x_data + true_b + noise  # Linear relation with noise

print("\nDummy Dataset:")
print("x_data (first 5):", x_data[:5].flatten().tolist())
print("y_data (first 5):", y_data[:5].flatten().tolist())

# %% [3. Enabling Gradient Tracking]
# Tensors with requires_grad=True track operations for gradient computation.

# Define model parameters (weight and bias)
w = torch.tensor(0.0, requires_grad=True)  # Initial weight
b = torch.tensor(0.0, requires_grad=True)  # Initial bias

# Forward pass: y_pred = w * x + b
y_pred = w * x_data + b

# Compute mean squared error (MSE) loss
loss = torch.mean((y_pred - y_data) ** 2)
print("\nInitial Loss:", loss.item())

# Compute gradients
loss.backward()

# Check gradients
print("Gradient of loss w.r.t. w:", w.grad)
print("Gradient of loss w.r.t. b:", b.grad)

# %% [4. Gradient Accumulation]
# Gradients accumulate in the .grad attribute. They must be zeroed before the next backward pass.

# Perform another forward and backward pass without zeroing gradients
y_pred = w * x_data + b
loss = torch.mean((y_pred - y_data) ** 2)
loss.backward()  # Gradients accumulate
print("\nAccumulated Gradient of w:", w.grad)  # Gradients are summed

# Zero gradients
w.grad.zero_()
b.grad.zero_()
print("After zeroing, Gradient of w:", w.grad)
print("After zeroing, Gradient of b:", b.grad)

# New backward pass with zeroed gradients
y_pred = w * x_data + b
loss = torch.mean((y_pred - y_data) ** 2)
loss.backward()
print("New Gradient of w:", w.grad)
print("New Gradient of b:", b.grad)

# %% [5. Detaching Tensors]
# detach() creates a new tensor that doesn't track gradients, useful for intermediate results.

# Example: Use y_pred for visualization without affecting gradients
y_pred_detached = y_pred.detach()  # No gradient tracking
print("\nDetached y_pred (first 5):", y_pred_detached[:5].flatten().tolist())
print("Is detached tensor tracked?", y_pred_detached.requires_grad)  # False

# Detached tensor can be used in operations without building a computational graph
detached_loss = torch.mean((y_pred_detached - y_data) ** 2)
print("Loss with detached tensor:", detached_loss.item())

# %% [6. Disabling Gradient Tracking]
# torch.no_grad() temporarily disables gradient computation, useful for inference.

# Compute loss without gradient tracking
with torch.no_grad():
    y_pred_no_grad = w * x_data + b
    loss_no_grad = torch.mean((y_pred_no_grad - y_data) ** 2)
print("\nLoss without gradient tracking:", loss_no_grad.item())

# Verify no gradients are computed
try:
    loss_no_grad.backward()  # Will raise an error
except RuntimeError as e:
    print("Error with no_grad:", str(e))

# %% [7. Gradient Manipulation]
# Gradients can be manipulated (e.g., zeroed, clipped) for optimization stability.

# Simulate a training step
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.01

# Training loop for 3 iterations
print("\nTraining for 3 iterations:")
for i in range(3):
    # Forward pass
    y_pred = w * x_data + b
    loss = torch.mean((y_pred - y_data) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Gradient descent update (manually)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Zero gradients
    w.grad.zero_()
    b.grad.zero_()
    
    print(f"Iteration {i+1}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

# %% [8. Practical Example: Gradient Clipping]
# Gradient clipping prevents exploding gradients by limiting their magnitude.

# Simulate large gradients
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y_pred = w * x_data + b
loss = torch.mean((y_pred - y_data) ** 2) * 1000  # Amplify loss to get large gradients
loss.backward()

print("\nBefore Clipping, Gradient of w:", w.grad)
print("Before Clipping, Gradient of b:", b.grad)

# Clip gradients (max norm = 1.0)
torch.nn.utils.clip_grad_norm_([w, b], max_norm=1.0)

print("After Clipping, Gradient of w:", w.grad)
print("After Clipping, Gradient of b:", b.grad)