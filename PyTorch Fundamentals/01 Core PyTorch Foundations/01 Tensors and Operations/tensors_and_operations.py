import torch
import numpy as np

# %% [1. Introduction to Tensors]
# Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with GPU support.
# They can represent scalars (0D), vectors (1D), matrices (2D), or higher-dimensional data (nD).

# Creating a scalar tensor
scalar = torch.tensor(42)
print("Scalar Tensor:", scalar)
print("Scalar Shape:", scalar.shape)  # Empty shape for 0D tensor
print("Scalar Dimension:", scalar.dim())  # 0 dimensions

# Creating a vector (1D tensor)
vector = torch.tensor([1, 2, 3, 4])
print("\nVector Tensor:", vector)
print("Vector Shape:", vector.shape)  # (4,)
print("Vector Dimension:", vector.dim())  # 1 dimension

# Creating a matrix (2D tensor)
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\nMatrix Tensor:", matrix)
print("Matrix Shape:", matrix.shape)  # (2, 3)
print("Matrix Dimension:", matrix.dim())  # 2 dimensions

# %% [2. Tensor Creation Methods]
# PyTorch provides various methods to create tensors with specific values.

# Tensor of zeros
zeros_tensor = torch.zeros(2, 3)  # 2x3 tensor filled with zeros
print("\nZeros Tensor:\n", zeros_tensor)

# Tensor of ones
ones_tensor = torch.ones(2, 3)  # 2x3 tensor filled with ones
print("\nOnes Tensor:\n", ones_tensor)

# Tensor of random values (from normal distribution)
random_tensor = torch.randn(2, 3)  # 2x3 tensor with random values
print("\nRandom Tensor:\n", random_tensor)

# Tensor from a range
range_tensor = torch.arange(0, 6, 1).reshape(2, 3)  # 2x3 tensor from 0 to 5
print("\nRange Tensor:\n", range_tensor)

# %% [3. Tensor Attributes]
# Tensors have attributes like shape, data type (dtype), and device (CPU or GPU).

# Creating a sample tensor
sample_tensor = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.float32)

# Shape
print("\nSample Tensor:\n", sample_tensor)
print("Shape:", sample_tensor.shape)  # (2, 3)
print("Size (alias for shape):", sample_tensor.size())  # Same as shape

# Data type
print("Data Type:", sample_tensor.dtype)  # torch.float32

# Device
print("Device:", sample_tensor.device)  # cpu (or cuda if GPU is available)

# Number of elements
print("Number of Elements:", sample_tensor.numel())  # 6

# %% [4. Dummy Dataset]
# Creating a dummy dataset: 3 "images" of size 4x4 (e.g., grayscale pixel intensities).
# Values are between 0 and 255, simulating pixel data.

dummy_images = torch.randint(0, 256, (3, 4, 4), dtype=torch.float32)  # 3 images, 4x4
print("\nDummy Dataset (3 images, 4x4):\n", dummy_images)

# Accessing a single image
single_image = dummy_images[0]  # First image
print("\nSingle Image (4x4):\n", single_image)

# %% [5. Tensor Operations: Indexing and Slicing]
# Indexing and slicing work similarly to NumPy arrays.

# Indexing
pixel_value = single_image[0, 0]  # Top-left pixel of the first image
print("\nTop-left Pixel Value:", pixel_value)

# Slicing
row_slice = single_image[0, :]  # First row of the first image
print("First Row of Single Image:", row_slice)

# Modifying a tensor via indexing
single_image[0, 0] = 255.0  # Set top-left pixel to max intensity
print("\nModified Single Image:\n", single_image)

# %% [6. Tensor Operations: Reshaping]
# Reshaping changes the tensor's shape without altering its data.

# Flattening the image (4x4 -> 16)
flattened_image = single_image.view(-1)  # or single_image.reshape(-1)
print("\nFlattened Image (16 elements):\n", flattened_image)

# Reshaping to 2x8
reshaped_image = single_image.view(2, 8)
print("\nReshaped Image (2x8):\n", reshaped_image)

# Ensure the total number of elements matches
try:
    invalid_reshape = single_image.view(3, 5)  # Will raise an error (3*5 != 16)
except RuntimeError as e:
    print("\nError in Reshaping:", str(e))

# %% [7. Tensor Operations: Element-wise Operations]
# Element-wise operations apply to each element independently.

# Adding a constant to all pixels
brightened_image = single_image + 50.0
print("\nBrightened Image (added 50):\n", brightened_image)

# Multiplying by a constant
scaled_image = single_image * 0.5
print("\nScaled Image (multiplied by 0.5):\n", scaled_image)

# Element-wise operations between tensors
another_image = torch.ones_like(single_image) * 100.0
combined_image = single_image + another_image
print("\nCombined Image (element-wise addition):\n", combined_image)

# %% [8. Tensor Operations: Matrix Operations]
# Matrix operations like dot products and matrix multiplication.

# Matrix multiplication
matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
matmul_result = torch.matmul(matrix_a, matrix_b)  # or matrix_a @ matrix_b
print("\nMatrix Multiplication (A @ B):\n", matmul_result)

# Transpose
transposed = matrix_a.t()
print("\nTransposed Matrix A:\n", transposed)

# Dot product (for 1D tensors)
vector_a = torch.tensor([1, 2], dtype=torch.float32)
vector_b = torch.tensor([3, 4], dtype=torch.float32)
dot_product = torch.dot(vector_a, vector_b)
print("\nDot Product:", dot_product)

# %% [9. Tensor Operations: Broadcasting]
# Broadcasting allows operations on tensors with compatible shapes.

# Adding a 1D tensor to a 2D tensor
row_vector = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
result_broadcast = single_image + row_vector  # Broadcasts row_vector to each row
print("\nBroadcasting Result (add row vector to image):\n", result_broadcast)

# Broadcasting rules: Shapes must be compatible (e.g., (4,4) and (4,) work)
try:
    invalid_vector = torch.tensor([1, 2], dtype=torch.float32)
    single_image + invalid_vector  # Will raise an error
except RuntimeError as e:
    print("\nError in Broadcasting:", str(e))

# %% [10. Converting between NumPy and PyTorch Tensors]
# PyTorch tensors can be converted to/from NumPy arrays.

# Tensor to NumPy
numpy_array = single_image.numpy()
print("\nTensor to NumPy Array:\n", numpy_array)

# NumPy to Tensor
numpy_array = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
tensor_from_numpy = torch.from_numpy(numpy_array)
print("\nNumPy to Tensor:\n", tensor_from_numpy)

# Note: NumPy arrays are CPU-only; ensure tensor is on CPU before conversion
if torch.cuda.is_available():
    gpu_tensor = single_image.to("cuda")
    try:
        gpu_tensor.numpy()  # Will raise an error
    except TypeError as e:
        print("\nError Converting GPU Tensor to NumPy:", str(e))

# %% [11. In-place Operations]
# In-place operations modify the tensor directly and end with `_`.

# In-place addition
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
tensor.add_(10)  # Adds 10 to each element in-place
print("\nIn-place Addition:\n", tensor)

# In-place multiplication
tensor.mul_(2)  # Multiplies each element by 2 in-place
print("\nIn-place Multiplication:\n", tensor)

# %% [12. Moving Tensors between CPU and GPU]
# Tensors can be moved to GPU for faster computation if available.

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)

# Move tensor to device
tensor_on_device = single_image.to(device)
print("Tensor Device:", tensor_on_device.device)

# Perform operation on device
if device.type == "cuda":
    result = tensor_on_device * 2
    print("\nOperation on GPU:\n", result)
else:
    print("\nGPU not available, operation performed on CPU.")

# Move back to CPU (if on GPU)
tensor_back_to_cpu = tensor_on_device.to("cpu")
print("Tensor Back on CPU:", tensor_back_to_cpu.device)