import torch
import numpy as np

# Initializing a Tensor - Directly from data
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(x_data)

# Initializing a Tensor - From a numpy array
np_array = np.array(data); np_array
x_np = torch.tensor(np_array); x_np

# Initializing a Tensor - From another tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float32)
print(x_rand)

# with random or constant values
shape = (2,3,) # It is a tuple of tensor dimensions
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

py_tensor = torch.rand(3,4)
print(f'Shape of tensor: {py_tensor.shape}')
print(f'Datatype of tensor: {py_tensor.dtype}')
print(f'Device tensor is stored on: {py_tensor.device}')

if torch.cuda.is_available():
    print("GPU is available")
    py_tensor = py_tensor.to('cuda')
else:
    print("GPU is not available")

tensor_ones = torch.ones(4,4); tensor_ones
print(f'First row: {tensor_ones[0]}')
print(f'First column: {tensor_ones[:, 0]}')
print(f'Last column: {tensor_ones[:, -1]}')
tensor_ones[:, 1] = 0; tensor_ones

t1 = torch.cat([tensor_ones, tensor_ones, tensor_ones], dim=1)
print(t1)

# Arithmetic operations - Matrix multiplication
y1 = tensor_ones@tensor_ones.T; print(y1)
y2 = tensor_ones.matmul(tensor_ones.T);print(y2)
y3 = torch.rand_like(tensor_ones);print(y3)
torch.matmul(tensor_ones, tensor_ones.T, out=y3)

# Element wise product
z1 = tensor_ones * tensor_ones; print(z1)
z2 = tensor_ones.mul(tensor_ones);print(z2)
z3 = torch.rand_like(tensor_ones)
torch.mul(tensor_ones, tensor_ones, out=z3)

# Single element tensors
agg = tensor_ones.sum(); print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations
print(tensor_ones, "\n")
tensor_ones.add_(5)
print(tensor_ones)

# Bridge with Numpy - # tensor to numpy array
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

t.add_(1)   # A change in tensor reflects in Numpy array
print(f't: {t}')
print(f'n: {n}')


# Bridge with Numpy - # Numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)
print(f'n:{n}')
print(f't:{t}')

np.add(n,1,out=n)   # A change in NumPy array reflects in tensor
print(f't: {t}')
print(f'n: {n}')

